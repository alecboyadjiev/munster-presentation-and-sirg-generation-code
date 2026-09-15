"""Shared sampling and validation for SIRG vertex processes.

Expressions intentionally use the same small NumPy-only evaluation environment as
the pre-existing weight-quantile and connection-kernel command-line options.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


EXPRESSION_GRID_SIZE = 257
QUANTILE_GRID_SIZE = 8193
PERIODICITY_RTOL = 1e-7
PERIODICITY_ATOL = 1e-9
INTEGER_ATOL = 1e-10
AUTOMATIC_BOUND_FACTOR = 1.01


ArrayFunction = Callable[..., np.ndarray]


@dataclass(frozen=True)
class VertexSample:
    """A sampled vertex process in Euclidean coordinates.

    ``points`` has one column per coordinate.  For the p-adic generator that
    coordinate is the measure-preserving base-p coordinate in ``[0, 1)``; the
    caller converts it to p-adic digits after sampling.
    """

    points: np.ndarray
    description: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite, positive number")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def evaluate_kernel(
    expression: str,
    distances: np.ndarray,
    weights: np.ndarray,
    other_weights: np.ndarray,
) -> np.ndarray:
    kernel = _compile_expression(expression, ("t", "w", "wp"), "--kernel")
    probabilities = _real_broadcast_values(
        kernel,
        (distances, weights, other_weights),
        distances.shape,
        "--kernel",
    )
    return np.clip(probabilities, 0.0, 1.0)


def _compile_expression(
    expression: str,
    variables: Sequence[str],
    option_name: str,
    constants: dict[str, float] | None = None,
) -> ArrayFunction:
    namespace: dict[str, object] = {"np": np, "__builtins__": {}}
    if constants:
        namespace.update(constants)
    try:
        return eval(f"lambda {','.join(variables)}: {expression}", namespace)
    except Exception as exc:
        raise ValueError(f"could not parse {option_name}: {exc}") from exc


def _real_broadcast_values(
    function: ArrayFunction,
    arguments: Sequence[np.ndarray],
    shape: tuple[int, ...],
    option_name: str,
) -> np.ndarray:
    try:
        raw_values = np.asarray(function(*arguments))
        if np.iscomplexobj(raw_values):
            raise ValueError("values must be real")
        values = np.asarray(raw_values, dtype=float)
        values = np.broadcast_to(values, shape).copy()
    except Exception as exc:
        raise ValueError(f"could not evaluate {option_name}: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{option_name} must produce finite real values")
    return values


def compile_quantile(expression: str, option_name: str = "--quantile") -> ArrayFunction:
    quantile = _compile_expression(expression, ("u",), option_name)
    epsilon = np.finfo(float).eps
    test_u = np.linspace(epsilon, 1.0 - epsilon, QUANTILE_GRID_SIZE)
    test_values = quantile_values(quantile, test_u, option_name)
    scale = np.maximum(
        1.0, np.maximum(np.abs(test_values[:-1]), np.abs(test_values[1:]))
    )
    if np.any(np.diff(test_values) < -1e-12 * scale):
        raise ValueError(f"{option_name} must be nondecreasing on u in (0, 1)")
    return quantile


def quantile_values(
    quantile: ArrayFunction,
    uniforms: np.ndarray,
    option_name: str = "--quantile",
) -> np.ndarray:
    return _real_broadcast_values(quantile, (uniforms,), uniforms.shape, option_name)


def compile_count_quantile(expression: str) -> ArrayFunction:
    """Compile an inverse CDF for a nonnegative integer-valued random count."""

    option_name = "--count-quantile"
    quantile = compile_quantile(expression, option_name)
    epsilon = np.finfo(float).eps
    test_u = np.linspace(epsilon, 1.0 - epsilon, QUANTILE_GRID_SIZE)
    values = quantile_values(quantile, test_u, option_name)
    nearest = np.rint(values)
    tolerance = INTEGER_ATOL * np.maximum(1.0, np.abs(values))
    if np.any(values < 0) or np.any(np.abs(values - nearest) > tolerance):
        raise ValueError(
            "--count-quantile must be a nonnegative integer-valued (discrete) "
            "quantile on u in (0, 1)"
        )
    return quantile


def sample_count_quantile(quantile: ArrayFunction, rng: np.random.Generator) -> int:
    epsilon = np.finfo(float).eps
    uniform = np.array([np.clip(rng.random(), epsilon, 1.0 - epsilon)])
    value = quantile_values(quantile, uniform, "--count-quantile")[0]
    nearest = round(float(value))
    if value < 0 or not np.isclose(
        value,
        nearest,
        rtol=0.0,
        atol=INTEGER_ATOL * max(1.0, abs(value)),
    ):
        raise ValueError("--count-quantile produced an invalid nonnegative-integer count")
    return int(nearest)


@dataclass(frozen=True)
class PeriodicIntensity:
    function: ArrayFunction
    dimension: int
    period: float
    maximum: float

    def values(self, points: np.ndarray, phase: np.ndarray) -> np.ndarray:
        shifted = np.mod(points + phase, self.period)
        arguments = tuple(shifted[:, axis] for axis in range(self.dimension))
        values = _real_broadcast_values(
            self.function, arguments, (len(points),), "--cox-function"
        )
        if np.any(values < 0):
            raise ValueError("--cox-function must be nonnegative")
        if np.any(values > self.maximum * (1.0 + 1e-10) + 1e-12):
            raise ValueError(
                "--cox-function exceeded the numerically established thinning "
                "bound; use a smoother function or increase its sampled support"
            )
        return values


def compile_periodic_intensity(
    expression: str,
    period: float,
    variables: Sequence[str],
    upper_bound: float | None = None,
) -> PeriodicIntensity:
    """Compile and numerically validate a nonnegative periodic multiplier.

    Periodicity of a command-line numerical expression cannot be proven in
    general.  We therefore check each coordinate on a deterministic dense grid,
    including a half-cell offset grid to catch common discontinuities and narrow
    features.  Candidate points are checked against the resulting thinning bound
    again during sampling.
    """

    if not np.isfinite(period) or period <= 0:
        raise ValueError("--period must be a finite, positive number")
    if upper_bound is not None and (
        not np.isfinite(upper_bound) or upper_bound <= 0
    ):
        raise ValueError("--cox-bound must be a finite, positive number")

    dimension = len(variables)
    if dimension not in (1, 2):
        raise ValueError("periodic intensities support one or two dimensions")
    function = _compile_expression(
        expression,
        variables,
        "--cox-function",
        constants={"period": period},
    )

    axis = np.linspace(0.0, period, EXPRESSION_GRID_SIZE, endpoint=False)
    shifted_axis = np.mod(axis + period / (2 * EXPRESSION_GRID_SIZE), period)
    grids: list[tuple[np.ndarray, ...]] = []
    for sampled_axis in (axis, shifted_axis):
        if dimension == 1:
            grids.append((sampled_axis,))
        else:
            grids.append(tuple(np.meshgrid(sampled_axis, sampled_axis, indexing="ij")))

    sampled_values = []
    for arguments in grids:
        shape = arguments[0].shape
        base = _real_broadcast_values(function, arguments, shape, "--cox-function")
        if np.any(base < 0):
            raise ValueError("--cox-function must be nonnegative")
        sampled_values.append(base)

        for coordinate in range(dimension):
            translated = list(arguments)
            translated[coordinate] = translated[coordinate] + period
            periodic = _real_broadcast_values(
                function, translated, shape, "--cox-function"
            )
            scale = max(1.0, float(np.max(np.abs(base))), float(np.max(np.abs(periodic))))
            if not np.allclose(
                base,
                periodic,
                rtol=PERIODICITY_RTOL,
                atol=PERIODICITY_ATOL * scale,
            ):
                variable = variables[coordinate]
                raise ValueError(
                    f"--cox-function is not periodic with --period {period:g} "
                    f"in {variable}"
                )

    sampled_maximum = max(float(np.max(values)) for values in sampled_values)
    if upper_bound is not None:
        if sampled_maximum > upper_bound * (1.0 + 1e-10) + 1e-12:
            raise ValueError("--cox-bound is smaller than --cox-function")
        maximum = upper_bound
    else:
        # The grid maximum is an estimate.  A modest cushion avoids false
        # failures around smooth off-grid maxima; runtime evaluation still
        # rejects an envelope that the function actually exceeds.
        maximum = sampled_maximum * AUTOMATIC_BOUND_FACTOR
    return PeriodicIntensity(function, dimension, period, maximum)


def sample_vertex_process(
    *,
    process: str,
    rng: np.random.Generator,
    lower: Sequence[float],
    upper: Sequence[float],
    intensity: float | None,
    cox_function: str | None,
    period: float | None,
    cox_bound: float | None,
    count_quantile: str | None,
    variables: Sequence[str],
) -> VertexSample:
    """Sample a supported point process in a rectangular coordinate window."""

    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    dimension = len(lower_array)
    if upper_array.shape != lower_array.shape or np.any(upper_array <= lower_array):
        raise ValueError("vertex-process window must have positive side lengths")
    if len(variables) != dimension:
        raise ValueError("one Cox variable name is required per point coordinate")
    volume = float(np.prod(upper_array - lower_array))
    if intensity is not None and (not np.isfinite(intensity) or intensity < 0):
        raise ValueError("--intensity must be a finite, nonnegative number")

    if process == "poisson":
        if intensity is None:
            raise ValueError("--intensity is required for --vertex-process poisson")
        if (
            cox_function is not None
            or period is not None
            or cox_bound is not None
            or count_quantile is not None
        ):
            raise ValueError(
                "Cox and mixed-binomial options only apply to "
                "their corresponding vertex processes"
            )
        count = int(rng.poisson(intensity * volume))
        points = rng.uniform(lower_array, upper_array, size=(count, dimension))
        return VertexSample(points, f"poisson(intensity={intensity:g})")

    if process == "mixed-binomial":
        if count_quantile is None:
            raise ValueError(
                "--count-quantile is required for --vertex-process mixed-binomial"
            )
        if cox_function is not None or period is not None or cox_bound is not None:
            raise ValueError(
                "--cox-function, --period, and --cox-bound only apply to "
                "--vertex-process cox"
            )
        if intensity is not None:
            raise ValueError(
                "--intensity does not apply to --vertex-process mixed-binomial; "
                "the count is determined by --count-quantile"
            )
        quantile = compile_count_quantile(count_quantile)
        count = sample_count_quantile(quantile, rng)
        points = rng.uniform(lower_array, upper_array, size=(count, dimension))
        return VertexSample(points, f"mixed-binomial(count={count})")

    if process == "cox":
        if intensity is None:
            raise ValueError("--intensity is required for --vertex-process cox")
        if cox_function is None or period is None:
            raise ValueError(
                "--cox-function and --period are required for --vertex-process cox"
            )
        if count_quantile is not None:
            raise ValueError(
                "--count-quantile only applies to --vertex-process mixed-binomial"
            )
        periodic = compile_periodic_intensity(
            cox_function, period, variables, upper_bound=cox_bound
        )
        phase = rng.uniform(0.0, period, size=dimension)
        if periodic.maximum == 0.0 or intensity == 0.0:
            return VertexSample(
                np.empty((0, dimension)),
                f"cox(period={period:g}, phase={_format_phase(phase)})",
            )

        candidate_count = int(rng.poisson(intensity * periodic.maximum * volume))
        candidates = rng.uniform(
            lower_array, upper_array, size=(candidate_count, dimension)
        )
        multipliers = periodic.values(candidates, phase)
        accepted = rng.random(candidate_count) < multipliers / periodic.maximum
        return VertexSample(
            candidates[accepted],
            f"cox(period={period:g}, phase={_format_phase(phase)})",
        )

    raise ValueError(f"unknown vertex process: {process}")


def _format_phase(phase: np.ndarray) -> str:
    return "(" + ",".join(f"{value:.6g}" for value in phase) + ")"


def add_vertex_process_arguments(
    parser: argparse.ArgumentParser,
    *,
    intensity_help: str,
    cox_variables: str,
) -> None:
    parser.add_argument(
        "--vertex-process",
        choices=("poisson", "cox", "mixed-binomial"),
        default="poisson",
        help="Vertex process to sample (default: poisson).",
    )
    parser.add_argument(
        "--intensity",
        type=nonnegative_float,
        help=intensity_help + " Required for Poisson and Cox processes.",
    )
    parser.add_argument(
        "--cox-function",
        type=str,
        help=(
            f"Nonnegative NumPy intensity-multiplier expression in {cox_variables}; "
            "the supplied period is also available as `period`."
        ),
    )
    parser.add_argument(
        "--period",
        type=positive_float,
        help="Positive period of --cox-function in every coordinate.",
    )
    parser.add_argument(
        "--cox-bound",
        type=positive_float,
        help=(
            "Optional certified upper bound for the Cox intensity multiplier; "
            "otherwise a conservative numerical grid estimate is used."
        ),
    )
    parser.add_argument(
        "--count-quantile",
        type=str,
        help=(
            "NumPy inverse-CDF expression in u for the nonnegative, integer-valued "
            "mixed-binomial point count."
        ),
    )

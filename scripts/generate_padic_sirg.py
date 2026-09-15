"""Generate a SIRG on the p-adic unit ball as a TikZ figure.

Example (from the project root):
    python scripts/generate_padic_sirg.py --p 3 --intensity 150 --quantile "2*u - 1" --line-thickness 0.25 --line-opacity 0.35 --kernel "np.exp(-3*t)"

    python scripts/generate_padic_sirg.py --p 3 --vertex-process cox --intensity 150 --cox-function "1 + .8*np.sin(2*np.pi*x/period)" --period 1 --kernel "np.exp(-3*t)"

    python scripts/generate_padic_sirg.py --p 3 --vertex-process mixed-binomial --count-quantile "np.where(u < .5, 100, 160)" --kernel "np.exp(-3*t)"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from .vertex_processes import (
        add_vertex_process_arguments,
        compile_quantile,
        evaluate_kernel,
        positive_float,
        probability,
        quantile_values,
        sample_vertex_process,
    )
except ImportError:  # Direct execution: python scripts/generate_padic_sirg.py
    from vertex_processes import (
        add_vertex_process_arguments,
        compile_quantile,
        evaluate_kernel,
        positive_float,
        probability,
        quantile_values,
        sample_vertex_process,
    )


SEED = 42
MAX_BACKGROUND_CIRCLES = 2500
MIN_BACKGROUND_RADIUS = 0.008
EXTRA_POINT_DEPTH = 4


def prime_int(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be a prime integer")
    if parsed == 2:
        return parsed
    if parsed % 2 == 0:
        raise argparse.ArgumentTypeError("must be a prime integer")
    for divisor in range(3, math.isqrt(parsed) + 1, 2):
        if parsed % divisor == 0:
            raise argparse.ArgumentTypeError("must be a prime integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a SIRG on B(0,1) in Q_p."
    )
    parser.add_argument(
        "--p",
        type=prime_int,
        required=True,
        help="Prime defining the p-adic field.",
    )
    add_vertex_process_arguments(
        parser,
        intensity_help=(
            "Base intensity relative to normalized Haar measure on the unit ball."
        ),
        cox_variables=(
            "x, the measure-preserving base-p coordinate of a p-adic point"
        ),
    )
    parser.add_argument(
        "--line-thickness",
        type=positive_float,
        default=0.4,
        help="Edge thickness in points (default: 0.4).",
    )
    parser.add_argument(
        "--line-opacity",
        type=probability,
        default=1.0,
        help="Edge opacity in [0, 1] (default: 1).",
    )
    parser.add_argument(
        "--plain-background",
        action="store_true",
        help="Omit the recursive p-adic ball boundaries.",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="Use fresh system entropy instead of the fixed seed 42.",
    )
    parser.add_argument(
        "--quantile",
        type=str,
        default="2*u - 1",
        help='NumPy inverse-CDF expression in u (default: "2*u - 1").',
    )
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        help='NumPy expression in t, w, and wp, e.g. "np.exp(-3*t)".',
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output TikZ path (default: figures/padic_sirg.tex).",
    )
    return parser.parse_args()


def packing_ratio(p: int) -> float:
    sine = math.sin(math.pi / p)
    return sine / (1.0 + sine)


def child_circle(
    center_x: float,
    center_y: float,
    radius: float,
    digit: int,
    level: int,
    p: int,
) -> tuple[float, float, float]:
    child_radius = radius * packing_ratio(p)
    center_offset = radius - child_radius
    angle = math.pi / 2 + level * math.pi / p + 2 * math.pi * digit / p
    return (
        center_x + center_offset * math.cos(angle),
        center_y + center_offset * math.sin(angle),
        child_radius,
    )


def background_circles(p: int) -> tuple[list[tuple[float, float, float, int]], int]:
    circles = [(0.0, 0.0, 1.0, 0)]
    frontier = [(0.0, 0.0, 1.0)]
    level = 0

    while frontier:
        next_radius = frontier[0][2] * packing_ratio(p)
        next_count = len(frontier) * p
        if level > 0 and (
            len(circles) + next_count > MAX_BACKGROUND_CIRCLES
            or next_radius < MIN_BACKGROUND_RADIUS
        ):
            break

        children = []
        for center_x, center_y, radius in frontier:
            for digit in range(p):
                child_x, child_y, child_radius = child_circle(
                    center_x, center_y, radius, digit, level, p
                )
                children.append((child_x, child_y, child_radius))
                circles.append((child_x, child_y, child_radius, level + 1))
        frontier = children
        level += 1

    return circles, level


def padic_digits_from_unit_coordinates(
    coordinates: np.ndarray,
    p: int,
    initial_depth: int,
) -> np.ndarray:
    """Map uniform coordinates in [0, 1) to Haar-uniform p-adic digits.

    The map reads the base-p expansion from left to right.  It is
    measure-preserving (up to floating-point precision), which lets all shared
    vertex processes sample in one coordinate representation.
    """

    values = np.asarray(coordinates, dtype=float).reshape(-1)
    if np.any(values < 0) or np.any(values >= 1):
        raise ValueError("p-adic sampling coordinates must lie in [0, 1)")
    digits = np.empty((len(values), 0), dtype=np.int64)
    remainders = values.copy()
    target_depth = max(1, initial_depth)

    while digits.shape[1] < target_depth:
        remainders *= p
        next_digits = np.floor(remainders).astype(np.int64)
        # Protect against a floating-point product rounding exactly to p.
        next_digits = np.minimum(next_digits, p - 1)
        digits = np.column_stack((digits, next_digits))
        remainders -= next_digits

    while len(values) > 1 and np.unique(digits, axis=0).shape[0] < len(values):
        if digits.shape[1] >= 64:
            raise ValueError("could not distinguish sampled p-adic points numerically")
        remainders *= p
        next_digits = np.minimum(np.floor(remainders).astype(np.int64), p - 1)
        digits = np.column_stack((digits, next_digits))
        remainders -= next_digits
    return digits


def visual_positions(digits: np.ndarray, p: int) -> np.ndarray:
    vertex_count, depth = digits.shape
    positions = np.zeros((vertex_count, 2), dtype=float)
    radius = 1.0
    ratio = packing_ratio(p)

    for level in range(depth):
        child_radius = radius * ratio
        center_offset = radius - child_radius
        angles = (
            math.pi / 2
            + level * math.pi / p
            + 2 * math.pi * digits[:, level] / p
        )
        positions[:, 0] += center_offset * np.cos(angles)
        positions[:, 1] += center_offset * np.sin(angles)
        radius = child_radius
    return positions


def padic_distances(
    digits: np.ndarray, first: np.ndarray, second: np.ndarray, p: int
) -> np.ndarray:
    if first.size == 0:
        return np.empty(0, dtype=float)
    differing = digits[first] != digits[second]
    first_difference = np.argmax(differing, axis=1)
    return np.power(float(p), -first_difference.astype(float))


def tex_number(value: float) -> str:
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def tikz_source(
    p: int,
    positions: np.ndarray,
    edges: np.ndarray,
    circles: list[tuple[float, float, float, int]],
    line_thickness: float,
    line_opacity: float,
    plain_background: bool,
    seed_label: str,
    process_description: str,
) -> str:
    vertex_count = len(positions)
    lines = [
        "\\begin{tikzpicture}[x=3.5cm, y=3.5cm]",
        (
            "% Generated by scripts/generate_padic_sirg.py: "
            f"p={p}, seed={seed_label}, vertices={vertex_count}, "
            f"process={process_description}, line width={line_thickness:g}pt, "
            f"opacity={line_opacity:g}"
        ),
    ]

    if not plain_background:
        lines.extend(["% Recursive p-adic ball hierarchy"])
        for center_x, center_y, radius, depth in circles:
            if depth == 0:
                style = "thin, gray!70"
            else:
                shade = max(15, 48 - 7 * (depth - 1))
                style = f"very thin, gray!{shade}"
            lines.append(
                f"\\draw[{style}] ({tex_number(center_x)},{tex_number(center_y)}) "
                f"circle ({tex_number(radius)});"
            )

    lines.extend(["", "% Vertex coordinates"])
    lines.extend(
        f"\\coordinate (v{i}) at ({tex_number(x)},{tex_number(y)});"
        for i, (x, y) in enumerate(positions)
    )

    lines.extend(["", "% Edges"])
    lines.extend(
        (
            f"\\draw[line width={line_thickness:.6g}pt, "
            f"opacity={line_opacity:.6g}] (v{i}) -- (v{j});"
        )
        for i, j in edges
    )

    lines.extend(["", "% Vertices"])
    if vertex_count:
        vertex_names = ",".join(f"v{i}" for i in range(vertex_count))
        lines.extend(
            [
                f"\\foreach \\v in {{{vertex_names}}} {{",
                "    \\fill (\\v) circle (0.4pt);",
                "}",
            ]
        )
    lines.extend(["\\end{tikzpicture}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seed = None if args.random_seed else SEED
    seed_label = "random" if args.random_seed else str(SEED)
    rng = np.random.default_rng(seed)

    circles, background_depth = background_circles(args.p)

    try:
        # The unit interval coordinate is a measure-preserving encoding of
        # normalized Haar measure by base-p digits.
        vertex_sample = sample_vertex_process(
            process=args.vertex_process,
            rng=rng,
            lower=(0.0,),
            upper=(1.0,),
            intensity=args.intensity,
            cox_function=args.cox_function,
            period=args.period,
            cox_bound=args.cox_bound,
            count_quantile=args.count_quantile,
            variables=("x",),
        )
        vertex_count = len(vertex_sample.points)
        digits = padic_digits_from_unit_coordinates(
            vertex_sample.points[:, 0],
            args.p,
            max(1, background_depth + EXTRA_POINT_DEPTH),
        )
        positions = visual_positions(digits, args.p)
        quantile = compile_quantile(args.quantile)
        epsilon = np.finfo(float).eps
        uniforms = np.clip(rng.random(vertex_count), epsilon, 1.0 - epsilon)
        weights = quantile_values(quantile, uniforms)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    first, second = np.triu_indices(vertex_count, k=1)
    distances = padic_distances(digits, first, second, args.p)
    try:
        probabilities = evaluate_kernel(
            args.kernel, distances, weights[first], weights[second]
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    selected = rng.random(probabilities.size) < probabilities
    edges = np.column_stack((first[selected], second[selected]))

    output = args.output or (
        Path(__file__).resolve().parents[1] / "figures" / "padic_sirg.tex"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        tikz_source(
            args.p,
            positions,
            edges,
            circles,
            args.line_thickness,
            args.line_opacity,
            args.plain_background,
            seed_label,
            vertex_sample.description,
        ),
        encoding="utf-8",
    )
    seed_mode = "random" if args.random_seed else f"fixed ({SEED})"
    print(
        f"Wrote {output} "
        f"(p={args.p}, {vertex_count} vertices, {len(edges)} edges, "
        f"process={vertex_sample.description}, seed mode={seed_mode})."
    )


if __name__ == "__main__":
    main()

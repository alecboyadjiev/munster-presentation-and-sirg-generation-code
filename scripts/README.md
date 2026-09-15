# SIRG figure generators

These scripts generate self-contained TikZ pictures of spatial inhomogeneous
random graphs (SIRGs):

- `generate_sirg.py` samples a graph in the Euclidean square
  `[-dim, dim]^2`.
- `generate_padic_sirg.py` samples a graph on the p-adic unit ball
  `B(0,1) \subset \mathbb{Q}_p` and lays it out as nested circles.
- `vertex_processes.py` contains the shared point-process sampling, expression
  parsing, and validation code. It is imported by both generators and is not
  normally run directly.

Both generators follow the same pipeline:

1. Sample vertex locations from a Poisson, phase-shifted Cox, or
   mixed-binomial point process.
2. Sample one weight per vertex from `--quantile`.
3. Evaluate `--kernel` for every unordered pair of vertices.
4. Sample each edge independently with the resulting probability.
5. Write the vertices and sampled edges as a TikZ picture.

Pair enumeration is quadratic in the number of vertices, so large intensities
or mixed-binomial counts can require substantial memory and output very large
`.tex` files.

## Requirements and invocation

Run the scripts from the project root with Python 3.10 or newer and NumPy
installed:

```powershell
python -m pip install numpy
python scripts/generate_sirg.py --help
python scripts/generate_padic_sirg.py --help
```

The generated files require TikZ when included in a LaTeX document. The
project's `main.tex` already loads it.

## Euclidean generator

Minimal Poisson example:

```powershell
python scripts/generate_sirg.py `
  --dim 10 `
  --intensity 1 `
  --kernel "np.exp(-t)"
```

This samples a homogeneous Poisson process of intensity `1` in
`[-10,10]^2`, assigns the default weights `2*u - 1`, samples edges with
probability `exp(-t)`, and writes `figures/sirg.tex`.

### Euclidean-only option

| Option | Required | Meaning |
| --- | --- | --- |
| `--dim DIM` | Yes | Positive integer half-width of the sampling square. Locations lie in `[-DIM,DIM]^2`, the area is `(2*DIM)^2`, and `t` in the kernel is ordinary Euclidean distance. |

Unless `--plain-background` is used, the TikZ output includes a unit grid and
an outline of the sampling square. Its coordinate scale is chosen so the
square renders approximately 7 cm wide.

## p-adic generator

Minimal Poisson example:

```powershell
python scripts/generate_padic_sirg.py `
  --p 3 `
  --intensity 150 `
  --kernel "np.exp(-3*t)"
```

This samples a homogeneous Poisson process on the unit ball of
`\mathbb{Q}_3`, using normalized Haar measure, and writes
`figures/padic_sirg.tex`.

### p-adic-only option

| Option | Required | Meaning |
| --- | --- | --- |
| `--p P` | Yes | Prime defining `\mathbb{Q}_p`. Composite values and integers below 2 are rejected. |

Uniform coordinates in `[0,1)` are converted to base-`p` digits. If two
points first differ at digit index `k`, their distance is `p^(-k)`. The nested
circle layout visualizes the p-adic ball hierarchy; it is a display layout,
not a Euclidean metric embedding. Digit depth is extended as needed to keep
sampled points distinct.

Unless `--plain-background` is used, the output includes recursive p-adic ball
boundaries. Background recursion stops before it exceeds 2,500 circles or the
next circles become smaller than the configured display threshold.

## Shared options

| Option | Default | Meaning |
| --- | --- | --- |
| `--vertex-process {poisson,cox,mixed-binomial}` | `poisson` | Selects the point process used for the vertex locations. Each choice has its own required and forbidden options, described below. |
| `--intensity VALUE` | None | Nonnegative base intensity. Required for Poisson and Cox processes. For the Euclidean generator it is per unit area; for the p-adic generator it is relative to normalized Haar measure on the unit ball. Do not pass it for a mixed-binomial process. |
| `--cox-function EXPR` | None | Nonnegative periodic intensity multiplier for a Cox process. It may use NumPy as `np`, the coordinate variables, and `period`. |
| `--period VALUE` | None | Positive common period of `--cox-function` in every coordinate. Required for a Cox process. |
| `--cox-bound VALUE` | Automatic estimate | Positive certified upper bound for the Cox multiplier, used as the thinning envelope. It is only valid with a Cox process. |
| `--count-quantile EXPR` | None | Inverse CDF for the mixed-binomial vertex count. It may use `u` and `np` and must return a nonnegative integer-valued, nondecreasing function. |
| `--quantile EXPR` | `2*u - 1` | Inverse CDF for vertex weights. It may use `u` and `np` and must be finite, real, and nondecreasing on `(0,1)`. This is independent of `--count-quantile`. |
| `--kernel EXPR` | None | Required edge-probability expression. It may use distance `t`, the first weight `w`, the second weight `wp`, and `np`. Results below 0 are clipped to 0 and results above 1 are clipped to 1. |
| `--line-thickness VALUE` | `0.4` | Positive TikZ edge width in points. |
| `--line-opacity VALUE` | `1` | Edge opacity in the interval `[0,1]`. |
| `--plain-background` | Off | Euclidean: omit the grid and square outline. p-adic: omit the recursive ball boundaries. |
| `--random-seed` | Off | Use fresh system entropy. Without this flag both scripts use the fixed seed `42`, making identical commands reproducible. |
| `--output PATH` | Generator-specific | Override the output path. Missing parent directories are created automatically. |
| `-h`, `--help` | - | Print the complete command-line help. |

`--kernel`, `--quantile`, `--count-quantile`, and `--cox-function` are NumPy
expressions evaluated in a restricted namespace with no Python built-ins.
Quote expressions at the shell so parentheses and operators reach the script
unchanged.

## Vertex processes

### Homogeneous Poisson process

Use `--vertex-process poisson`, or omit the option because Poisson is the
default. `--intensity` is required.

- Euclidean expected vertex count:
  `intensity * (2*dim)^2`.
- p-adic expected vertex count: `intensity`, because the unit ball has
  normalized Haar measure one.

```powershell
python scripts/generate_sirg.py `
  --dim 10 `
  --vertex-process poisson `
  --intensity 1 `
  --quantile "2*u - 1" `
  --kernel "np.exp(-(t / 2)**2)" `
  --line-thickness 0.25 `
  --line-opacity 0.35 `
  --plain-background `
  --output figures/sirg-poisson.tex
```

### Phase-shifted periodic Cox process

Use `--vertex-process cox` with `--intensity`, `--cox-function`, and
`--period`. The realized intensity is

```text
--intensity * --cox-function(point + random_phase).
```

The script samples one phase uniformly from a period in each coordinate. For
the Euclidean generator, the expression may use `x`, `y`, `np`, and `period`:

```powershell
python scripts/generate_sirg.py `
  --dim 10 `
  --vertex-process cox `
  --intensity 1 `
  --cox-function "1 + .8*np.sin(2*np.pi*x/period)*np.cos(2*np.pi*y/period)" `
  --period 4 `
  --kernel "np.exp(-t)" `
  --output figures/sirg-cox.tex
```

For the p-adic generator, `x` is the measure-preserving base-`p` coordinate in
`[0,1)`. The multiplier is therefore a density relative to normalized Haar
measure:

```powershell
python scripts/generate_padic_sirg.py `
  --p 3 `
  --vertex-process cox `
  --intensity 150 `
  --cox-function "1 + .8*np.sin(2*np.pi*x/period)" `
  --period 1 `
  --kernel "np.exp(-3*t)" `
  --output figures/padic-sirg-cox.tex
```

The Cox expression must be finite, real, nonnegative, and periodic in every
coordinate. The scripts test it on dense regular and half-cell-offset grids.
They then use rejection sampling and check accepted candidates against the
thinning envelope.

Without `--cox-bound`, the envelope is the largest grid value with a small
safety factor. Supply a known global maximum for a sharply peaked or
discontinuous multiplier:

```powershell
--cox-bound 1.8
```

The script rejects a supplied bound that is below a sampled value. Runtime
evaluation also fails if an off-grid value exceeds the established envelope.
These numerical checks detect common errors but cannot prove periodicity or a
global bound for an arbitrary expression.

### Mixed-binomial process

Use `--vertex-process mixed-binomial` with `--count-quantile`. Do not pass
`--intensity`. The script samples one `u` uniformly from `(0,1)`, evaluates the
count quantile, and then places exactly that many independent uniform points.

```powershell
python scripts/generate_sirg.py `
  --dim 10 `
  --vertex-process mixed-binomial `
  --count-quantile "np.where(u < .5, 100, 160)" `
  --kernel "np.exp(-t)" `
  --output figures/sirg-mixed-binomial.tex
```

The count quantile must be finite, nondecreasing, nonnegative, and
integer-valued throughout `(0,1)`. Valid examples include:

```text
np.floor(20*u) + 50
np.where(u < .5, 100, 160)
0*u + 120
```

A continuous count expression such as `20*u` is rejected because it is not
integer-valued.

The same process works for the p-adic generator:

```powershell
python scripts/generate_padic_sirg.py `
  --p 3 `
  --vertex-process mixed-binomial `
  --count-quantile "0*u + 150" `
  --kernel "np.exp(-3*t)" `
  --plain-background `
  --output figures/padic-sirg-binomial.tex
```

## Weight and kernel expressions

The two quantile options control different random variables:

- `--quantile` is evaluated once per vertex and produces its weight.
- `--count-quantile` is evaluated once per graph and produces the number of
  vertices for a mixed-binomial process.

The kernel receives NumPy arrays and is evaluated simultaneously for every
unordered pair. It can ignore weights:

```powershell
--kernel "np.exp(-t)"
```

or use them explicitly:

```powershell
--quantile "-np.log(1-u)" `
--kernel "np.minimum(1, w*wp/(1+t)**2)"
```

Scalar expressions are broadcast across all vertices or pairs. Every
expression must produce finite real values of the required shape. Complex or
non-finite results are rejected.

## Output and LaTeX use

Each output file contains one complete `tikzpicture`. Include it directly:

```tex
\resizebox{0.82\linewidth}{!}{\input{figures/sirg-poisson}}
```

The first comment in each generated file records the generator, seed mode,
vertex count, process description, line width, and opacity. The p-adic script
also records `p`. Each command prints the output path and sampled graph size
when it finishes.

Existing output files are overwritten when the same `--output` path is used.
Use distinct filenames when keeping several parameter choices or random
realizations.

## Reproducibility and failures

By default, NumPy's generator uses seed `42`. Point locations, weights, and
edges are therefore reproducible for the same script version and arguments.
`--random-seed` uses fresh system entropy and labels the output as `seed=random`;
the exact entropy seed is not recorded.

The scripts exit with an error for invalid option combinations, nonpositive
dimensions or line widths, non-prime `p`, invalid probabilities, malformed or
non-monotone quantiles, invalid Cox multipliers, or expression results that
cannot be broadcast to the required shape.

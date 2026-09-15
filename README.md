# Adaptif

[![Actions status](https://github.com/bglid/adaptive-filters/workflows/build/badge.svg)](https://github.com/bglid/adaptive-filters/actions)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/bglid/adaptive-filters)](https://github.com/bglid/adaptive-filters/blob/master/LICENSE)

#### Python adaptive filtering DSP algorithms package that uses pyo3 for fast processing.

---

_Project is still WIP. Going through massive refactor to Rust [pyo3](https://github.com/pyo3/pyo3) internals_

---

## Filters:

This project contains Python implementations of Adaptive filtering algorithms, currently including:

| Adaptive Filter Algorithm            |    Status     |
| ------------------------------------ | :-----------: |
| Least Mean Squares (LMS)             |       ✔       |
| Normalized Least Mean Squares (NLMS) |       ✔       |
| Recursive Least Squares (RLS)        |       ✔       |
| Affine Projection Algorithm (APA)    |       ✔       |
| Frequency Domain Adaptive Filters    | _in progress_ |

> [!NOTE]
> The frequency-domain implementations are still experimental and currently require further work before being appropriate for practical use.

---

## Installation

The projects currently supports python 3.10 - 3.14. The project uses `uv` for dependency and environment management.

Cloning the repo:

```bash
git clone https://github.com/bglid/adaptive-filters.git
```

Install the project dev dependencies and build the Python bindings with `maturin`:

```bash
uv sync
uv run maturin develop
```

You can then run commands in the project environment with this format:

```bash
uv run <command>

# example:
uv run pytest
```

#### Contributing

See [CONTRIBUTING.md](https://github.com/bglid/adaptive-filters/blob/main/CONTRIBUTING.md) for setup and contribution guidelines.

In short,

- Open an issue for a discussion. We will likely handle it.
- Undisclosed AI PRs will be closed and no further PRs from said user will be considered.

---

## Usage

Filters can be imported from the `adaptif` package.

LMS adaptive filter example:

```python
from adaptif import LMSFilter

# setting up filter
lms_af = LMSFilter(mu=0.001, n=32)

# Assuming signals are already present and named accordingly
cleaned_signal = lms_af.adapt(d=noisy_signal, x=noise)
```

Here:

- `d` is the desired/noisy signal.
- `x` is the reference noise signal.

---

## Credits

Organization and project originally inspired by [`Padasip`](https://github.com/matousc89/padasip)

## Citation

If you found any of this helpful, feel free to cite it, or just send us an email.

```bibtex
@misc{adaptif,
  authors = {Benjamin Glidden, Elias Naske},
  title = {Adaptif: Rust and Python implementation of DSP adaptive filters},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bglid/adaptive_filter}}
}
```

---

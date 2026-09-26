# Adaptif

[![Actions status](https://github.com/bglid/adaptif/workflows/build/badge.svg)](https://github.com/bglid/adaptif/actions)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/bglid/adaptive-filters)](https://github.com/bglid/adaptive-filters/blob/master/LICENSE)

#### Rust crate for DSP Adaptive Filters with Python bindings. 

---

_Project is still WIP. Going through massive refactor to Rust [pyo3](https://github.com/pyo3/pyo3) internals_

---

## Filters:

This project contains Rust implementations of Adaptive filtering algorithms, with bindings for Python.
We're currently working towards a first full release, which will include the following algorithms:

| Adaptive Filter Algorithm            |    Status     |
| ------------------------------------ | :-----------: |
| Least Mean Squares (LMS)             |       ✔       |
| Normalized Least Mean Squares (NLMS) |       ✔       |
| Recursive Least Squares (RLS)        |       ✔       |
| Affine Projection Algorithm (APA)    | _in progress_ |

---

## Installation

The projects currently supports Rust stable and Python 3.10 - 3.14.

Cloning the repo:

```bash
git clone https://github.com/bglid/adaptif.git
```

Then use Make to set up the dev environment:
```bash
make setup-rust # Install/update the Rust toolchain through Rustup
make setup-uv # Install and setup uv for Python dependency management
# or
make setup # Run both of the above
```

All development processes (running checks, test, etc.) are codified in the project's Makefile.
See [CONTRIBUTING.md](https://github.com/bglid/adaptif/blob/main/CONTRIBUTING.md) for a quick reference on the available commands.

#### Contributing

See [CONTRIBUTING.md](https://github.com/bglid/adaptif/blob/main/CONTRIBUTING.md) for setup and contribution guidelines.

In short,

- Open an issue for a discussion. We will likely handle it.
- Undisclosed AI PRs will be closed and no further PRs from said user will be considered.

---

## Usage

### Rust

See the examples in `examples/`.

### Python

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
  title = {Adaptif: Rust crate for DSP adaptive filters with Python bindings},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bglid/adaptif}}
}
```

---

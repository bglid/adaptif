# Adaptif

[![Actions status](https://github.com/bglid/adaptif/workflows/build/badge.svg)](https://github.com/bglid/adaptif/actions)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/bglid/adaptif)](https://github.com/bglid/adaptif/blob/main/LICENSE)

#### Rust crate for DSP Adaptive Filters with Python bindings. 


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

The projects currently supports Rust stable and Python 3.11 - 3.15.

#### Rust crate
Via crates.io:

```bash
cargo add adaptif
```

#### Python bindings
Via PyPI:

```bash
# With uv (preferred).
uv add adaptif
```

```bash
# With pip.
pip install adaptif
```
- - - 

## Usage

### Rust

See the examples in `examples/`.

### Python

Filters can be imported from the `adaptif` package.

LMS adaptive filter example:

```python
from adaptif import LMSFilter

# setting up filter
lms_af = LMSFilter(mu=0.001, window_size=32)

# Assuming signals are already present and named accordingly
cleaned_signal = lms_af.adapt(input_signal=noisy_signal, noise_ref=noise)
```

Here:

- `input_signal` is the desired/noisy signal, AKA the mixed noisy signal containing the clean signal to extract.
- `noise_ref` is the reference noise signal. The noise which you wish to remove from the input signal.

---
## Contributing

See [CONTRIBUTING.md](https://github.com/bglid/adaptif/blob/main/CONTRIBUTING.md) for setup and contribution guidelines.

In short,

- Open an issue for a discussion. We will likely handle it.
- Undisclosed AI PRs will be closed and no further PRs from said user will be considered.

---

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

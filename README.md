# Adaptif

[![Actions status](https://github.com/bglid/adaptif/workflows/build/badge.svg)](https://github.com/bglid/adaptif/actions)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/bglid/adaptif)](https://github.com/bglid/adaptif/blob/main/LICENSE)

#### Rust crate for DSP Adaptive Filters with Python bindings. 

---

## Overview

This project contains Rust implementations of [adaptive filtering](https://en.wikipedia.org/wiki/Adaptive_filter) algorithms, with bindings for Python.
Currently, the following algorithms are supported:

| Adaptive Filter Algorithm            |    Status     |
| ------------------------------------ | :-----------: |
| Least Mean Squares (LMS)             |       ✔       |
| Normalized Least Mean Squares (NLMS) |       ✔       |
| Recursive Least Squares (RLS)        |       ✔       |
| Affine Projection Algorithm (APA)    | _in progress_ |

We're also working on real-time processing and `no_std` functionality for use in embedded systems, which are planned for the next major release.

---

## Installation

The projects currently supports Rust stable (no MSRV) and Python 3.11 - 3.15.

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

This section shows basic examples for a least mean squares (LMS) filter in Rust and Python.

Here:

- `input_signal` is the desired/noisy signal. Commonly referred to as `d(n)` in mathematical notation.

- `noise_ref` is the reference noise signal. Commonly referred to as `x(n)` in mathematical notation. 

More examples can be found in `examples/`.

### Rust

```rust
use adaptif::filters::{LmsFilter, LmsOptions, ProcessingMode};
use adaptif::types::{InputSignal, NoiseReference};

// Sample inputs
let input_signal = InputSignal::new(vec![1.0, -2.5, 3.0])?;
let noise_ref = NoiseReference::new(vec![2.0, -1.2, -3.8])?;

// Initialize the filter
let mut lms = LmsFilter::new(
    LmsOptions {
        step_size: 0.1,  // How fast the filter adapts
        window_size: 16, // How many samples the filter processes at a time
    },
    ProcessingMode::Sample,
)?;

// Adapt the filter to the inputs and get the iteratively cleaned signal.
let cleaned_signal = lms.adapt(&input_signal, &noise_ref)?;

Ok(())
```

### Python

```python
from adaptif import LMSFilter

# setting up filter
lms = LMSFilter(step_size=0.001, window_size=32)

# Assuming signals are already present and named accordingly
cleaned_signal = lms.adapt(input_signal=noisy_signal, noise_ref=noise)
```

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

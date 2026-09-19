# How to contribute

### READ BELOW

---

#### If a PR is submitted with no issue attached and a green light on working on a PR, it will be generally closed. Undisclosed AI PRs will be **closed** automatically

#### **Overall, submit an issue instead.**

---

## General Information

### Makefile

Project setup and common development processes are all codified using Make.
Please read through the project's Makefile for more details on the steps described below.

#### Quick Reference
Environment setup:
- `make check-installs`: Check if tooling is installed.
- `make setup`: Install/update dev tools and dependencies.

Development processes:
- `make check-rs` / `check-py` / `check-all`: Run formating, linting, and type-checking (Rust, Python, or both).
- `make test-rs` / `test-py` / `test-all`: Run tests with coverage (Rust, Python, or both).
- `make python`: Generate Python bindings for the Rust code (Runs automatically before `check-py` and `test-py`).
- `make audit`: Audit dependencies to check for supply-chain vulnerabilites.
- `make clean`: Remove build artifacts and Python bindings.

### File tracking

We maintain `.gitignore` as a whitelist by ignoring all files by default and implicitly including only the files we actually need by prefixing them with `!`, e.g.:

```bash
# directory
!src/

# all subdirectories
!src/**/

# only Rust files
!src/**/*.rs
```

If you add new files to `.gitignore`, try to keep your additions reasonably concise by using wildcard like in the example above.
Likewise, if you delete any files, make sure to remove any lines that are no longer necessary.

## Tooling & Dependencies

To install and set up tooling and dependencies:
```bash
make setup
```

### Rust

`adaptif` is mainly written in Rust.
Use the following command to install the Rust toolchain via Rustup, as well as the necessary Cargo extensions:

```bash
make setup-rust
```

### uv

We use `uv` to manage Python dependencies and the dev env.
The following command will install/update `uv` and install the dev dependencies:

```bash
make setup-uv
```

`uv` automatically manages the project virtual environment in `.venv`. Manually activating it is unnecessary. Run project tools with `uv run`.

## Python bindings

The Python package offers bindings for the Rust code via [PyO3](https://github.com/PyO3/pyo3).
To generate the Python bindings:

```bash
make python
```

The generated `.so` file will be placed in `adaptif/`

Note that `make check-py` and `make test-py` will build the bindings automatically if necessary,
no need to run `make python`.

### Adding new bindings

New bindings should be imported in `adaptif/__init__.py`.
In order for `ty` to resolve imports correctly, you also have too add any new functions/classes to the stub file (`adaptif/adaptif.pyi`).

## Codestyle

Format & lint the code with:

```bash
# Rust
make check-rs 

# Python
make check-py

# Both
make check-all
```

To run the test suite:

```bash
# Rust
make test-rs 

# Python
make test-py

# Both
make test-all
```

To run the security checks:

```bash
make audit
```

To remove build artifacts:

```bash
make clean
```

### Before submitting

**Again, READ the section at the [top](#if-a-pr-is-submitted-with-no-issue-attached-and-a-green-light-on-working-on-a-pr-it-will-be-generally-closed-ai-prs-will-be-closed-automatically)**

Before submitting your code please do the following steps:

1. Add tests for new changes
   - _Update documentation for significant changes._
2. Update `.gitignore` to whitelist any files you created and remove any files you deleted.
3. Run `make check-all`
3. Run `make test-all`
7. Commit any changes to `Cargo.lock`/`uv.lock` if you modified project dependencies

## Other help

You can contribute by spreading a word about this library.
You can also share your best practices with us.

---

**In particular**, if you use this in any DSP research, please let us know!!

1. Because we love the topic and would love to check out and share your research
2. It gives us an opportunity to see how this library is being used and how it can be improved.

---

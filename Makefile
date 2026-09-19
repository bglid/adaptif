RUST_SRC = src/*
PY_BINDINGS = python/adaptif/adaptif.cpython*.so
CARGO_FLAGS = --all-features --all-targets

##################################################
# ENVIRONMENT SETUP
##################################################

# Check installs
.PHONY: check-rust check-uv check-installs
check-rust-install:
	@command rustup --version >/dev/null 2>&1 || \
		{ echo "Rust toolchain not found; install it with make setup-rust"; exit 1; }
	@command cargo audit --version >/dev/null 2>&1 || \
		{ echo "cargo-audit not found; install it with make setup-rust"; exit 1; }
	@command cargo tarpaulin --version >/dev/null 2>&1 || \
		{ echo "cargo-tarpaulin not found; install it with make setup-rust"; exit 1; }
check-uv-install:
	@command uv --version >/dev/null 2>&1 || \
		{ echo "uv not found; install it with make setup-uv"; exit 1; }
check-installs: check-rust-install check-uv-install

# Install/update tooling and dependencies
.PHONY: install-rust install-uv setup-rust setup-uv setup
install-rust:
	@command rustup --version >/dev/null 2>&1 && \
		rustup update || \
		{ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh; }
install-uv:
	@command uv --version >/dev/null 2>&1 && \
		uv self update || \
		{ curl -LsSf https://astral.sh/uv/install.sh | sh; }
setup-rust: install-rust
	cargo install cargo-audit
	cargo install cargo-tarpaulin
setup-uv: install-uv
	uv sync --dev
	uv run pre-commit install
setup: setup-rust setup-uv


##################################################
# GENERAL DEVELOPMENT PROCESSES
##################################################

# Generate Python bindings
.PHONY: python
python: check-uv-install $(PY_BINDINGS)
$(PY_BINDINGS): $(RUST_SRC)
	uv run maturin develop


# Formating, linting, and type-checking
.PHONY: check-rs check-py check-all
check-rs: check-rust-install
	cargo fmt --all
	cargo check $(CARGO_FLAGS)
	cargo clippy $(CARGO_FLAGS) -- -Dwarnings
check-py: $(PY_BINDINGS)
	uv run ruff format
	uv run ruff check --fix
	uv run ty check
check-all: check-rs check-py


# Run tests
.PHONY: test-rs test-py test-all
test-rs: check-rust-install
	# NOTE: `--all-features` currently produces a linker error, smth to do with PyO3.
	# Since we don't have any tests in the Python features, leaving it out for now.
	cargo tarpaulin --frozen --skip-clean
test-py: $(PY_BINDINGS)
	uv run pytest
test-all: test-rs test-py


# Audit dependencies
.PHONY: audit
audit: check-rust-install check-uv-install
	cargo audit
	uv audit


# Remove build artifacts and bindings
.PHONY: clean
clean: check-rust-install
	cargo clean
	rm -rf $(PY_BINDINGS)


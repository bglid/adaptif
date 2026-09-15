RUST_SRC = src/*
PY_BINDINGS = python/adaptif/adaptif.cpython*.so
CARGO_FLAGS = --all-features --all-targets

# Regenerate bindings if Rust source has changed
$(PY_BINDINGS) : $(RUST_SRC)
	uv run maturin develop

# Formating, linting, and type-checking
check-rs :
	cargo fmt --all
	cargo check $(CARGO_FLAGS)
	cargo clippy $(CARGO_FLAGS) -- -Dwarnings
check-py : $(PY_BINDINGS)
	uv run ruff format
	uv run ruff check --fix
	uv run ty check
check-all :
	make check-rs
	make check-py

# Run tests
test-rs : 
	cargo test $(CARGO_FLAGS)
test-py : $(PY_BINDINGS)
	uv run pytest
test-all :
	make test-rs
	make test-py


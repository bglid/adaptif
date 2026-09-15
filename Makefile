RUST_SRC = src/*
PY_BINDINGS = python/adaptif/adaptif.cpython*.so
CARGO_FLAGS = --all-features --all-targets


# Generate Python bindings
.PHONY : python
python : $(PY_BINDINGS)
$(PY_BINDINGS) : $(RUST_SRC)
	uv run maturin develop


# Formating, linting, and type-checking
.PHONY : check-rs check-py check-all
check-rs :
	cargo fmt --all
	cargo check $(CARGO_FLAGS)
	cargo clippy $(CARGO_FLAGS) -- -Dwarnings
check-py : $(PY_BINDINGS)
	uv run ruff format
	uv run ruff check --fix
	uv run ty check
check-all : check-rs check-py


# Run tests
.PHONY : test-rs test-py test-all
test-rs : 
	cargo test $(CARGO_FLAGS)
test-py : $(PY_BINDINGS)
	uv run pytest
test-all : test-rs test-py


# Remove build artifacts and bindings
.PHONY : clean
clean :
	cargo clean
	rm -rf $(PY_BINDINGS)


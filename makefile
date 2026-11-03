.PHONY: install unit-test format format-check docs
QUIET ?=
Q_FLAG = $(if $(QUIET),-q,)
WITH ?=
WITH_FLAG = $(if $(WITH),--with $(WITH),)

# Build the python package and the Rust extension
install:
	poetry install $(Q_FLAG) $(WITH_FLAG)
	poetry run maturin develop --release $(Q_FLAG)

unit-test:
	poetry run pytest -m "not speed_bench and not memory_bench"

format:
	poetry run ruff format .
	poetry run ruff check --fix .
	cargo fmt

format-check:
	poetry run ruff format --check .
	poetry run ruff check .
	cargo fmt --check

doc:
	poetry run sphinx-build -b html docs docs/_build/html

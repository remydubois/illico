# Installation

`illico` can be installed via pip, compatible with Python 3.11 and onward:

```bash
pip install illico -U
```

Documentation dependencies (used to build the Sphinx docs) are managed by Poetry. Install them with:

```bash
poetry install --with doc
```

For CI we recommend installing dependencies with Poetry and running the build inside Poetry's environment (the project's workflow uses `poetry install --with doc`).

# Installation

`illico` can be installed via pip, compatible with Python 3.11 and onward:

```bash
pip install illico -U
```
This will pull pre-compiled rust libraries for your platform. In case you encounter issues, you can clone and build manually:
```bash
git clone https://github.com/remydubois/illico
cd illico
pipx install -U poetry # Use system-wide pip, not venv-specific
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # install Rust
make install # This will PATH's poetry, which will in turn create poetry's venv and compile Rust library
```

Documentation dependencies (used to build the Sphinx docs) are managed by Poetry. Install them with:

```bash
poetry install --with doc
```

For CI we recommend installing dependencies with Poetry and running the build inside Poetry's environment (the project's workflow uses `poetry install --with doc`).

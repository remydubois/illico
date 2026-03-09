Building the documentation
==========================

Prerequisites
-------------

Install the required packages (preferably in a virtualenv):

```bash
poetry install --with doc
```

Build
-----

From the repository root run:

```bash
cd docs
make html
```

The generated site will be in `docs/_build/html`.

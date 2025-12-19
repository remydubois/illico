import os
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from tqdm import tqdm

from illico.utils.type import scipy_to_nb

CACHE_ROOT = Path(os.environ.get("ILLICO_PYTEST_CACHE", "/tmp/pytest_cache"))
CELL_LINE_URLS = {
    "k562": "https://plus.figshare.com/ndownloader/files/35773219",
    "rpe1": "https://plus.figshare.com/ndownloader/files/35775606",
    "jurkat": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01%2Eh5ad",
    "hepg2": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fhepg2%5Fraw%5Fsinglecell%5F01%2Eh5ad",
}

DSET_FRACTIONS = [
    pytest.param(0.01, id="1%"),
    # pytest.param(1., id="100%", marks=pytest.mark.full),
    pytest.param(1.0, id="100%"),
]
# DSET_FRACTIONS = pytest.param([0.01, 1.], marks=[pytest.mark.debug, pytest.mark.full]),
# # pytest.param(1., id="100%", marks=pytest.mark.full),
# pytest.param(1., id="100%"),
#               ]


@pytest.fixture(
    params=[
        (cell_line, fmt, fraction)
        for cell_line in ["k562", "rpe1", "jurkat", "hepg2"]
        for fmt in ["dense", "csr", "csc"]
        for fraction in [0.0, 1.0]
    ],
    scope="function",
    ids=lambda p: f"{p[0]}-{p[1]}-{p[2]:.0%}",
)
def adata(request):
    """Fixture to download, convert and cache cell line dataset with subsampling"""
    cell_line, fmt, fraction = request.param
    # if fraction.values[0] < 1.0:
    #     request.node.add_marker("debug")

    cache_root = CACHE_ROOT / cell_line
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_path = cache_root / f"{cell_line}.h5ad"
    download_if_missing(cell_line, raw_path)

    target_path = cache_root / f"{cell_line}_{fmt}_{fraction:.0%}.h5ad"

    if target_path.exists():
        adata = ad.read_h5ad(target_path)
        if fmt == "dense":
            adata.X += 1.0
            adata.X -= 1.0
    else:
        if fmt == "csr":
            adata = ad.read_h5ad(raw_path, as_sparse="X", as_sparse_fmt=sparse.csr_matrix)
        elif fmt == "csc":
            adata = ad.read_h5ad(raw_path, as_sparse="X", as_sparse_fmt=sparse.csc_matrix)
        else:
            adata = ad.read_h5ad(raw_path)

        if fraction == 0.0:
            col_idxs = np.random.RandomState(0).choice(adata.n_vars, size=1, replace=False)
            # Sort the column indices because that's how scipy builds sparse matrics.
            col_idxs.sort()
            adata = adata[:, col_idxs].copy()

        adata.write_h5ad(target_path)

    return adata


# TODO: params on log1p and normalization ? A lot of tests would result
@pytest.fixture(scope="session", params=["dense", "csc", "csr"])
def rand_adata(request):
    n_cells = 20_000
    n_genes = 150
    n_groups = 20
    assert n_groups >= 2
    sparsity = 0.5  # ~50% zeros
    rng = np.random.RandomState(0)

    # Random gene-specific mean expression levels
    gene_means = rng.uniform(0.1, 15, size=n_genes)

    # Sample Poisson counts
    dense_counts = rng.poisson(gene_means, size=(n_cells, n_genes)).astype(np.float32)

    # Impose ~50% sparsity by random masking
    mask = rng.rand(n_cells, n_genes) < sparsity
    dense_counts[mask] = 0

    # Create groups associated
    groups = rng.randint(0, n_groups, size=n_cells)  # Add one ref group

    if request.param == "dense":
        data_matrix = dense_counts
    elif request.param == "csc":
        data_matrix = sparse.csc_matrix(dense_counts)
    elif request.param == "csr":
        data_matrix = sparse.csr_matrix(dense_counts)
    else:
        raise ValueError(f"Unknown data format: {request.param}")

    adata = ad.AnnData(
        data_matrix,
        obs=pd.DataFrame({"pert": [f"pert_{g}" for g in groups]}),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )

    return adata


def download_if_missing(cell_line: Literal["k562", "rpe1"], dst: Path) -> Path:
    """
    Stream-download k562-essential. Skips if exists.
    """
    url = CELL_LINE_URLS[cell_line]
    chunk_size = 10 * 1024 * 1024
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return dst

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(req) as resp:
        total = resp.length  # may be None, but Figshare provides it
        with NamedTemporaryFile(delete=False, dir=CACHE_ROOT) as tmp_file:
            try:
                with (
                    open(tmp_file.name, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        smoothing=0.1,
                        desc=f"Downloading {dst}",
                    ) as pbar,
                ):
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            except Exception as e:
                # Clean up temp file on error
                Path(tmp_file.name).unlink(missing_ok=True)
                raise e
            # Move temp file to final destination
            Path(tmp_file.name).rename(dst)

    return dst


@pytest.fixture(scope="function")
def rand_csr():
    n_rows = 1000
    n_cols = 200
    density = 0.1
    rng = np.random.RandomState(0)
    data = sparse.random(n_rows, n_cols, density=density, format="csr", random_state=rng, dtype=np.float64)
    return scipy_to_nb(data)


@pytest.fixture(scope="function")
def rand_csc():
    n_rows = 1000
    n_cols = 200
    density = 0.1
    rng = np.random.RandomState(0)
    data = sparse.random(n_rows, n_cols, density=density, format="csc", random_state=rng, dtype=np.float64)
    return scipy_to_nb(data)

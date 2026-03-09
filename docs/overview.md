# Overview

*illico* is a python library performing blazing fast asymptotic wilcoxon rank-sum tests (same as `scanpy.tl.rank_genes_groups(…, method="wilcoxon")`), useful for single-cell RNASeq data analyses and processing. `illico`'s features are:

1. 🚀 Blazing fast: On K562 (essential) dataset (~300k cells, 8k genes, 2k perturbations), `illico` computes DE genes (with `reference="non-targeting"`) in a mere 30 seconds. That's more than 100 times faster than both `pdex` or `scanpy` with the same compute ressources (8 CPUs).
2. 💠 No compromise: on synthetic data, `illico`'s p-values matched `scipy.stats.mannwhitneyu` up to a relative difference of 1.e-12, and an absolute tolerance of 0.
3. ⚡ Thread-first: `illico` eventually parallelizes the processing (if specified by the user) over **threads**, never processes. This saves you from all the fixed cost of multiprocessing, such as spanning processes, duplicating data across processes, and communication costs.
4. 🐞 Data format agnostic: whether your data is dense, sparse along rows, or sparse along columns, `illico` will deal with it while never converting the whole data to whichever format is more optimized.
5. 🪶 Lightweight: `illico` will process the input data in batches, making any memory allocation needed along the way much smaller than if it processed the whole data at once.
6. 📈 Scalable: Because thread-first and batchable, `illico` scales reasonably with your compute budget. Tests showed that spanning 8 threads brings a 7-fold speedup over spanning 1 single thread.
7. 💾 Out-of-core: `illico` supports h5-based, on-disk-backed, dense and CSC datasets natively.
8. 🎆 All-purpose: `illico` performs both one-versus-reference (useful for perturbation analyses) and one-versus-rest (useful for clustering analyses) wilcoxon rank-sum tests, both equally optimized and fast.

Approximate speed benchmarks ran on k562-essential can be found in the Benchmarks section. All the code used to generate those numbers can be found in `tests/test_asymptotic_wilcoxon.py::test_speed_benchmark`.

💡 Note:

1. This library only performs wilcoxon rank-sum tests, also known as Mann-Whitney test, also performed by `scanpy.tl.rank_genes_groups(…, method="wilcoxon")`. It **does not** perform wilcoxon signed-sum tests, those are less often used in for single-cell data analyses as it requires samples to be **paired**.
2. Exact benchmarks ran on a subset of the whole k562 can be found at the end of this readme.
3. OVO refers to one-versus-one: this test computes u-stats and p-values between control cells and perturbed cells. Equivalent to `scanpy`'s `rank_gene_groups(…, reference="non-targeting")`.
4. OVR refers to one-versus-rest: this test computes u-stats and p-values between each group cells, and all other cells, for each group. Equivalent to `scanpy.tl.rank_genes_groups(…, reference="rest").`

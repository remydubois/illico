# How it works

The rank-sum tests performed by `illico` are classical, asymptotic, rank-sum tests. No approximation nor assumption is done. `Illico` relies on a few optimization tricks that are non-exhaustively listed below:

1. 🧀 Sparse first: if the input data is sparse, that can be a lot less values to sort. Instead of converting it to dense, `illico` will only sort and rank non-zero values, and adjust rank-sums and tie sums later on with missing zeros.
2. 🗑️ Memory-conscious: ranking and sorting values across groups often requires to slice and convert the data numerous times, especially for CSC or CSR data. Memory allocations are minimized and optimized so as to ensure better scalability and lower overall memory footprint.
3. :brain: Sort controls only once: for the one-versus-reference use case, `illico` takes care of not repeatdly sorting the control values. Controls are sorted only once, after what each "perturbation" chunk is sorted, and the two sorted sub-arrays are merged (linear cost). Because there are often much more control cells than perturbed cells, this is a huge economy of processing.
4. :loop: Vectorize everything: for the one-versus-ref use case, `illico` performs one single sorting of the whole batch (all groups combined) and sums ranks for each group in a vectorized manner. This allows to sort only once instead of repeatedly performing `scipy.stats.mannwhitneyu` on all-but-group-*g* and group-*g*, for all *g* - involving one sorting each.
5. :snake: Generally speaking, `illico` relies heavily on `numba`'s JIT kernels to ensure GIL-free operations and efficient vectorization.

[Copilot-CLI generated code review summary]

Roadmap: Rust implementation sanity review

Context

You implemented a Rust backend for the library. This document collects a focused, actionable, file-by-file review of Rust implementation practices (no architecture changes). All unit tests pass currently; this roadmap highlights idiomatic issues, potential bugs, and minimal surgical fixes.

Summary by file (issue → risk → minimal fix)

- src/lib.rs
  - Issue: module registration is fine; no panics observed here.
  - Risk: none.
  - Minimal fix: none required.

- src/groups.rs
  - Issue: encoded_ref_group is an isize sentinel (-1) but code casts it to usize in many places without checking.
  - Risk: negative sentinel -> huge usize -> out-of-bounds panics when indexing indptr/indices.
  - Minimal fix: make encoded_ref_group Option<usize> or validate (>=0) before any cast; replace direct casts with checked branching.

- src/sparse/types.rs
  - Issue: internal indices are i32 while code repeatedly casts to usize; FromPyObject impls use verbose map_err messages and unusual lifetimes.
  - Risk: repeated casts clutter code and increase chance of off-by-one/overflow bugs; odd FromPyObject signatures reduce clarity.
  - Minimal fix: keep i32 at the Python/Rust boundary but convert to usize once in a small helper at entry points; tighten FromPyObject signatures to canonical form and unify error messages.

- src/sparse/csr.rs
  - Issue: many as_slice().expect("") and empty expect messages; many assumptions about contiguity.
  - Risk: non-contiguous inputs will panic instead of returning controlled errors.
  - Minimal fix: replace expect/unwrap with .as_slice().ok_or_else(|| "CSR indices are not contiguous".to_string())? and return Err up the stack; give descriptive messages.

- src/sparse/csc.rs
  - Issue: sort_columns_inplace sorts values only (data) and explicitly states "This func will break indices and values matching".
  - Risk: data/indices mismatch is surprising and fragile — downstream code may implicitly rely on matching ordering.
  - Minimal fix: either (A) implement stable sorting of (indices,data) pairs (zip into tuple and sort) to preserve alignment, or (B) rename/document the function clearly and enforce a precondition that indices are unused after calling it. Prefer (A) as a minimal safe fix.

- src/sparse/mod.rs
  - Issue: none (simple module re-exports).
  - Minimal fix: none.

- src/sparse/sparse_matrix.rs
  - Issue: type definitions and conversions are OK but many conversions use to_owned(); no validation for expected shapes/indptr invariants.
  - Risk: silent mis-shaped inputs may later panic.
  - Minimal fix: add lightweight asserts (or Result returns) in FromPyObject or as_* helpers to validate shapes and monotonic indptr.

- src/ranking.rs
  - Issue A: sort_along_axis_0 (f64) builds a Vec per column instead of sorting in-place when columns are contiguous; many .as_slice().unwrap() calls.
  - Risk A: extra allocations and panics on non-contig inputs.
  - Minimal fix A: check .as_slice_mut() and sort in-place when possible, otherwise fall back to to_vec/copy-back.
  - Issue B: argsort builds Vec<(usize, &f32)>; this allocates more than necessary.
  - Risk B: avoidable allocations and indirections.
  - Minimal fix B: implement argsort by sorting an index vector: let mut idx: Vec<usize> = (0..n).collect(); idx.sort_by(|&i,&j| x[i].total_cmp(&x[j])); return idx.

- src/stats.rs
  - Issue: compute_pvalue uses panic!("Invalid alternative!") for invalid alternative strings.
  - Risk: panics propagate into host process; should be error-returning.
  - Minimal fix: return Err(format!("Invalid alternative: {}", alternative)) and map it to PyValueError at the Python boundary.

- src/math.rs
  - Issue A: chunk_and_fortranize uses nested loops and unsafe as_array() in the Python wrapper.
  - Risk A: unsafe use can be fine but should be guarded; nested loops allocate for every chunk.
  - Minimal fix A: keep unsafe but add a short comment and ensure wrapper docs promise contiguous input or use .to_owned() when necessary; optional micro-optimization: use iterators to fill Fortran array.
  - Issue B: add_at_vec uses match on max_idx with verbose code and returns String errors.
  - Minimal fix B: replace with if-let Some(val) = indices.iter().max() { … } else { … } to simplify; keep error type (String) or refactor later to a concrete Error.

- src/dense_ovr.rs
  - Issue: many uses of .as_slice().unwrap() after chunk_and_fortranize, relying on chunk layout being Fortran; some heavy per-column allocations (argsort).
  - Risk: panics if memory layout changes; performance could be improved.
  - Minimal fix: either assert the layout (debug_assert!) or return Err when .as_slice() is None; use argsort fix from ranking.rs to reduce allocations.

- src/dense_ovo.rs
  - Issue: same class as dense_ovr: in-place sorts rely on contiguous columns; small repeated allocations.
  - Minimal fix: assert contiguity or handle non-contiguous arrays gracefully; reuse temporary buffers where possible.

- src/sparse_ovo.rs
  - Issue: index_rows_into_csc and contig_cols logic use expect/as_slice and do manual cumsum loops—fine but repetitive.
  - Risk: panics on unexpected layouts and missing validation.
  - Minimal fix: replace expect/unwrap with error propagation; add small validations (e.g., check chunk bounds, empty-chunk edgecases noted by TODOs).

- src/sparse_ovr.rs
  - Issue: heavy numeric conversions (usize <-> f64) sprinkled through logic and compute_pvalue expects usize (commented).
  - Risk: losing clarity about when conversions happen and micro-costs.
  - Minimal fix: keep current behavior but centralize conversions (convert counts to f64 once at column-level), and as above, avoid panics from as_slice/expect.

Cross-cutting quick wins (applyable with minimal edits)
- Replace critical unwraps/expect("")/panic! in public-facing code with Result/Err and map_err(PyValueError::new_err) at the wrapper boundary.
- Fix compute_pvalue to return Err on invalid alternative instead of panic.
- Replace argsort allocation pattern with index-based sort to reduce allocations.
- Replace sort_columns_inplace(data-only) with 1-line change to sort (indices,data) pairs, or document precondition; recommended: sort pairs to preserve invariants.
- Add a 1–2 line guard around every use of grpc.encoded_ref_group as usize to assert >=0 before cast.
- Improve error messages on contiguity asserts and use .ok_or_else(...) to fail gracefully.

Suggested minimal batch of surgical edits (small, low-risk)
- Change argsort implementation to index-sort (one-file change).
- Change compute_pvalue to return Err for invalid alternative and update wrapper to map it.
- Replace a handful of .expect("")/.unwrap() in public boundary code to return Err with descriptive messages (search for as_slice().unwrap() / expect("") / panic!).
- Implement safe sort in csc sorting: sort (index,data) pairs instead of data-only to maintain matching.
- Add a simple check before casting encoded_ref_group to usize (or convert to Option<usize> depending on appetite for change).

Notes and rationale
- The code is reasonably structured and tests pass: good starting point. Most issues are safety/robustness (panics on malformed Python inputs), some avoidable allocations, and a few surprising invariants (sorting values without indices).
- The recommended edits avoid changing API/architecture: they focus on error handling, clearer conversions, and small performance improvements.

Next steps
- If desired, apply the small edits above and run the test suite to confirm nothing breaks.
- Optionally, add lightweight benchmarks on hot spots (argsort, sort/column routines) to measure benefits of index-based sort and in-place sorts.

If you want, I can now apply the minimal edits (argsort, compute_pvalue, a few expect->Err replacements, sort pairs) and run tests; say "apply fixes" and I will implement them surgically and run the test suite.

-- End of roadmap

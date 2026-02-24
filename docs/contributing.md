# Contributing

All contributions are welcome through merge requests. Developers are highly encouraged to rely on `tox` as the testing process is quite cumbersome.

## Testing

The reason to be of this package is its speed, hence the need for extensive speed benchmarks in order to compare it exhaustively and accurately against existing solutions. `tox` is used to manage tests and testing environments.

```bash
pip install tox # this can be system-wide, no need to install it within an environment
```

💡 The test suite below can be very long, especially the benchmarks (up to 48 hours). All "bench-" tox commands can be appended with the `-quick` suffix ensuring they are ran on 1 gene (column) of the benchmark data, just to make sure everything runs correctly. Example:

```bash
tox -e bench-all-quick # This will run speed and memory benchmarks for illico, scanpy and pdex
# OR:  tox -e bench-illico-quick # This will run speed and memory benchmarks for illico only
# OR :tox -e bench-ref-quick # This will run speed and memory benchmarks for scanpy and pdex only
```

Appending the `-quick` suffix will not write any result file or json inside the `.benchmarks` or `.memray` folders (that are versioned). Instead, benchmark result files will be written to `/tmp`.

### Unit testing

Those tests are simply used to ensure the p-values and fold-change returned by `illico` are correct, should be quick to run:

`tox -e unit-tests`

:warning: Those tests do not run `-quick` as they use synthetic data that results in much shorter runtime.

### Speed benchmarks

Speed benchmarks are ran against: `pdex` and `scanpy` as references. Those benchmarks take **a lot** of time (>10 hours on 8 CPUs) so they should not be re-ran for every new PR or release. However, if needed:

```bash
tox -e speed-bench-ref # Run speed benchmarks for scanpy and pdex, should not be re-ran, ideally.
```

Before issuing a new PR, in order to see if the updated code does not decrease speed performance, make sure to run:

```bash
tox -e speed-bench-illico #-quick
```

💡 Because benchmark performance depends on the testing environment (type of machine or OS), it is recommended to run this benchmark from `main` on your machine as well. This will give you a clear comparison point apple-to-apple.

Once the benchmarks have ran, you can cat the benchmark results in terminal with:

```bash
tox -e speed-bench-compare
```

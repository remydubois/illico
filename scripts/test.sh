# Scaling benchmark for illico only on 8
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[.*-.*-.*-illico-ovo-nthreads=8\]" -s -vv --disable-warnings --benchmark-enable --benchmark-save "illico-scaling-w-genes"
poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[.*-.*-100%-illico-ovo-nthreads=.*\]" -s -vv --disable-warnings --benchmark-enable --benchmark-save "illico-scaling-w-threads"

# Speed benchmark for all methods on 5% of the the data on 8 threads, one method per file so we dont have to reproduce all when a new version of whichever comes
# On 5% k562: scanpy is roughly 30mins for OVR, 5mins for OVO - regadless sparse or dense. CSR+dense benchmark for scanpy takes a total of 1h15mins.
# On 5% k562: pdex is roughly 15mins for OVO - regadless sparse or dense. CSR+dense benchmark for pdex takes a total of 30mins.
# On 5% illico is neglectible.
# Total runtime on 5% k562: 2hours, skipping CSC
echo "scanpy" "pdex" "illico" | tr " " "\n" | \
xargs -I {} sh -c "poetry run pytest --regex 'tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[.*-(csr|dense)-1%-{}-(ovo|ovr)-nthreads=8\]' -svv --disable-warnings --benchmark-enable --benchmark-save '{}-1%-8-threads'"

# Memory benchmark for all methods on 5% of the data on 8 threads
echo "scanpy" "pdex" "illico" | tr " " "\n" | \
xargs -I {} sh -c "poetry run pytest --regex 'tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[k562-(csr|dense)-5%-{}-(ovo|ovr)-nthreads=8\]' -svv --disable-warnings"
# Grab statistics from traces
find .memray-trackings/ -iname "*.bin" | xargs -I {} sh -c "poetry run memray stats {} --force --json"
# Display comparison
poetry run python scripts/generate-footprint-comparison.py



poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[(csr|dense)-(small)-(scanpy)-(ovo|ovr)-nthreads=8\]" -s --benchmark-enable --benchmark-save "scanpy-small-8-threads" -s -vv --disable-warnings
poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[(csr|dense)-(small)-(pdex)-(ovo)-nthreads=8\]" -s --benchmark-enable --benchmark-save "pdex-small-8-threads" -s -vv --disable-warnings
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[(csr|dense)-(small)-(pdexp)-(ovo)-nthreads=8\]" -s --benchmark-enable --benchmark-save "pdex-small-8-threads" -s -vv --disable-warnings
poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[(csr|dense)-(small)-(illico)-(ovo|ovr)-nthreads=8\]" -s --benchmark-enable --benchmark-save "illico-small-8-threads" -s -vv --disable-warnings
poetry run pytest-benchmark compare "scanpy-small-8-threads" "pdex-small-8-threads" "illico-small-8-threads"
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[(csr|dense)-(median)-(scanpy)-(ovo|ovr)-nthreads=8\]" -s --benchmark-enable --benchmark-save "scanpy-median-8-threads" -s -vv --disable-warnings
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[(csr|dense)-(median)-(pdex)-(ovo)-nthreads=8\]" -s --benchmark-enable --benchmark-save "pdex-median-8-threads" -s -vv --disable-warnings
# # poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[(csr|dense)-(median)-(pdexp)-(ovo)-nthreads=8\]" -s --benchmark-enable --benchmark-save "pdex-median-8-threads" -s -vv --disable-warnings
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_speed_benchmark\[(csr|dense)-(median)-(illico)-(ovo|ovr)-nthreads=8\]" -s --benchmark-enable --benchmark-save "illico-median-8-threads" -s -vv --disable-warnings
# poetry run pytest-benchmark compare "scanpy-median-8-threads" "pdex-median-8-threads" "illico-median-8-threads" --show-sd --csv=footprint_comparison_median_8_threads.csv

# Memory benchmark for all methods on 1/100th of the data on 1 thread
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[(csc|csr|dense)-(small)-(scanpy)-(ovo|ovr)-nthreads=1\]" -s -vv
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[(csc|csr|dense)-(small)-(pdex)-(ovo)-nthreads=1\]" -s -vv
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[(csc|csr|dense)-(small)-(pdexp)-(ovo)-nthreads=1\]" -s -vv
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[(csc|csr|dense)-(small)-(illico)-(ovo|ovr)-nthreads=1\]" -s -vv

# Pdexp must be ran on normalized and raw data as it allocates extra memory for normalization step, in the future
# poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[(csr|dense)-(small)-(tcnorm)-(pdexp)-(ovo)-nthreads=1\]" -vv --disable-warnings
poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[(csr|dense)-(small)-(raw)-(pdexp)-(ovo)-nthreads=1\]" -s -vv --disable-warnings
poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[(csr|dense)-(small)-(raw)-(pdex)-(ovo)-nthreads=1\]" -s -vv --disable-warnings
poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[(csr|dense)-(small)-(raw)-(illico)-(ovo|ovr)-nthreads=1\]" -s -vv --disable-warnings
poetry run pytest --regex "tests/test_asymptotic_wilcoxon.py::test_memory_benchmark\[(csr|dense)-(small)-(raw)-(scanpy)-(ovo|ovr)-nthreads=1\]" -s -vv --disable-warnings
# Grab statistics from traces
find .memray-trackings/ -iname "*.bin" | xargs -I {} sh -c "poetry run memray stats {} --force --json"

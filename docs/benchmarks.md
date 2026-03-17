# Benchmarks

## Benchmarking against other solutions

A *benchmark* is defined by:

1. The cell line (K562 essential, RPE1, Hep-G2, Jurkat) used as input.
2. The data format (CSR, or dense) used to contain the expression matrix.
3. The test performed: OVO (`reference="non-targeting"`) or OVR (`reference=None`).


<center>
  <img src="https://github.com/remydubois/illico/blob/main/assets/method-runtimes-comparison.png?raw=true" width="100%" />
  <figcaption>Runtime comparison for scanpy, pdex and illico on four cell lines.</figcaption>
</center>

## Scalability

`illico` scales reasonably well with your compute budget, with a quasi-linear speedup up to 16 threads.

<center>
  <img src="https://github.com/remydubois/illico/blob/main/assets/illico-scaling-rust.png?raw=true" width="100%" />
  <figcaption>Throughput of illico with increasing compute budget, compared to a perfect scaling.</figcaption>
</center>

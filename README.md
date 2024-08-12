# fsst-rs

A pure-Rust, zero-dependency implementation of the [FSST string compression algorithm][whitepaper].

FSST is a string compression algorithm meant for use in database systems. It was designed by
[Peter Boncz, Thomas Neumann, and Viktor Leis][whitepaper]. It provides 1-3GB/sec compression
and decompression of strings at compression rates competitive with or better than LZ4.

This implementation is somewhat inspired by the [MIT-licensed implementation] from the paper authors, written in C++,
but it is mostly written from a careful reading of the paper.

**NOTE: This current implementation is still in-progress and is not production ready, please use at your own risk.**


[whitepaper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
[MIT-licensed implementation]: https://github.com/cwida/fsst

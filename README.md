<p align="center">
  <img src="https://raw.githubusercontent.com/spiraldb/fsst/develop/logo.webp" height="300">
</p>

![Crates.io Version](https://img.shields.io/crates/v/fsst_rs)
![docs.rs](https://img.shields.io/docsrs/fsst-rs)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/spiraldb/fsst/ci.yml?branch=develop)


# fsst-rs


A pure-Rust, zero-dependency implementation of the [FSST string compression algorithm][whitepaper].

FSST is a string compression algorithm meant for use in database systems. It was designed by
[Peter Boncz, Thomas Neumann, and Viktor Leis][whitepaper]. It provides 1-3GB/sec compression
and decompression of strings at compression rates competitive with or better than LZ4.

This implementation is somewhat inspired by the [MIT-licensed implementation] from the paper authors, written in C++,
but it is mostly written from a careful reading of the paper.

**NOTE: This current implementation is still in-progress and is not production ready, please use at your own risk.**

**NOTE: This crate only works on little-endian architectures currently. There are no current plans to support big-endian targets.**

## FSST12 variant

The `fsst::fsst12` module implements the 12-bit-code FSST variant from the
[cwida/fsst][MIT-licensed implementation] reference (also mentioned in the
[FastLanes File Format paper][fastlanes]). Codes are 12 bits wide (4096 entries), the first 256
codes are reserved as single-byte identity codes, and there is no escape mechanism. Single-byte
fallbacks still cost 1.5× their plaintext bytes, but the penalty is lighter than classic FSST's
2× escape cost.

```rust
use fsst::fsst12::Compressor12;

let compressor = Compressor12::train(&[b"the quick brown fox".as_slice()]);
let compressed = compressor.compress(b"the quick brown fox");
let decompressed = compressor.decompressor().decompress(&compressed);
assert_eq!(decompressed, b"the quick brown fox");
```

[whitepaper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
[MIT-licensed implementation]: https://github.com/cwida/fsst
[fastlanes]: https://www.vldb.org/pvldb/vol18/p4629-afroozeh.pdf

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2](https://github.com/spiraldb/fsst/compare/v0.4.1...v0.4.2) - 2024-09-17

### Fixed

- search for first non-empty chunk ([#35](https://github.com/spiraldb/fsst/pull/35))
- docs first3 ([#33](https://github.com/spiraldb/fsst/pull/33))

### Other

- Assertion should allow empty compression ([#36](https://github.com/spiraldb/fsst/pull/36))

## [0.4.1](https://github.com/spiraldb/fsst/compare/v0.4.0...v0.4.1) - 2024-09-12

### Other

- Use wrapping operations in fsst_hash ([#31](https://github.com/spiraldb/fsst/pull/31))

## [0.4.0](https://github.com/spiraldb/fsst/compare/v0.3.0...v0.4.0) - 2024-09-03

### Fixed
- hash_table_sizing, inline hints, lint rule ([#29](https://github.com/spiraldb/fsst/pull/29))

## [0.3.0](https://github.com/spiraldb/fsst/compare/v0.2.3...v0.3.0) - 2024-09-03

### Added
- port in more from the C++ code ([#24](https://github.com/spiraldb/fsst/pull/24))

### Other
- centering ([#26](https://github.com/spiraldb/fsst/pull/26))

## [0.2.3](https://github.com/spiraldb/fsst/compare/v0.2.2...v0.2.3) - 2024-08-22

### Added
- reuse and clear instead of allocate, 2x speedup ([#22](https://github.com/spiraldb/fsst/pull/22))

## [0.2.2](https://github.com/spiraldb/fsst/compare/v0.2.1...v0.2.2) - 2024-08-21

### Other
- implement second bitmap, ~2x speedup for train ([#21](https://github.com/spiraldb/fsst/pull/21))
- remove spurious check ([#18](https://github.com/spiraldb/fsst/pull/18))

## [0.2.1](https://github.com/spiraldb/fsst/compare/v0.2.0...v0.2.1) - 2024-08-20

### Added
- make Compressor::train 2x faster with bitmap index ([#16](https://github.com/spiraldb/fsst/pull/16))

## [0.2.0](https://github.com/spiraldb/fsst/compare/v0.1.0...v0.2.0) - 2024-08-20

### Other
- tput improvements ([#13](https://github.com/spiraldb/fsst/pull/13))

## [0.1.0](https://github.com/spiraldb/fsst/compare/v0.0.1...v0.1.0) - 2024-08-16

### Added
- separate Compressor and Decompressor ([#11](https://github.com/spiraldb/fsst/pull/11))

### Other
- add badges ([#10](https://github.com/spiraldb/fsst/pull/10))
- release v0.0.1 ([#8](https://github.com/spiraldb/fsst/pull/8))

## [0.0.1](https://github.com/spiraldb/fsst/releases/tag/v0.0.1) - 2024-08-15

### Fixed
- fix doc link

### Other
- turn on release-plz
- add fuzzer, fix bug ([#7](https://github.com/spiraldb/fsst/pull/7))
- logo ([#6](https://github.com/spiraldb/fsst/pull/6))
- bugfix, comment fix, force compile fails for big-endian ([#5](https://github.com/spiraldb/fsst/pull/5))
- Configure Renovate ([#1](https://github.com/spiraldb/fsst/pull/1))
- Get compress performance to match paper algorithm 4 ([#3](https://github.com/spiraldb/fsst/pull/3))
- docs
- cleanup
- words
- README
- disable release action for now
- deny(missing_docs), 512 -> 511
- add toolchain
- add actions files
- implementation v0
- initial impl
- Initial commit

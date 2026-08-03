# SIMD-Friendly Block Decoder

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current as work proceeds.

## Purpose / Big Picture

FSST decompression currently handles a no-escape block of eight codes as eight dependent operations: store an eight-byte symbol and advance the output pointer by that symbol's one-to-eight-byte logical length. The stores intentionally overlap, so this is correct, but the pointer dependency prevents LLVM from scheduling the block broadly or vectorizing it.

The goal is a decoder that preserves the existing public API and byte-for-byte behavior while making no-escape blocks friendlier to wide execution. The portable form will load all code lengths and compute prefix offsets before storing. On x86-64 with AVX2, a specialized form will pack pairs of symbols with byte shuffles and emit four 16-byte stores instead of eight 8-byte stores. Escape handling and the scalar tail remain unchanged. Success is observable through the existing and new correctness tests, generated assembly containing the intended shuffle/store sequence on x86-64, and CodSpeed results from a supported runner. Because the local machine is not recognized by CodSpeed, local wall-clock timings are not evidence for this plan.

## Progress

- [x] (2026-08-03 11:11Z) Read the decoder, benchmarks, test coverage, ExecPlan guide, CodSpeed workflow, and cargo-show-asm workflow.
- [x] (2026-08-03 11:11Z) Established an exact remote CodSpeed baseline for local commit `6a87d0c` and inspected the CF=8 decoder flamegraph.
- [x] (2026-08-03 11:17Z) Added a mixed one-to-eight-byte-symbol decompression benchmark that exercises only ordinary codes.
- [x] (2026-08-03 11:19Z) Refactored the no-escape block into a portable staged decoder without changing escape handling.
- [x] (2026-08-03 11:20Z) Added an x86-64 AVX2 pair-packing decoder with runtime feature dispatch and mask/block correctness coverage.
- [x] (2026-08-03 11:23Z) Inspected AArch64, x86-64 Haswell, and x86-64 AVX-512-capable generated code.
- [x] (2026-08-03 11:27Z) Ran formatting, 26 crate tests, native all-target checks, Clippy, x86-64 library cross-check, and CodSpeed-compatible benchmark validation.
- [x] (2026-08-03 11:27Z) Recorded final evidence and tradeoffs in this plan.

## Surprises & Discoveries

- Observation: LLVM does not vectorize the normal eight-code path even when cross-compiling with AVX2 or AVX-512 features. The per-symbol output-pointer advance is a loop-carried dependency, and the stores may overlap.
  Evidence: `cargo asm` for `x86_64-apple-darwin` showed scalar loads and stores in the normal path. With AVX-512 enabled, LLVM only vectorized extraction of four raw bytes in the all-escape special case.

- Observation: The exact CodSpeed baseline exists remotely for local commit `6a87d0c`, but this local machine is reported as an unknown environment and records no measurements.
  Evidence: remote run `6a706e5dbd033b11959de1ba`; local `cargo codspeed run -m simulation --bench micro 'decompress'` completed with a notice that no performance measurement would be made.

- Observation: In the CF=8 one-megabyte baseline, the decoder accounts for 99.31% of 2.1 ms and `write_unaligned::<u64>` accounts for 61.49% self cost. Its modeled cost is dominated by memory and cache effects.
  Evidence: CodSpeed simulation flamegraph for `decompress-regimes/1mb::decompress-into-reuse-cf8` in run `6a706e5dbd033b11959de1ba`.

- Observation: Packing the length table into nibbles adds scalar shifts and is worse in generated code. Merely asserting the one-to-eight invariant produces LLVM assumptions but does not change final assembly.
  Evidence: isolated code-generation experiments made before this plan.

- Observation: An exactly one-megabyte repetition truncated the final mixed symbol and correctly caused an escape code, which would have contaminated the new benchmark.
  Evidence: the benchmark's ordinary-code assertion failed until the corpus length was rounded up to an integral number of the 36-byte seed sequence.

- Observation: The generic implementation fully inlines into the AVX2-targeted entry point. Runtime feature detection is performed once and the public entry point tail-jumps to the specialized decoder; there is no target-feature call inside the block loop.
  Evidence: default x86-64 assembly has a cached feature test and a tail jump to `<fsst::Decompressor>::decompress_into_avx2`. Haswell assembly for the no-escape block contains two `vpshufb`, two low-lane `vmovdqu` stores, and two high-lane `vextracti128` stores.

- Observation: The portable AArch64 form loads all eight lengths and computes their prefix offsets before issuing the eight symbol stores. LLVM emits no hot bounds checks for the staged block.
  Evidence: host `cargo asm` output for `<fsst::Decompressor>::decompress_into`.

- Observation: Cross-checking all x86-64 targets is blocked by the development-only `curl` dependency's `openssl-sys` build script, which cannot locate a cross-compilation OpenSSL sysroot. The library itself cross-checks and cargo-show-asm builds the optimized library successfully.
  Evidence: `cargo check --lib --target x86_64-apple-darwin` passes; `cargo check --all-targets --target x86_64-apple-darwin` fails in `openssl-sys` before compiling repository targets.

## Decision Log

- Decision: Keep lengths as bytes and focus packing effort on the output symbols.
  Rationale: AVX2 has no byte gather, while a 255-byte length table is already compact and hot. Nibble packing increases instruction count without removing indexed loads.
  Date/Author: 2026-08-03 / Codex

- Decision: Stage scalar lengths and prefix offsets for every architecture, then use AVX2 byte shuffles only on x86-64 CPUs that report AVX2.
  Rationale: Staging removes the artificial output-pointer dependency and remains a straightforward portable fallback. AVX2 `vpshufb` can concatenate adjacent symbols inside independent 128-bit lanes, reducing the dominant store count without requiring AVX-512 byte compression.
  Date/Author: 2026-08-03 / Codex

- Decision: Preserve the existing escape branches and tail logic in this iteration.
  Rationale: The evidence identifies symbol stores in the no-escape path as the opportunity. Keeping escape behavior stable limits correctness risk and makes the benchmark comparison attributable.
  Date/Author: 2026-08-03 / Codex

- Decision: Do not claim a speedup from local timing.
  Rationale: The project's required measurement mechanism, CodSpeed, cannot measure on this host. Correctness and assembly can validate the implementation locally; performance acceptance requires a supported CodSpeed run.
  Date/Author: 2026-08-03 / Codex

- Decision: Validate the zero-to-eight table invariant once in `Decompressor::new` instead of asserting lengths in the hot decode block.
  Rationale: Unpopulated table entries legitimately use zero; populated symbols use one through eight. Constructor validation makes the unchecked shuffle-mask lookup and scalar tail copy sound without adding eight branches per block.
  Date/Author: 2026-08-03 / Codex

- Decision: Monomorphize the decoder into scalar and AVX2 entry points and accept the added machine-code size for this prototype.
  Rationale: A target-feature helper called once per block would spend a function call to save four store instructions. One dispatch per decompression preserves the intended hot-block code shape. CodSpeed still needs to decide whether the throughput gain justifies code-size duplication.
  Date/Author: 2026-08-03 / Codex

- Decision: Skip feature detection when the input or destination is too short to enter the eight-code block loop.
  Rationale: Short decompression benchmarks otherwise pay dispatch overhead but cannot execute a SIMD block.
  Date/Author: 2026-08-03 / Codex

## Outcomes & Retrospective

The implementation and generated-code goals are complete. The retained prototype has a portable staged fallback and a runtime-dispatched AVX2 decoder. The AVX2 block uses two lane-local byte shuffles and four 16-byte stores for eight symbols; the fallback retains eight stores but removes the interleaved address dependency. Runtime detection is skipped for inputs that cannot enter the block loop. The constructor now rejects lengths over eight once, allowing hot unchecked indexing without per-block assertions.

All 26 tests pass, including mixed block, exact-capacity, property-based, mask-table, escape, and large round-trip coverage. Native all-target checking and Clippy pass. The x86-64 library cross-check and Haswell/Skylake assembly generation pass; x86-64 all-target checking reaches an unrelated development-only OpenSSL sysroot limitation. The decompression benchmark suite builds and all seven workloads execute under cargo-codspeed, including the new mixed-length case.

The unresolved question is whether halving store instructions outweighs vector construction and shuffle work on supported x86-64 CPUs. This host cannot record CodSpeed measurements, so no speedup is claimed. A supported CodSpeed comparison should decide whether to retain the machine-code duplication and AVX2 path. If it does not improve representative text workloads, the staged scalar refactor and mixed benchmark can remain while the AVX2 specialization is removed cleanly.

## Context and Orientation

The crate is a Rust FSST implementation. `src/lib.rs` defines `Decompressor::decompress_into`, which accepts compressed code bytes and a `MaybeUninit<u8>` destination. Its main loop reads eight compressed bytes into a little-endian `u64`, detects the reserved escape code `255`, and handles three cases: no escapes, four even-position escapes, or a first escape at an arbitrary position. The no-escape case is the target of this work.

Each ordinary code indexes two decompressor tables: `symbols: &[Symbol; 255]` contains an eight-byte padded value, and `lengths: &[u8; 255]` contains its logical length. Existing code writes all eight padded bytes at the current output address, then advances only by the logical length. A later store overwrites padding from an earlier store. The fast loop keeps at least 64 writable bytes available so its eight-byte over-writes remain within the destination.

`benches/micro.rs` contains Criterion benchmarks that are compiled and executed through cargo-codspeed. The existing CF=8 decompression case is valuable but uses a single length. A mixed-length benchmark must alternate codes whose lengths cover one through eight so it measures prefix-offset and packing overhead rather than only the easiest length.

`tests/exact_capacity.rs` contains exact-capacity and property-based decompression coverage, including mixed symbols and escapes. Unit tests near the target-specific helper should additionally compare a packed block against the scalar expected bytes when the relevant CPU feature is available.

## Plan of Work

First, add a one-megabyte benchmark constructed from eight explicit symbols with lengths one through eight. Compress a repeated concatenation of those symbols, verify the compressed representation contains ordinary codes, and reuse a destination allocation in the measured closure. Build and attempt to run it through cargo-codspeed; the expected local result is compilation plus the known unsupported-runner notice.

Second, introduce a small internal representation of an eight-code no-escape block. Extract the eight codes, load their lengths, and compute exclusive prefix offsets plus the total decoded length. The portable writer will store each padded symbol at its precomputed offset. This keeps the aliasing/overwrite semantics visible and removes successive mutations of the output pointer from address calculation.

Third, for `target_arch = "x86_64"`, add an AVX2 writer. It will load four adjacent symbol pairs into two 256-bit values. A 16-byte shuffle mask selected by each first symbol's length concatenates the pair inside each independent 128-bit lane. Four unaligned 16-byte stores write the four packed pairs at offsets for codes 0, 2, 4, and 6. Stores remain ordered; excess bytes from one pair are overwritten by the next, just as in the scalar decoder. Runtime feature detection occurs once per `decompress_into` call, and non-AVX2 hosts use the portable writer.

Fourth, run focused and full crate correctness checks. Inspect the relevant functions with cargo-show-asm on the AArch64 host and by cross-compiling for `x86_64-apple-darwin` with Haswell/AVX2 and a newer AVX-512 feature set. Count loads, shuffles, stores, branches, and spills in the no-escape path. If the target-specific path contains unexpected scalar packing or spills, simplify it before retention.

Finally, build and attempt the decompression benchmarks through cargo-codspeed. If a supported CodSpeed comparison becomes available, compare the exact baseline run against the candidate, prioritizing database text and mixed-length cases as well as CF=8 and all-escape regression checks. If it is unavailable, leave the performance claim explicitly pending rather than substituting another timer.

## Concrete Steps

Run all commands from `/Users/adamgs/code/fsst`.

1. Add the mixed benchmark and build it:

       cargo fmt --check
       cargo codspeed build -m simulation --bench micro
       cargo codspeed run -m simulation --bench micro 'decompress'

2. After each decoder edit, run the affected crate tests first:

       cargo nextest run
       cargo check --all-targets

3. Inspect host code generation:

       cargo asm --lib --release '<fsst::Decompressor>::decompress_into'

4. Inspect x86-64 code generation with AVX2 and newer features:

       cargo asm --target=x86_64-apple-darwin --target-cpu=haswell --lib --release --asm decompress_into
       cargo asm --target=x86_64-apple-darwin --target-cpu=skylake-avx512 --lib --release --asm decompress_into
       cargo asm --target=x86_64-apple-darwin --lib --release --asm decompress_into 0
       cargo asm --target=x86_64-apple-darwin --lib --release --asm decompress_into 1

5. Re-run formatting, nextest, all-target checks, CodSpeed build/run, and `git diff --check` before completion.

## Validation and Acceptance

Correctness acceptance requires `cargo nextest run` to pass, `cargo check --all-targets` to pass, and the target-specific packing test to produce exactly the concatenation dictated by all symbol lengths one through eight. Existing exact-capacity and escape-heavy tests must continue to pass.

Generated-code acceptance requires the x86-64 AVX2 helper to contain packed byte-shuffle operations and no more than four 16-byte stores for an eight-code block. The portable AArch64 and x86-64 paths must compute offsets before symbol writes and must not introduce bounds-check branches in the hot block.

Performance acceptance requires a supported CodSpeed comparison with no meaningful regression in all-escape decompression and a useful improvement in at least one representative no-escape or mixed-length workload. Until such a run exists, the implementation can be delivered as a tested prototype with assembly evidence, not as a proven optimization.

## Idempotence and Recovery

All build, test, assembly, and benchmark commands are repeatable. Generated artifacts remain under Cargo's target directories. Source edits are localized to `src/lib.rs`, `benches/micro.rs`, tests if needed, and this plan. If the AVX2 helper is incorrect or produces poor code, remove its runtime branch and helper while retaining the independently useful staged portable decoder and benchmark. Do not reset unrelated working-tree changes.

## Artifacts and Notes

- Baseline commit: `6a87d0c`
- Baseline CodSpeed run: `6a706e5dbd033b11959de1ba`
- CF=8 baseline: approximately 2.1 ms for `decompress-regimes/1mb::decompress-into-reuse-cf8` in CodSpeed simulation.
- Local cargo-codspeed: 4.4.1; build succeeds, runner reports an unknown environment and records no measurement.
- Haswell no-escape block: two `vpshufb`, two low-lane `vmovdqu` stores, and two high-lane `vextracti128` stores; no helper call.
- Default x86-64 entry point: cheap size checks precede cached AVX2 feature detection, followed by a tail jump to the specialized decoder on supported CPUs.
- AArch64 fallback: eight byte length loads and prefix additions occur before the first of eight `str` symbol stores.
- Final native validation: `cargo nextest run` passed 26 tests; `cargo check --all-targets`, `cargo clippy --all-targets`, and `cargo fmt --all --check` passed.
- Final cross validation: `cargo check --lib --target x86_64-apple-darwin` passed. The all-target variant is blocked in the `openssl-sys` build script because no x86-64 OpenSSL cross sysroot is configured.
- Final benchmark validation: cargo-codspeed built `micro` and checked all seven decompression workloads, but recorded no measurements on the unknown local environment.

## Interfaces and Dependencies

No public interface changes are planned. `Decompressor::decompress_into`, `Decompressor::decompress`, and table representations remain stable. The implementation uses only `core`/`std` architecture intrinsics already supplied by Rust; no crate dependency is added. The x86-64 helper requires AVX2 and is called only after `is_x86_feature_detected!("avx2")` succeeds. Other architectures use the portable implementation.

Revision note (2026-08-03 11:11Z): Created the plan after baseline assembly, CodSpeed, and flamegraph investigation so implementation decisions and the unavailable local measurement constraint are explicit before source changes.

Revision note (2026-08-03 11:25Z): Recorded the completed benchmark, scalar staging, AVX2 pair-packing implementation, exact generated-code result, constructor invariant, dispatch refinement, focused correctness result, and the development-dependency limitation on all-target x86 cross-checking.

Revision note (2026-08-03 11:27Z): Closed local validation, documented all passing checks and the exact cross-target limitation, and recorded the final outcome as a correct codegen-verified prototype pending supported CodSpeed measurements.

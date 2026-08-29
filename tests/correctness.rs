//! Correctness tests for FSST.

#![cfg(test)]

use fsst::{Compressor, CompressorBuilder, Symbol};

static PREAMBLE: &str = r#"
When in the Course of human events, it becomes necessary for one people to dissolve
the political bands which have connected them with another, and to assume among the
powers of the earth, the separate and equal station to which the Laws of Nature and
of Nature's God entitle them, a decent respect to the opinions of mankind requires
that they should declare the causes which impel them to the separation."#;

static DECLARATION: &str = include_str!("./fixtures/declaration.txt");

static ART_OF_WAR: &str = include_str!("./fixtures/art_of_war.txt");

/// Miri interprets every memory access, so the cost of these tests scales with the size of the
/// corpus rather than being dominated by fixed setup. The unsafe code paths under test are hit
/// just as thoroughly by a small corpus, so the size-sensitive tests below scale their inputs
/// down under miri.
const fn scaled(full: usize, under_miri: usize) -> usize {
    if cfg!(miri) { under_miri } else { full }
}

fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn training_golden(input: &str) -> (usize, u64, usize, u64) {
    let trained = Compressor::train(&vec![input.as_bytes()]);
    let table_fingerprint = fnv1a64(
        trained
            .symbol_table()
            .iter()
            .flat_map(|symbol| symbol.to_u64().to_le_bytes())
            .chain(trained.symbol_lengths().iter().copied()),
    );
    let compressed = trained.compress(input.as_bytes());

    (
        trained.n_symbols(),
        table_fingerprint,
        compressed.len(),
        fnv1a64(compressed),
    )
}

// Full-corpus training is prohibitively slow under Miri.
#[cfg_attr(miri, ignore)]
#[test]
fn test_training_is_cross_architecture_deterministic() {
    // These goldens guard against x86_64 and aarch64 hashbrown iteration-order differences.
    // When training intentionally changes, regenerate them by temporarily printing these tuples.
    assert_eq!(
        training_golden(DECLARATION),
        (243, 2302118744919910234, 3736, 16602696328334332958),
    );
    assert_eq!(
        training_golden(ART_OF_WAR),
        (239, 11323366151389446290, 4744, 10727487692352854482),
    );
}

#[test]
fn test_basic() {
    // Roundtrip the declaration
    let trained = Compressor::train(&vec![PREAMBLE.as_bytes()]);
    let compressed = trained.compress(PREAMBLE.as_bytes());
    let decompressed = trained.decompressor().decompress(&compressed);
    assert_eq!(decompressed, PREAMBLE.as_bytes());
}

#[test]
fn test_train_on_empty() {
    let trained = Compressor::train(&vec![]);
    // We can still compress with it, but the symbols are going to be empty.
    let compressed = trained.compress("the quick brown fox jumped over the lazy dog".as_bytes());
    assert_eq!(
        trained.decompressor().decompress(&compressed),
        "the quick brown fox jumped over the lazy dog".as_bytes()
    );
}

#[test]
fn test_one_byte() {
    let mut empty = CompressorBuilder::new();
    empty.insert(Symbol::from_u8(0x01), 1);

    let empty = empty.build();

    let compressed = empty.compress(&[0x01]);
    assert_eq!(compressed, vec![0u8]);

    assert_eq!(empty.decompressor().decompress(&compressed), vec![0x01]);
}

#[test]
fn test_zeros() {
    let training_data: Vec<u8> = vec![0, 1, 2, 3, 4, 0];
    let trained = Compressor::train(&vec![&training_data]);
    let compressed = trained.compress(&[4, 0]);
    assert_eq!(trained.decompressor().decompress(&compressed), &[4, 0]);
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_large() {
    let corpus: Vec<u8> = DECLARATION.bytes().cycle().take(10_240).collect();

    let trained = Compressor::train(&vec![&corpus]);
    let massive: Vec<u8> = DECLARATION
        .bytes()
        .cycle()
        .take(16 * 1_024 * 1_024)
        .collect();

    let compressed = trained.compress(&massive);
    assert_eq!(trained.decompressor().decompress(&compressed), massive);
}

#[test]
fn test_chinese() {
    // A byte prefix is enough under miri: this is a byte-level roundtrip, and the multi-byte
    // UTF-8 sequences that make this case interesting are dense from the very first byte.
    let corpus = &ART_OF_WAR.as_bytes()[..scaled(ART_OF_WAR.len(), 1_024)];

    let trained = Compressor::train(&vec![corpus]);
    assert_eq!(
        corpus,
        trained.decompressor().decompress(&trained.compress(corpus))
    );
}

#[test]
fn test_all_escape_roundtrip() {
    // Empty symbol table: every byte is encoded as ESCAPE + raw byte.
    let compressor = CompressorBuilder::new().build();
    let decompressor = compressor.decompressor();

    // Large enough to exercise the 8-byte block loop in decompress_into.
    let input: Vec<u8> = (0..=255u8).cycle().take(scaled(4096, 512)).collect();
    let compressed = compressor.compress(&input);
    // All-escape compressed size should be exactly 2x input.
    assert_eq!(compressed.len(), input.len() * 2);
    assert_eq!(decompressor.decompress(&compressed), input);
}

#[test]
fn test_invalid_code_not_in_symbol_works() {
    let compressor = CompressorBuilder::new().build();
    let decompressor = compressor.decompressor();

    // Empty symbol table: code 0 is malformed input, not a valid symbol code.
    // Use more than 8 bytes so the unrolled decode loop is exercised.
    let _ = decompressor.decompress(&[0; 9]);
}

#[test]
#[should_panic]
fn test_invalid_tail_code_not_in_symbol_table_panics() {
    let compressor = CompressorBuilder::new().build();
    let decompressor = compressor.decompressor();
    let mut decoded = [];

    // A one-byte malformed input reaches the final byte-copy fallback path.
    let _ = decompressor.decompress_into(&[0], &mut decoded);
}

#[test]
fn test_large_with_rebuild() {
    let corpus: Vec<u8> = DECLARATION
        .bytes()
        .cycle()
        .take(scaled(10_240, 1_024))
        .collect();
    // `DECLARATION` is pure ASCII, so slicing it at a byte index stays on a char boundary.
    let text = &DECLARATION[..scaled(DECLARATION.len(), 1_024)];

    let trained = Compressor::train(&vec![&corpus]);
    let compressed = trained.compress(text.as_bytes());

    let rebuilt = Compressor::rebuild_from(trained.symbol_table(), trained.symbol_lengths());
    let recompressed = rebuilt.compress(text.as_bytes());

    assert_eq!(compressed, recompressed);

    // Ensure round-trip after rebuilding the compressor
    let decompressed = rebuilt.decompressor().decompress(&recompressed);
    assert_eq!(
        unsafe { std::str::from_utf8_unchecked(&decompressed) },
        text,
    );
}

#[test]
fn test_pruning_small_input() {
    // 'a' × 100 plus bytes 200..210 appearing once each.
    // Without pruning, the count >= 5 filter drops the rare bytes.
    // With pruning, the count threshold is lowered to 1, but
    // saves (1) <= cost (1+1=2) still filters them out.
    // Bytes 0xFF appears 3 times: passes the lowered threshold,
    // AND saves (3) > cost (2), so pruning keeps it.
    // This proves the pruning path is active: 0xFF would be dropped
    // by the normal count >= 5 filter but survives via pruning.
    let mut corpus = vec![b'a'; 100];
    corpus.extend(200u8..210);
    corpus.extend([0xFF, 0xFF, 0xFF]);

    // Use multiple sample lines so earlier training generations see data.
    let compressor = Compressor::train(&vec![&corpus[..30], &corpus[30..60], &corpus[60..]]);

    // 0xFF (count=3) survives: pruning lowers the count threshold to 1,
    // and saves (3) > cost (2). Bytes 200..210 (count=1) are pruned.
    assert_eq!(
        &compressor.symbol_table()[0..compressor.n_symbols()],
        &[
            Symbol::from_slice(b"aa\0\0\0\0\0\0"),
            Symbol::from_slice(b"aaaaaaaa"),
            Symbol::from_u8(b'a'),
            Symbol::from_u8(0xFF),
        ],
    );

    let compressed = compressor.compress(&corpus);
    assert_eq!(compressor.decompressor().decompress(&compressed), corpus);
}

/// The compression loop evaluates the masked symbol comparison before it has established that
/// the hash slot is in use, so an unused slot (whose stored symbol is zero and whose
/// `ignored_bits` is 64) must never be mistaken for a match. An all-zero word probing an unused
/// slot is the case that catches it: with a full-width mask, `word & mask` would equal the
/// slot's zero symbol.
#[test]
fn test_zero_word_does_not_match_unused_hash_slot() {
    let mut builder = CompressorBuilder::new();
    // A single three-byte symbol, so the hash table has exactly one slot in use.
    assert!(builder.insert(Symbol::from_slice(b"xyz\0\0\0\0\0"), 3));
    let compressor = builder.build();

    for len in [1usize, 7, 8, 9, 16, 33, 64] {
        let corpus = vec![0u8; len];
        let compressed = compressor.compress(&corpus);
        assert_eq!(
            compressor.decompressor().decompress(&compressed),
            corpus,
            "all-zero input of length {len} did not round trip"
        );
        // Every byte is an escape: no symbol in the table matches a zero byte.
        assert_eq!(
            compressed.len(),
            2 * len,
            "length {len} should be all escapes"
        );
    }
}

/// `compress_bulk_into` must produce, for every value, exactly the bytes `compress` would
/// produce for that value on its own, so any single value can be decompressed from its range.
fn check_bulk(compressor: &Compressor, values: &[&[u8]], label: &str) {
    let mut output = Vec::new();
    let mut offsets = Vec::new();
    compressor.compress_bulk_into(values, &mut output, &mut offsets);

    assert_eq!(offsets.len(), values.len(), "{label}: one offset per value");
    let mut start = 0usize;
    for (i, value) in values.iter().enumerate() {
        let end = offsets[i] as usize;
        assert!(
            start <= end && end <= output.len(),
            "{label}: value {i} has a bogus range"
        );
        assert_eq!(
            &output[start..end],
            compressor.compress(value),
            "{label}: value {i}"
        );
        assert_eq!(
            compressor.decompressor().decompress(&output[start..end]),
            *value,
            "{label}: value {i} round trip"
        );
        start = end;
    }
    assert_eq!(start, output.len(), "{label}: output has trailing bytes");
}

#[test]
fn test_compress_bulk_into_matches_compress() {
    let lines: Vec<&[u8]> = DECLARATION.as_bytes().split(|&b| b == b'\n').collect();
    let compressor = Compressor::train(&lines);
    check_bulk(&compressor, &lines, "declaration");

    // Fewer values than cursors, and counts either side of the cursor count, exercise the
    // serial fallback and the drain that runs once the first cursor runs out of values.
    for n in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17] {
        check_bulk(
            &compressor,
            &lines[..n.min(lines.len())],
            &format!("declaration[..{n}]"),
        );
    }
}

#[test]
fn test_compress_bulk_into_ragged_values() {
    let long = &ART_OF_WAR.as_bytes()[..scaled(ART_OF_WAR.len(), 512)];
    let compressor = Compressor::train(&vec![long]);

    // Values shorter than a word never enter the interleaved loop at all, and empty values must
    // still get an offset.
    let values: Vec<&[u8]> = vec![
        b"",
        b"a",
        b"",
        b"abcdefg",
        b"abcdefgh",
        b"abcdefghi",
        long,
        b"",
        b"zz",
    ];
    check_bulk(&compressor, &values, "ragged");

    // One long value beside many short ones leaves the cursors badly unbalanced.
    let mut skewed: Vec<&[u8]> = vec![long];
    skewed.extend(std::iter::repeat_n(b"xy".as_slice(), scaled(64, 8)));
    check_bulk(&compressor, &skewed, "skewed");
}

#[test]
fn test_compress_bulk_into_appends() {
    let lines: Vec<&[u8]> = PREAMBLE.as_bytes().split(|&b| b == b'\n').collect();
    let compressor = Compressor::train(&lines);

    let mut fresh_output = Vec::new();
    let mut fresh_offsets = Vec::new();
    compressor.compress_bulk_into(&lines, &mut fresh_output, &mut fresh_offsets);

    // Existing contents are preserved, and the offsets returned are absolute indices into the
    // output, so they are shifted by whatever was already there.
    let mut output = vec![0xAA; 5];
    let mut offsets = vec![u64::MAX; 2];
    compressor.compress_bulk_into(&lines, &mut output, &mut offsets);

    assert_eq!(&output[..5], &[0xAA; 5]);
    assert_eq!(&offsets[..2], &[u64::MAX; 2]);
    assert_eq!(&output[5..], &fresh_output[..]);
    for (shifted, fresh) in offsets[2..].iter().zip(&fresh_offsets) {
        assert_eq!(*shifted, fresh + 5);
    }
}

#[test]
fn test_compress_bulk_into_cursor_counts_agree() {
    let corpus = &DECLARATION.as_bytes()[..scaled(DECLARATION.len(), 512)];
    let lines: Vec<&[u8]> = corpus.split(|&b| b == b'\n').collect();
    let compressor = Compressor::train(&lines);

    let mut expected_output = Vec::new();
    let mut expected_offsets = Vec::new();
    compressor.compress_bulk_lanes::<1>(&lines, &mut expected_output, &mut expected_offsets);

    // The cursor count is a scheduling choice; it must not change a single output byte.
    fn compare<const K: usize>(c: &Compressor, lines: &[&[u8]], output: &[u8], offsets: &[u64]) {
        let mut got_output = Vec::new();
        let mut got_offsets = Vec::new();
        c.compress_bulk_lanes::<K>(lines, &mut got_output, &mut got_offsets);
        assert_eq!(got_output, output, "K={K} output");
        assert_eq!(got_offsets, offsets, "K={K} offsets");
    }
    compare::<2>(&compressor, &lines, &expected_output, &expected_offsets);
    compare::<3>(&compressor, &lines, &expected_output, &expected_offsets);
    compare::<4>(&compressor, &lines, &expected_output, &expected_offsets);
    compare::<8>(&compressor, &lines, &expected_output, &expected_offsets);
}

/// Every cursor is handed an output slice sized at two bytes per byte of its own input, which is
/// exactly what an all-escape value consumes. An empty symbol table makes every value hit that
/// bound at once, so nothing is left over for a cursor that overruns its slice.
#[test]
fn test_compress_bulk_into_all_escape_fills_every_slice() {
    let compressor = CompressorBuilder::new().build();

    // Lengths either side of a word, so cursors reach the bound both inside the interleaved
    // loop and in the per-value remainder.
    let owned: Vec<Vec<u8>> = (0..scaled(64, 12))
        .map(|i| (0..=255u8).cycle().take(i * 3 + 1).collect())
        .collect();
    let values: Vec<&[u8]> = owned.iter().map(|value| value.as_slice()).collect();

    let mut output = Vec::new();
    let mut offsets = Vec::new();
    compressor.compress_bulk_into(&values, &mut output, &mut offsets);

    let mut start = 0usize;
    for (i, value) in values.iter().enumerate() {
        let end = offsets[i] as usize;
        assert_eq!(
            end - start,
            value.len() * 2,
            "value {i} should be all escapes"
        );
        assert_eq!(
            compressor.decompressor().decompress(&output[start..end]),
            *value
        );
        start = end;
    }
    assert_eq!(start, output.len());
}

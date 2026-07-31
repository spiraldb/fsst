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

    // `symbol_table()` / `symbol_lengths()` are padded to 255 entries, so they have to be sliced
    // to `n_symbols()` before being handed back to `rebuild_from`. Passing the padded arrays
    // makes the rebuilt table disagree with the trained one whenever the table is not full.
    let n_symbols = trained.n_symbols();
    let rebuilt = Compressor::rebuild_from(
        &trained.symbol_table()[..n_symbols],
        &trained.symbol_lengths()[..n_symbols],
    );
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

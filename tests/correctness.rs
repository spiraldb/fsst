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
    let trained = Compressor::train(&vec![ART_OF_WAR.as_bytes()]);
    assert_eq!(
        ART_OF_WAR.as_bytes(),
        trained
            .decompressor()
            .decompress(&trained.compress(ART_OF_WAR.as_bytes()))
    );
}

#[test]
fn test_all_escape_roundtrip() {
    // Empty symbol table: every byte is encoded as ESCAPE + raw byte.
    let compressor = CompressorBuilder::new().build();
    let decompressor = compressor.decompressor();

    // Large enough to exercise the 8-byte block loop in decompress_into.
    let input: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
    let compressed = compressor.compress(&input);
    // All-escape compressed size should be exactly 2x input.
    assert_eq!(compressed.len(), input.len() * 2);
    assert_eq!(decompressor.decompress(&compressed), input);
}

#[test]
fn test_large_with_rebuild() {
    let corpus: Vec<u8> = DECLARATION.bytes().cycle().take(10_240).collect();

    let trained = Compressor::train(&vec![&corpus]);
    let compressed = trained.compress(DECLARATION.as_bytes());

    // let compressed = trained.compress(&massive);
    let rebuilt = Compressor::rebuild_from(trained.symbol_table(), trained.symbol_lengths());
    let recompressed = rebuilt.compress(DECLARATION.as_bytes());

    assert_eq!(compressed, recompressed);

    // Ensure round-trip after rebuilding the compressor
    let decompressed = rebuilt.decompressor().decompress(&recompressed);
    assert_eq!(
        unsafe { std::str::from_utf8_unchecked(&decompressed) },
        DECLARATION,
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
        compressor.symbol_table(),
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

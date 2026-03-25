//! Correctness tests for 12-bit FSST.

#![cfg(test)]

use fsst::Symbol;
use fsst::fsst12::{Compressor12, CompressorBuilder12};

static PREAMBLE: &str = r#"
When in the Course of human events, it becomes necessary for one people to dissolve
the political bands which have connected them with another, and to assume among the
powers of the earth, the separate and equal station to which the Laws of Nature and
of Nature's God entitle them, a decent respect to the opinions of mankind requires
that they should declare the causes which impel them to the separation."#;

static DECLARATION: &str = include_str!("./fixtures/declaration.txt");

static ART_OF_WAR: &str = include_str!("./fixtures/art_of_war.txt");

#[test]
fn test_basic_12() {
    let trained = Compressor12::train(&vec![PREAMBLE.as_bytes()]);
    let compressed = trained.compress(PREAMBLE.as_bytes());
    let decompressed = trained.decompressor().decompress(&compressed);
    assert_eq!(decompressed, PREAMBLE.as_bytes());
}

#[test]
fn test_train_on_empty_12() {
    let trained = Compressor12::train(&vec![]);
    let compressed = trained.compress("the quick brown fox jumped over the lazy dog".as_bytes());
    assert_eq!(
        trained.decompressor().decompress(&compressed),
        "the quick brown fox jumped over the lazy dog".as_bytes()
    );
}

#[test]
fn test_zeros_12() {
    let training_data: Vec<u8> = vec![0, 1, 2, 3, 4, 0];
    let trained = Compressor12::train(&vec![&training_data]);
    let compressed = trained.compress(&[4, 0]);
    assert_eq!(trained.decompressor().decompress(&compressed), &[4, 0]);
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_large_12() {
    let corpus: Vec<u8> = DECLARATION.bytes().cycle().take(10_240).collect();

    let trained = Compressor12::train(&vec![&corpus]);
    let massive: Vec<u8> = DECLARATION
        .bytes()
        .cycle()
        .take(16 * 1_024 * 1_024)
        .collect();

    let compressed = trained.compress(&massive);
    assert_eq!(trained.decompressor().decompress(&compressed), massive);
}

#[test]
fn test_chinese_12() {
    let trained = Compressor12::train(&vec![ART_OF_WAR.as_bytes()]);
    assert_eq!(
        ART_OF_WAR.as_bytes(),
        trained
            .decompressor()
            .decompress(&trained.compress(ART_OF_WAR.as_bytes()))
    );
}

#[test]
fn test_all_identity_roundtrip_12() {
    // Empty symbol table: every byte is an identity code.
    let compressor = CompressorBuilder12::new().build();
    let decompressor = compressor.decompressor();

    let input: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
    let compressed = compressor.compress(&input);
    assert_eq!(decompressor.decompress(&compressed), input);
}

#[test]
fn test_large_with_rebuild_12() {
    let corpus: Vec<u8> = DECLARATION.bytes().cycle().take(10_240).collect();

    let trained = Compressor12::train(&vec![&corpus]);
    let compressed = trained.compress(DECLARATION.as_bytes());

    let rebuilt = Compressor12::rebuild_from(trained.symbol_table(), trained.symbol_lengths());
    let recompressed = rebuilt.compress(DECLARATION.as_bytes());

    assert_eq!(compressed, recompressed);

    let decompressed = rebuilt.decompressor().decompress(&recompressed);
    assert_eq!(
        unsafe { std::str::from_utf8_unchecked(&decompressed) },
        DECLARATION,
    );
}

#[test]
fn test_manual_symbol_12() {
    // Insert a known multi-byte symbol and verify round-trip.
    let mut builder = CompressorBuilder12::new();
    builder.insert(Symbol::from_slice(b"hello\0\0\0"), 5);
    let compressor = builder.build();

    let compressed = compressor.compress(b"hellohello");
    let decompressed = compressor.decompressor().decompress(&compressed);
    assert_eq!(&decompressed, b"hellohello");
}

#[test]
fn test_odd_even_code_count_12() {
    let compressor = CompressorBuilder12::new().build();
    let decompressor = compressor.decompressor();

    // Odd number of codes (1, 3, 5, 7).
    for len in [1, 3, 5, 7] {
        let input: Vec<u8> = (0..len).map(|i| (i as u8) + 10).collect();
        let compressed = compressor.compress(&input);
        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(decompressed, input, "failed for input length {len}");
    }

    // Even number of codes (2, 4, 6, 8).
    for len in [2, 4, 6, 8] {
        let input: Vec<u8> = (0..len).map(|i| (i as u8) + 10).collect();
        let compressed = compressor.compress(&input);
        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(decompressed, input, "failed for input length {len}");
    }
}

#[test]
fn test_json_compression_12() {
    // FSST12 is designed to handle diverse data like JSON well.
    let json = br#"[
        {"id": 1, "name": "Alice", "email": "alice@example.com", "active": true},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "active": false},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "active": true}
    ]"#;

    let corpus: Vec<&[u8]> = std::iter::repeat_n(json.as_slice(), 50).collect();
    let trained = Compressor12::train(&corpus);

    let compressed = trained.compress(json);
    let decompressed = trained.decompressor().decompress(&compressed);
    assert_eq!(&decompressed, json);

    // Should achieve compression.
    assert!(
        compressed.len() < json.len(),
        "12-bit: compressed={} should be smaller than original={}",
        compressed.len(),
        json.len()
    );
}

#[test]
fn test_compressed_output_lengths_12() {
    // Verify the 12-bit packing: n codes → ⌊n/2⌋*3 + (n%2)*2 bytes.
    let compressor = CompressorBuilder12::new().build();

    // 1 byte input → 1 code → 2 bytes compressed (odd trailing).
    let c1 = compressor.compress(&[42]);
    assert_eq!(c1.len(), 2);

    // 2 bytes → 2 codes → 3 bytes (one pair).
    let c2 = compressor.compress(&[1, 2]);
    assert_eq!(c2.len(), 3);

    // 3 bytes → 3 codes → 5 bytes (one pair + trailing).
    let c3 = compressor.compress(&[1, 2, 3]);
    assert_eq!(c3.len(), 5);

    // 4 bytes → 4 codes → 6 bytes (two pairs).
    let c4 = compressor.compress(&[1, 2, 3, 4]);
    assert_eq!(c4.len(), 6);
}

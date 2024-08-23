#![cfg(test)]

use fsst::Compressor;

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
    let trained = Compressor::train(PREAMBLE);
    let compressed = trained.compress(PREAMBLE.as_bytes());
    let decompressed = trained.decompressor().decompress(&compressed);
    assert_eq!(decompressed, PREAMBLE.as_bytes());
}

#[test]
fn test_train_on_empty() {
    let trained = Compressor::train("");
    // We can still compress with it, but the symbols are going to be empty.
    let compressed = trained.compress("the quick brown fox jumped over the lazy dog".as_bytes());
    assert_eq!(
        trained.decompressor().decompress(&compressed),
        "the quick brown fox jumped over the lazy dog".as_bytes()
    );
}

#[test]
fn test_one_byte() {
    let empty = Compressor::train(&[0x01]);

    let compressed = empty.compress(&[0x01]);
    assert_eq!(compressed, vec![0u8]);

    assert_eq!(empty.decompressor().decompress(&compressed), vec![0x01]);
}

#[test]
fn test_zeros() {
    let training_data: Vec<u8> = vec![0, 1, 2, 3, 4, 0];
    let trained = Compressor::train(&training_data);
    let compressed = trained.compress(&[4, 0]);
    assert_eq!(trained.decompressor().decompress(&compressed), &[4, 0]);
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_large() {
    let corpus: Vec<u8> = DECLARATION.bytes().cycle().take(10_240).collect();

    let trained = Compressor::train(&corpus);
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
    let trained = Compressor::train(ART_OF_WAR.as_bytes());
    assert_eq!(
        ART_OF_WAR.as_bytes(),
        trained
            .decompressor()
            .decompress(&trained.compress(ART_OF_WAR.as_bytes()))
    );
}

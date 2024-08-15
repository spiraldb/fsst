#![cfg(test)]

use fsst_rs::Symbol;

static PREAMBLE: &str = r#"
When in the Course of human events, it becomes necessary for one people to dissolve
the political bands which have connected them with another, and to assume among the
powers of the earth, the separate and equal station to which the Laws of Nature and
of Nature's God entitle them, a decent respect to the opinions of mankind requires
that they should declare the causes which impel them to the separation."#;

static DECLARATION: &str = include_str!("./fixtures/declaration.txt");

#[test]
fn test_basic() {
    // Roundtrip the declaration
    let trained = fsst_rs::train(PREAMBLE);
    let compressed = trained.compress(PREAMBLE.as_bytes());
    let decompressed = trained.decompress(&compressed);
    assert_eq!(decompressed, PREAMBLE.as_bytes());
}

#[test]
fn test_train_on_empty() {
    let trained = fsst_rs::train("");
    // We can still compress with it, but the symbols are going to be empty.
    let compressed = trained.compress("the quick brown fox jumped over the lazy dog".as_bytes());
    assert_eq!(
        trained.decompress(&compressed),
        "the quick brown fox jumped over the lazy dog".as_bytes()
    );
}

#[test]
fn test_one_byte() {
    let mut empty = fsst_rs::SymbolTable::default();
    // Assign code 0 to map to the symbol containing byte 0x01
    empty.insert(Symbol::from_u8(0x01));

    let compressed = empty.compress(&[0x01]);
    assert_eq!(compressed, vec![0u8]);

    assert_eq!(empty.decompress(&compressed), vec![0x01]);
}

#[test]
fn test_zeros() {
    println!("training zeros");
    let training_data: Vec<u8> = vec![0, 1, 2, 3, 4];
    let trained = fsst_rs::train(&training_data);
    println!("compressing with zeros");
    let compressed = trained.compress(&[0, 4]);
    println!("decomperssing with zeros");
    assert_eq!(trained.decompress(&compressed), &[0, 4]);
    println!("done");
}

#[test]
fn test_large() {
    let mut corpus = String::new();
    // TODO(aduffy): make this larger once table build performance is better.
    while corpus.len() < 10 * 1_024 {
        corpus.push_str(DECLARATION);
    }

    let trained = fsst_rs::train(&corpus);
    let mut massive = String::new();
    while massive.len() < 16 * 1_024 * 1_024 {
        massive.push_str(DECLARATION);
    }
    let compressed = trained.compress(massive.as_bytes());
    assert_eq!(trained.decompress(&compressed), massive.as_bytes());
}

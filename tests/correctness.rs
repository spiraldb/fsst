#![cfg(test)]

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
fn test_zeros() {
    // make sure we don't panic if there are zeros in the training or input data
    let training_data: Vec<u8> = vec![0, 1, 2, 3, 4];
    let trained = fsst_rs::train(&training_data);
    let compressed = trained.compress(&[0, 4]);
    assert_eq!(trained.decompress(&compressed), &[0, 4]);
}

#[test]
fn test_large() {
    // Generate 100KB of test data
    let mut corpus = String::new();
    while corpus.len() < 8 * 1_024 * 1_024 {
        corpus.push_str(DECLARATION);
    }

    let trained = fsst_rs::train(&corpus);
    let compressed = trained.compress(corpus.as_bytes());
    assert_eq!(trained.decompress(&compressed), corpus.as_bytes());
}

//! End-to-end example using the FSST12 (12-bit code) variant.

use core::str;

use fsst::fsst12::Compressor12;

fn main() {
    let phrase = "the quick brown fox jumped over the lazy dog. ";
    let sample: String = phrase.repeat(32);

    let trained = Compressor12::train(&[sample.as_bytes()]);
    let compressed = trained.compress(sample.as_bytes());
    println!(
        "compressed: {} => {} bytes ({} learned symbols, {:.2}:1 ratio)",
        sample.len(),
        compressed.len(),
        trained.symbol_table().len() - 256,
        sample.len() as f64 / compressed.len() as f64,
    );

    let decoded = trained.decompressor().decompress(&compressed);
    let output = str::from_utf8(&decoded).unwrap();
    assert_eq!(output, sample);
    println!("decoded: len={} bytes round-tripped", decoded.len());
}

//! Simple example where we show round-tripping a string through the static symbol table.

use core::str;

use fsst::Compressor;

fn main() {
    // Train on a sample.
    let sample = "the quick brown fox jumped over the lazy dog";
    let trained = Compressor::train(&vec![sample.as_bytes()]);
    let compressed = trained.compress(sample.as_bytes());
    println!("compressed: {} => {}", sample.len(), compressed.len());
    // decompress now
    let decode = trained.decompressor().decompress(&compressed);
    let output = str::from_utf8(&decode).unwrap();
    println!(
        "decoded to the original: len={} text='{}'",
        decode.len(),
        output
    );
}

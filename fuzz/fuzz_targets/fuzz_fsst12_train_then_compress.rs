#![no_main]

// Trains on one byte sequence and compresses a separate `payload`. Unlike
// `fuzz_fsst12_compress`, the compress input may contain bytes or sequences the trainer
// never saw, exercising the identity-fallback and PHT-miss code paths.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (Vec<Vec<u8>>, Vec<u8>)| {
    let (train_corpus, payload) = input;
    let lines: Vec<&[u8]> = train_corpus.iter().map(|v| v.as_slice()).collect();
    let compressor = fsst::fsst12::Compressor12::train(&lines);
    let compressed = compressor.compress(&payload);
    let decompressed = compressor.decompressor().decompress(&compressed);
    assert_eq!(decompressed, payload);
});

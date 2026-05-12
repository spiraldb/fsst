#![no_main]

// Trains and compresses on the same buffer, so every byte of the input is in the training
// corpus. PHT-miss and unseen-byte paths are exercised by
// `fuzz_fsst12_train_then_compress`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let compressor = fsst::fsst12::Compressor12::train(&[data]);
    let compressed = compressor.compress(data);
    let decompressed = compressor.decompressor().decompress(&compressed);
    assert_eq!(&decompressed, data);
});

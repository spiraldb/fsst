#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let compressor =
        fsst::Compressor::train("the quick brown fox jumped over the lazy dog".as_bytes());
    let compress = compressor.compress(data);
    let decompress = compressor.decompressor().decompress(&compress);
    assert_eq!(&decompress, data);
});

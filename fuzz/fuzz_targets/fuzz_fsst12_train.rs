#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = fsst::fsst12::Compressor12::train(&[data]);
});

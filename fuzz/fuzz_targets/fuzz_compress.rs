#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let table = fsst_rs::train("the quick brown fox jumped over the lazy dog".as_bytes());
    let _ = table.compress(data);
});

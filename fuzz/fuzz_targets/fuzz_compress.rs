#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let compressor = fsst::Compressor::train(&vec![data]);
    let compressed = compressor.compress(data);
    let decompressed = compressor.decompressor().decompress(&compressed);
    assert_eq!(&decompressed, data);

    // Rebuild a compressor using the symbol table, and assert that it compresses and roundtrips
    // identically.
    let recompressor =
        fsst::Compressor::rebuild_from(compressor.symbol_table(), compressor.symbol_lengths());
    let recompressed = recompressor.compress(data);
    assert_eq!(
        &compressed,
        &recompressed,
        "failed comparison with data {:?} symbols: {:?}",
        data,
        compressor.symbol_table(),
    );
});

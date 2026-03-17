//! Test to demonstrate successful decompression on exact capacity bounds without panicking

#![cfg(test)]

use fsst::{CompressorBuilder, Symbol};

#[test]
fn test_decompress_exact_capacity() {
    // Train a compressor with a symbol that expands significantly
    let compressor = {
        let mut builder = CompressorBuilder::new();
        // Insert a highly compressible 8-byte symbol
        builder.insert(Symbol::from_slice(b"aaaaaaaa"), 8);
        builder.build()
    };

    // Create a large, highly compressible string
    let plaintext = vec![b'a'; 100_000];

    // Compress it into an over-allocated buffer to avoid any compression bugs
    let mut compressed = Vec::with_capacity(plaintext.len() * 2);
    unsafe {
        compressor.compress_into(&plaintext, &mut compressed);
    }

    let decompressor = compressor.decompressor();

    // If the caller allocates EXACTLY the uncompressed length (which is theoretically
    // sufficient and correct), `decompress_into` should successfully decode the entire
    // stream using the safe byte-by-byte fallback loop without early termination.
    let mut decompressed = Vec::with_capacity(plaintext.len());
    let spare = decompressed.spare_capacity_mut();

    // This previously panicked with `exhaust input before output`. It should now succeed.
    let len = decompressor.decompress_into(&compressed, spare);
    unsafe { decompressed.set_len(len) };

    assert_eq!(decompressed, plaintext);
}

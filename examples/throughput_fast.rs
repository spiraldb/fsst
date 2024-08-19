//! Simple throughput tester.
//!
//! Runs compression continuously over a configurable amount of auto-generated data and then
//! prints throughput stats.

use fsst::Compressor;

const DRACULA: &str = include_str!("../benches/dracula.txt");

fn main() {
    let bytes = std::env::args()
        .skip(1)
        .next()
        .unwrap_or_else(|| "1073741824".to_string());
    let parsed_bytes = usize::from_str_radix(&bytes, 10).unwrap();

    println!("building a simple symbol table");
    let compressor = Compressor::train(&DRACULA);
    for idx in 256..compressor.symbol_table().len() {
        let symbol = &compressor.symbol_table()[idx];
        println!(
            "symbol[{idx}] => '{:?}'",
            symbol
                .as_slice()
                .iter()
                .map(|c| *c as char)
                .map(|c| if c.is_ascii() {
                    format!("{c}")
                } else {
                    format!("{c:X?}")
                })
                .collect::<Vec<String>>()
        );
    }

    println!("building new text array of {parsed_bytes} bytes");

    let to_compress: Vec<u8> = DRACULA.bytes().cycle().take(parsed_bytes).collect();

    println!("beginning compression benchmark...");
    let start_time = std::time::Instant::now();
    let compressed = compressor.compress(&to_compress);
    let end_time = std::time::Instant::now();

    let duration = end_time.duration_since(start_time);

    println!("test completed");

    let ratio = (to_compress.len() as f64) / (compressed.len() as f64);

    println!("compression ratio: {ratio}");
    println!("wall time = {duration:?}");

    let bytes_per_sec = (parsed_bytes as f64) / duration.as_secs_f64();
    println!("tput: {bytes_per_sec} bytes/sec");

    // Measure decompression speed.
    println!("beginning decompression benchmark...");
    let start_time = std::time::Instant::now();
    let decompressed = compressor.decompressor().decompress(&compressed);
    let end_time = std::time::Instant::now();

    let duration = end_time.duration_since(start_time);

    println!("test completed");

    let ratio = (decompressed.len() as f64) / (compressed.len() as f64);

    println!("inflation ratio ratio: {ratio}");
    println!("wall time = {duration:?}");

    let bytes_per_sec = (compressed.len() as f64) / duration.as_secs_f64();
    println!("tput: {bytes_per_sec} bytes/sec");
}

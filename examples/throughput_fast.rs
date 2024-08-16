//! Simple throughput tester.
//!
//! Runs compression continuously over a configurable amount of auto-generated data and then
//! prints throughput stats.

use fsst::Compressor;

const ALPHA: &str = "abcdefghijklmnopqrstuvwxyz";

fn main() {
    let bytes = std::env::args()
        .skip(1)
        .next()
        .unwrap_or_else(|| "1073741824".to_string());
    let parsed_bytes = usize::from_str_radix(&bytes, 10).unwrap();

    println!("building a simple symbol table");
    let compressor = Compressor::train(&ALPHA);

    println!("building new text array of {parsed_bytes} bytes");
    let mut data = vec![0u8; parsed_bytes];
    for i in 0..parsed_bytes {
        // Return each byte individually as a char style thing.
        data[i] = ALPHA.chars().nth(i % ALPHA.len()).unwrap() as u8;
    }

    println!("beginning compression benchmark...");
    let start_time = std::time::Instant::now();
    let compressed = compressor.compress(&data);
    let end_time = std::time::Instant::now();

    let duration = end_time.duration_since(start_time);

    println!("test completed");

    let ratio = (parsed_bytes as f64) / (compressed.len() as f64);

    println!("compression ratio: {ratio}");
    println!("wall time = {duration:?}");

    let bytes_per_sec = (parsed_bytes as f64) / duration.as_secs_f64();
    println!("tput: {bytes_per_sec} bytes/sec");
}

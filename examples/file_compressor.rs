#![allow(missing_docs, clippy::use_debug)]

//! This is a command line program that expects an input file as an argument,
//! and trains a symbol table that it then uses to compress the file in-memory.
//!
//! Example:
//!
//! ```
//! cargo run --release --example file_compressor -- lineitem.tbl
//! ```
use std::{
    fs::File,
    io::Read,
    // io::{Read, Write},
    path::Path,
};

use fsst::Compressor;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();

    let input_path = Path::new(&args[0]);

    let mut string = String::new();
    {
        let mut f = File::open(input_path).unwrap();
        f.read_to_string(&mut string).unwrap();
    }
    let uncompressed_size = string.as_bytes().len();
    let lines: Vec<&[u8]> = string.lines().map(|line| line.as_bytes()).collect();

    // let mut output = File::create(output_path).unwrap();
    let start = std::time::Instant::now();
    let compressor = Compressor::train(&lines);
    let duration = std::time::Instant::now().duration_since(start);
    println!("train took {}µs", duration.as_micros());
    let mut compressed_size = 0;

    let mut buffer = Vec::with_capacity(8 * 1024 * 1024);

    let start = std::time::Instant::now();
    for text in lines {
        unsafe { compressor.compress_into(text, &mut buffer) };
        compressed_size += buffer.len();
    }
    let duration = std::time::Instant::now().duration_since(start);
    println!("compression took {}µs", duration.as_micros());
    println!(
        "compressed {} -> {} ({}%)",
        uncompressed_size,
        compressed_size,
        100.0 * (compressed_size as f64) / (uncompressed_size as f64)
    );
}

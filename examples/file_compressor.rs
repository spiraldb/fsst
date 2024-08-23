#![allow(missing_docs, clippy::use_debug)]

//! This is a command line program that expects two input files as arguments.
//!
//! The first is the file to train a symbol table on.
//!
//! The second is the file to compress. The compressor will run and compress
//! in chunks of 16MB, logging the compression ratio for each chunk.
//!
//! Example:
//!
//! ```
//! cargo run --release --example file_compressor -- file1.csv file2.csv
//! ```
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use fsst::Compressor;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(args.len() >= 2, "args TRAINING and FILE must be provided");

    let input_path = Path::new(&args[0]);
    let output_path = Path::new(&args[1]);

    let mut string = String::new();
    {
        let mut f = File::open(input_path).unwrap();
        f.read_to_string(&mut string).unwrap();
    }
    let uncompressed_size = string.as_bytes().len();
    let lines: Vec<&[u8]> = string.lines().map(|line| line.as_bytes()).collect();

    let mut output = File::create(output_path).unwrap();

    let compressor = Compressor::train(&lines);
    let mut compressed_size = 0;
    for text in lines {
        let compressed = compressor.compress(text);
        compressed_size += output.write(&compressed).unwrap();
    }

    println!(
        "compressed {} -> {} ({}%)",
        uncompressed_size,
        compressed_size,
        100.0 * (compressed_size as f64) / (uncompressed_size as f64)
    );
}

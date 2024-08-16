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
    io::Read,
    os::unix::fs::{FileExt, MetadataExt},
    path::Path,
};

use fsst::Compressor;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(args.len() >= 2, "args TRAINING and FILE must be provided");

    let train_path = Path::new(&args[0]);
    let input_path = Path::new(&args[1]);

    let mut train_bytes = Vec::new();
    {
        let mut f = File::open(train_path).unwrap();
        f.read_to_end(&mut train_bytes).unwrap();
    }

    println!("building the compressor from {train_path:?}...");
    let compressor = Compressor::train(&train_bytes);

    println!("compressing blocks of {input_path:?} with compressor...");

    let f = File::open(input_path).unwrap();
    let size_bytes = f.metadata().unwrap().size() as usize;

    const CHUNK_SIZE: usize = 16 * 1024 * 1024;

    let mut chunk_idx = 1;
    let mut pos = 0;
    let mut chunk = vec![0u8; CHUNK_SIZE];
    while pos + CHUNK_SIZE < size_bytes {
        f.read_exact_at(&mut chunk, pos as u64).unwrap();
        // Compress the chunk, don't write it anywhere.
        let compact = compressor.compress(&chunk);
        let compression_ratio = (CHUNK_SIZE as f64) / (compact.len() as f64);
        println!("compressed chunk {chunk_idx} with ratio {compression_ratio}");

        pos += CHUNK_SIZE;
        chunk_idx += 1;
    }

    // Read last chunk with a new custom-sized buffer.
    if pos < size_bytes {
        let amount = size_bytes - pos;
        chunk = vec![0u8; size_bytes - pos];
        f.read_exact_at(&mut chunk, pos as u64).unwrap();
        // Compress the chunk, don't write it anywhere.
        let compact = compressor.compress(&chunk[0..amount]);
        let compression_ratio = (amount as f64) / (compact.len() as f64);
        println!("compressed chunk {chunk_idx} with ratio {compression_ratio}");
    }
    println!("done");
}

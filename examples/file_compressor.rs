#![allow(missing_docs)]

//! This is a command line program that expects two input files as arguments.
//!
//! The first is the file to train a symbol table on.
//!
//! The second is the file to compress. The compressed file will be written
//! as a sibling with the suffix ".fsst"

use std::{
    fs::File,
    io::Read,
    os::unix::fs::{FileExt, MetadataExt},
    path::Path,
};

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(args.len() >= 2, "args TRAINING and FILE must be provided");

    let train_path = Path::new(&args[0]);
    let input_path = Path::new(&args[1]);

    let mut train_text = String::new();
    {
        let mut f = File::open(train_path).unwrap();
        f.read_to_string(&mut train_text).unwrap();
    }

    println!("building the compressor from {train_path:?}...");
    let compressor = fsst_rs::train(&train_text);

    println!("compressing blocks of {input_path:?} with compressor...");

    let f = File::open(input_path).unwrap();
    let size_bytes = f.metadata().unwrap().size() as usize;

    const CHUNK_SIZE: usize = 16 * 1024 * 1024;

    let mut chunk_idx = 1;
    let mut pos = 0;
    let mut chunk = Vec::with_capacity(CHUNK_SIZE);
    unsafe { chunk.set_len(CHUNK_SIZE) };
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
        chunk = Vec::with_capacity(size_bytes - pos);
        unsafe { chunk.set_len(amount) };
        f.read_exact_at(&mut chunk, pos as u64).unwrap();
        // Compress the chunk, don't write it anywhere.
        let compact = compressor.compress(&chunk[0..amount]);
        let compression_ratio = (amount as f64) / (compact.len() as f64);
        println!("compressed chunk {chunk_idx} with ratio {compression_ratio}");
    }
    println!("done");
}

//! docs

use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

use fsst::Compressor;

/// docs
fn main() -> io::Result<()> {
    // Step 1: Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <infile> <outfile>", args[0]);
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Expected two arguments: infile and outfile",
        ));
    }

    let infile = &args[1];
    let outfile = &args[2];

    // Step 2: Read the infile into memory as a Vec<&[u8]> with one slice per line
    let input_file = File::open(infile)?;
    let reader = BufReader::new(input_file);
    let mut lines: Vec<Vec<u8>> = Vec::new(); // We'll store Vec<u8> and convert to &[u8] later

    for line in reader.lines() {
        let line = line?; // Handle potential IO errors
        lines.push(line.into_bytes()); // Convert each line into a Vec<u8>
    }

    // Convert Vec<Vec<u8>> to Vec<&[u8]> (which is required for Compressor::train)
    let line_slices: Vec<&[u8]> = lines.iter().map(|line| line.as_slice()).collect();

    // Step 3: Train the compressor on the data
    let compressor = Compressor::train(&line_slices);

    // Step 4: Compress the data using compress_bulk
    let compressed_data = compressor.compress_bulk(&line_slices);

    // Step 5: Write the compressed data to the outfile
    let mut output_file = File::create(outfile)?;
    for x in compressed_data {
        output_file.write_all(&x)?;
    }

    println!("Compression complete and written to {}", outfile);

    Ok(())
}

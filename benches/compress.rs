//! Benchmarks for FSST compression, decompression, and symbol table training.
//!
//! We use the dbtext data at https://github.com/cwida/fsst/tree/master/paper/dbtext
#![allow(missing_docs)]
use core::str;
use std::{
    error::Error,
    fs::{self, DirBuilder, File},
    io::{Read, Write},
    path::Path,
};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use curl::easy::Easy;
use fsst::Compressor;

fn download_dataset(url: &str, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
    let target = path.as_ref();

    let mut dir_builder = DirBuilder::new();
    dir_builder.recursive(true);

    dir_builder.create(target.parent().unwrap())?;

    // Avoid downloading the file twice.
    if target.exists() {
        return Ok(());
    }

    let mut handle = Easy::new();

    let mut buffer = Vec::new();
    handle.url(url)?;
    {
        let mut transfer = handle.transfer();
        transfer.write_function(|data| {
            buffer.extend_from_slice(data);

            Ok(data.len())
        })?;
        transfer.perform()?;
    }

    let mut output = File::create(target)?;
    match output.write_all(&buffer) {
        Ok(()) => {}
        Err(err) => {
            // cleanup in case of failure
            fs::remove_file(target).unwrap();

            return Err(Box::new(err));
        }
    }

    Ok(())
}

fn bench_dbtext(c: &mut Criterion) {
    fn run_dataset_bench(name: &str, url: &str, path: &str, c: &mut Criterion) {
        let mut group = c.benchmark_group(name);
        download_dataset(url, path).unwrap();

        let mut text = String::new();
        let lines: Vec<&[u8]> = {
            let mut file = File::open(path).unwrap();
            file.read_to_string(&mut text).unwrap();

            text.lines().map(|line| line.as_bytes()).collect()
        };

        group.bench_function("train-and-compress", |b| {
            b.iter(|| {
                let compressor = Compressor::train(&lines);
                let _ =
                    std::hint::black_box(compressor.compress_bulk(std::hint::black_box(&lines)));
            });
        });

        let compressor = Compressor::train(&lines);
        group.throughput(Throughput::Bytes(
            lines.iter().map(|l| l.len() as u64).sum::<u64>(),
        ));
        group.bench_function("compress-only", |b| {
            b.iter(|| {
                let _ =
                    std::hint::black_box(compressor.compress_bulk(std::hint::black_box(&lines)));
            });
        });

        group.finish();

        // Report the compression factor for this dataset.
        let uncompressed_size = lines.iter().map(|l| l.len()).sum::<usize>();
        let compressor = Compressor::train(&lines);

        // Show the symbols
        for code in 256..compressor.symbol_table().len() {
            let symbol = compressor.symbol_table()[code];
            let code = code - 256;
            println!("symbol[{code}] = {symbol:?}");
        }

        let compressed = compressor.compress_bulk(&lines);
        let compressed_size = compressed.iter().map(|l| l.len()).sum::<usize>();
        let ratio = 100.0 * (compressed_size as f64) / (uncompressed_size as f64);
        println!(
            "compressed {name} {uncompressed_size} => {compressed_size}B ({ratio}% of original)"
        )
    }

    run_dataset_bench(
        "dbtext/wikipedia",
        "https://raw.githubusercontent.com/cwida/fsst/4e188a/paper/dbtext/wikipedia",
        "benches/data/wikipedia",
        c,
    );

    run_dataset_bench(
        "dbtext/l_comment",
        "https://raw.githubusercontent.com/cwida/fsst/4e188a/paper/dbtext/l_comment",
        "benches/data/l_comment",
        c,
    );

    run_dataset_bench(
        "dbtext/urls",
        "https://raw.githubusercontent.com/cwida/fsst/4e188a/paper/dbtext/urls",
        "benches/data/urls",
        c,
    );

    run_dataset_bench(
        "dbtext/urls",
        "https://raw.githubusercontent.com/cwida/fsst/4e188a/paper/dbtext/urls",
        "benches/data/urls",
        c,
    );
}

criterion_group!(compress_bench, bench_dbtext);
criterion_main!(compress_bench);

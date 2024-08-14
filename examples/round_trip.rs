// Copyright 2024 Spiral, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Simple example where we show round-tripping a string through the static symbol table.

use core::str;

fn main() {
    // Train on a sample.
    let sample = "the quick brown fox jumped over the lazy dog";
    let trained = fsst_rs::train(sample.as_bytes());
    let compressed = trained.compress(sample.as_bytes());
    println!("compressed: {} => {}", sample.len(), compressed.len());
    // decompress now
    let decode = trained.decompress(&compressed);
    let output = str::from_utf8(&decode).unwrap();
    println!(
        "decoded to the original: len={} text='{}'",
        decode.len(),
        output
    );
}

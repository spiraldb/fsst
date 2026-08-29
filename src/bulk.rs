//! Bulk compression that interleaves several independent cursors.
//!
//! A single compression cursor is latency-bound: the number of bytes each iteration consumes
//! comes out of that iteration's symbol-table lookup, so the next lookup cannot start until the
//! previous one lands. Values are compressed independently of one another, so running several
//! cursors over disjoint ranges of values keeps several of those chains in flight at once.

use std::mem::MaybeUninit;
use std::ptr;

use crate::Compressor;

/// Number of independent cursors [`Compressor::compress_bulk_into`] interleaves.
///
/// Throughput climbs steeply from one to four cursors and then falls back as the working set of
/// each iteration outgrows the register file.
const LANES: usize = 4;

/// Largest lane count the fixed-size per-lane arrays below can hold.
///
/// Throughput already falls off well before this, so the ceiling only has to be high enough to
/// measure the falloff.
const MAX_LANES: usize = 16;

/// State of a single cursor.
struct Lane {
    /// Next input byte to consume.
    in_ptr: *const u8,
    /// End of the value currently being compressed.
    in_end: *const u8,
    /// Next output byte to write.
    out_ptr: *mut u8,
    /// Index of the value currently being compressed.
    value: usize,
    /// One past the last value index owned by this lane.
    value_end: usize,
}

impl Compressor {
    /// Compress many values into a single contiguous buffer.
    ///
    /// The compressed bytes of every value are appended to `output`, and the end offset of every
    /// value is appended to `offsets`. Offsets are absolute indices into `output`, so with `k`
    /// the length of `offsets` before the call and `start` the length of `output` before the
    /// call, `values[i]` occupies `output[s..offsets[k + i]]`, where `s` is `start` for `i == 0`
    /// and `offsets[k + i - 1]` otherwise.
    ///
    /// Each value is compressed independently, exactly as [`Self::compress`] would compress it,
    /// so any one of them can be decompressed on its own from that range.
    ///
    /// ```
    /// use fsst::{Compressor, CompressorBuilder, Symbol};
    ///
    /// let compressor = {
    ///     let mut builder = CompressorBuilder::new();
    ///     builder.insert(Symbol::from_slice(&[b'h', b'e', b'l', b'l', b'o', 0, 0, 0]), 5);
    ///     builder.build()
    /// };
    ///
    /// let values: &[&[u8]] = &[b"hello", b"hellohello"];
    /// let mut output = Vec::new();
    /// let mut offsets = Vec::new();
    /// compressor.compress_bulk_into(values, &mut output, &mut offsets);
    ///
    /// // "hello" is a single code, so the first value is one byte and the second is two.
    /// assert_eq!(offsets, vec![1, 3]);
    ///
    /// let decompressor = compressor.decompressor();
    /// assert_eq!(decompressor.decompress(&output[..1]), b"hello");
    /// assert_eq!(decompressor.decompress(&output[1..3]), b"hellohello");
    /// ```
    pub fn compress_bulk_into(
        &self,
        values: &[&[u8]],
        output: &mut Vec<u8>,
        offsets: &mut Vec<u64>,
    ) {
        self.compress_bulk_lanes::<LANES>(values, output, offsets)
    }

    /// [`Self::compress_bulk_into`] with the cursor count exposed, so the best value for a given
    /// machine can be measured rather than assumed.
    ///
    /// # Panics
    ///
    /// Panics unless `K` is in `1..=16`.
    #[doc(hidden)]
    pub fn compress_bulk_lanes<const K: usize>(
        &self,
        values: &[&[u8]],
        output: &mut Vec<u8>,
        offsets: &mut Vec<u64>,
    ) {
        assert!(
            K > 0 && K <= MAX_LANES,
            "lane count must be in 1..={MAX_LANES}"
        );
        if values.is_empty() {
            return;
        }

        let total_in: usize = values.iter().map(|value| value.len()).sum();
        let base = output.len();
        offsets.reserve(values.len());

        // Every value compresses to at most two bytes per input byte, so this is enough room for
        // all of them, and enough for each lane to be given a disjoint slice of it up front.
        output.reserve(2 * total_in);
        let spare = output.spare_capacity_mut();
        let spare_ptr: *mut u8 = spare.as_mut_ptr().cast();

        if values.len() < K {
            // SAFETY: `output` was just reserved to two bytes per byte of input.
            let written = unsafe { self.compress_bulk_serial(values, spare_ptr, base, offsets) };
            // SAFETY: bytes `base..base + written` of the allocation were initialized above.
            unsafe { output.set_len(base + written) };
            return;
        }

        // Split the values into K contiguous ranges. Splitting by index rather than by byte
        // count keeps every lane non-empty, and values within one array are usually of similar
        // length, so the ranges come out close to balanced anyway.
        let mut bounds = [0usize; MAX_LANES + 1];
        for (lane, bound) in bounds.iter_mut().enumerate().take(K + 1) {
            *bound = lane * values.len() / K;
        }

        // Give each lane a disjoint slice of the output, sized to the worst case for its own
        // values, so lanes never contend and no bounds check is needed in the interleaved loop.
        let mut lane_out_base = [0usize; MAX_LANES + 1];
        let mut cursor = 0usize;
        for lane in 0..K {
            lane_out_base[lane] = cursor;
            let bytes: usize = values[bounds[lane]..bounds[lane + 1]]
                .iter()
                .map(|value| value.len())
                .sum();
            cursor += 2 * bytes;
        }
        lane_out_base[K] = cursor;

        // End offset of every value, relative to `spare_ptr`, before the lanes are compacted.
        let mut ends = vec![0u64; values.len()];

        let mut lanes: [Lane; MAX_LANES] = std::array::from_fn(|_| Lane {
            in_ptr: ptr::null(),
            in_end: ptr::null(),
            out_ptr: ptr::null_mut(),
            value: 0,
            value_end: 0,
        });
        for lane in 0..K {
            let value = bounds[lane];
            let bytes = values[value];
            lanes[lane] = Lane {
                in_ptr: bytes.as_ptr(),
                // SAFETY: one past the end of the value's own allocation.
                in_end: unsafe { bytes.as_ptr().add(bytes.len()) },
                // SAFETY: `lane_out_base[lane]` is within the reserved spare capacity.
                out_ptr: unsafe { spare_ptr.add(lane_out_base[lane]) },
                value,
                value_end: bounds[lane + 1],
            };
        }

        'interleaved: loop {
            // Bring every lane up to at least a full word of input, finishing values and moving
            // on to the next one as needed. This is the only branch in the loop that depends on
            // the data, and it is taken about once per value rather than once per iteration.
            for lane in 0..K {
                while (lanes[lane].in_ptr as usize) + 8 > lanes[lane].in_end as usize {
                    // SAFETY: the lane's pointers are within its own value and output slice.
                    unsafe { self.finish_value(&mut lanes[lane], values, spare_ptr, &mut ends) };
                    lanes[lane].value += 1;
                    if lanes[lane].value == lanes[lane].value_end {
                        break 'interleaved;
                    }
                    let bytes = values[lanes[lane].value];
                    lanes[lane].in_ptr = bytes.as_ptr();
                    // SAFETY: one past the end of the value's own allocation.
                    lanes[lane].in_end = unsafe { bytes.as_ptr().add(bytes.len()) };
                }
            }

            // One step of every cursor. These are independent, so their symbol-table lookups
            // overlap instead of serializing.
            for lane in &mut lanes[..K] {
                // SAFETY: the check above leaves at least 8 readable bytes in the current value,
                // and the lane's output slice has room for two bytes per input byte consumed.
                unsafe {
                    let word = ptr::read_unaligned(lane.in_ptr as *const u64);
                    let (advance_in, advance_out) = self.compress_word(word, lane.out_ptr);
                    lane.in_ptr = lane.in_ptr.add(advance_in);
                    lane.out_ptr = lane.out_ptr.add(advance_out);
                }
            }
        }

        // Drain whatever is left in each lane once the first one has run out of values.
        for lane in 0..K {
            while lanes[lane].value < lanes[lane].value_end {
                // SAFETY: as above.
                unsafe { self.finish_value(&mut lanes[lane], values, spare_ptr, &mut ends) };
                lanes[lane].value += 1;
                if lanes[lane].value < lanes[lane].value_end {
                    let bytes = values[lanes[lane].value];
                    lanes[lane].in_ptr = bytes.as_ptr();
                    // SAFETY: one past the end of the value's own allocation.
                    lanes[lane].in_end = unsafe { bytes.as_ptr().add(bytes.len()) };
                }
            }
        }

        // The lanes wrote into disjoint slices sized for the worst case, so the compressed bytes
        // are separated by gaps. Slide each lane down onto the end of the previous one, and shift
        // its offsets by the same amount.
        let mut written = 0usize;
        for lane in 0..K {
            let lane_start = lane_out_base[lane];
            let lane_len = if bounds[lane] == bounds[lane + 1] {
                0
            } else {
                ends[bounds[lane + 1] - 1] as usize - lane_start
            };
            if lane_start != written {
                // SAFETY: both ranges are inside the reserved spare capacity, and the
                // destination starts at or before the source, so the copy is well-formed even
                // when the ranges overlap.
                unsafe { ptr::copy(spare_ptr.add(lane_start), spare_ptr.add(written), lane_len) };
            }
            let shift = (lane_start - written) as u64;
            for end in &ends[bounds[lane]..bounds[lane + 1]] {
                offsets.push(end - shift + base as u64);
            }
            written += lane_len;
        }

        // SAFETY: bytes `base..base + written` of the allocation were initialized above.
        unsafe { output.set_len(base + written) };
    }

    /// Compress the unconsumed remainder of the lane's current value and record its end offset.
    ///
    /// # Safety
    ///
    /// The lane's pointers must lie within its current value and its own output slice.
    unsafe fn finish_value(
        &self,
        lane: &mut Lane,
        values: &[&[u8]],
        spare_ptr: *mut u8,
        ends: &mut [u64],
    ) {
        let bytes = values[lane.value];
        // SAFETY: `in_ptr` is within the value, by the caller's contract.
        let consumed = unsafe { lane.in_ptr.offset_from(bytes.as_ptr()) } as usize;
        let rest = &bytes[consumed..];

        if !rest.is_empty() {
            // The lane's slice always holds two bytes per byte of input it has left, so a
            // worst-case all-escape remainder still fits.
            let capacity = 2 * rest.len();
            // SAFETY: `out_ptr` is inside the lane's slice, which has at least `capacity` bytes
            // left, and uninitialized `u8` is a valid `MaybeUninit<u8>`.
            let out = unsafe {
                std::slice::from_raw_parts_mut(lane.out_ptr.cast::<MaybeUninit<u8>>(), capacity)
            };
            // SAFETY: `out` is large enough for the worst case, as argued above.
            let written = unsafe { self.compress_into(rest, out) };
            // SAFETY: `written` bytes were just written into the lane's slice.
            lane.out_ptr = unsafe { lane.out_ptr.add(written) };
        }

        // SAFETY: `out_ptr` is derived from `spare_ptr`.
        ends[lane.value] = unsafe { lane.out_ptr.offset_from(spare_ptr) } as u64;
    }

    /// Compress every value one after another, with no interleaving.
    ///
    /// Used when there are too few values to give every cursor one.
    ///
    /// # Safety
    ///
    /// `spare_ptr` must have room for two bytes per byte of input.
    unsafe fn compress_bulk_serial(
        &self,
        values: &[&[u8]],
        spare_ptr: *mut u8,
        base: usize,
        offsets: &mut Vec<u64>,
    ) -> usize {
        let mut written = 0usize;
        for value in values.iter() {
            if !value.is_empty() {
                let capacity = 2 * value.len();
                // SAFETY: the caller reserved two bytes per input byte for every value.
                let out = unsafe {
                    std::slice::from_raw_parts_mut(
                        spare_ptr.add(written).cast::<MaybeUninit<u8>>(),
                        capacity,
                    )
                };
                // SAFETY: `out` is large enough for the worst case.
                written += unsafe { self.compress_into(value, out) };
            }
            offsets.push((base + written) as u64);
        }
        written
    }
}

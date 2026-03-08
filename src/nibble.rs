// //! Nibble-oriented I/O for the LZR bitstream.
// //!
// //! The LZR format packs data as 4-bit nibbles. Nibble 2K occupies bits 7–4 of
// //! byte K; nibble 2K+1 occupies bits 3–0. Literal bytes are stored low nibble
// //! first, high nibble second.

// use std::io::{self, Read, Write};

// /// Growable nibble writer backed by a `Vec<u8>`.
// ///
// /// Nibbles are packed high-first: the first nibble written occupies bits 7–4
// /// of byte 0, the second occupies bits 3–0, and so on.
// #[derive(Debug, Clone)]
// pub(crate) struct NibbleWriter {
//     bytes: Vec<u8>,
//     /// `true` when the next nibble goes into bits 7–4 of a **new** byte.
//     high: bool,
// }

// impl Default for NibbleWriter {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl NibbleWriter {
//     /// Create an empty writer.
//     pub(crate) const fn new() -> Self {
//         Self {
//             bytes: Vec::new(),
//             high: true,
//         }
//     }

//     /// Create an empty writer with pre-allocated byte capacity.
//     pub(crate) fn with_capacity(byte_cap: usize) -> Self {
//         Self {
//             bytes: Vec::with_capacity(byte_cap),
//             high: true,
//         }
//     }

//     /// Push one nibble (only the lower 4 bits of `nibble` are used).
//     #[inline]
//     pub(crate) fn write(&mut self, nibble: u8) {
//         let n = nibble & 0xF;
//         if self.high {
//             self.bytes.push(n << 4);
//         } else {
//             *self.bytes.last_mut().expect("low nibble implies non-empty buffer") |= n;
//         }
//         self.high = !self.high;
//     }

//     /// Push a literal byte as two nibbles: low nibble first, then high nibble.
//     #[inline]
//     pub(crate) fn write_byte(&mut self, byte: u8) {
//         if self.high {
//             // Fast path: aligned — single push, no branches, high stays true.
//             self.bytes.push(byte.rotate_right(4));
//         } else {
//             self.write(byte & 0xF);
//             self.write(byte >> 4);
//         }
//     }

//     /// Push an entire slice of literal bytes.
//     ///
//     /// When aligned (high == true), this avoids per-nibble overhead entirely.
//     #[inline]
//     pub(crate) fn write_bytes(&mut self, data: &[u8]) {
//         if self.high {
//             self.bytes.reserve(data.len());
//             for &byte in data {
//                 self.bytes.push(byte.rotate_right(4));
//             }
//         } else {
//             for &byte in data {
//                 self.write_byte(byte);
//             }
//         }
//     }

//     /// If the nibble count is odd, push a zero nibble to reach byte alignment.
//     pub(crate) fn pad(&mut self) {
//         if !self.high {
//             self.write(0);
//         }
//     }

//     /// Consume the writer, auto-padding if necessary, and return the packed bytes.
//     pub(crate) fn into_bytes(mut self) -> Vec<u8> {
//         self.pad();
//         self.bytes
//     }

//     /// Flush completed bytes to `output`, retaining any partial (odd-nibble) byte.
//     #[allow(dead_code)]
//     ///
//     /// - Even nibble count (`high`): write all bytes, clear vec.
//     /// - Odd nibble count (`!high`): write all but last byte, move last byte to position 0, truncate to 1.
//     pub(crate) fn drain_bytes(&mut self, output: &mut impl Write) -> io::Result<()> {
//         if self.high {
//             // Even nibble count — all bytes are complete.
//             output.write_all(&self.bytes)?;
//             self.bytes.clear();
//         } else {
//             // Odd nibble count — last byte is partial.
//             let complete = self.bytes.len() - 1;
//             if complete > 0 {
//                 output.write_all(&self.bytes[..complete])?;
//                 self.bytes[0] = self.bytes[complete];
//                 self.bytes.truncate(1);
//             }
//         }
//         Ok(())
//     }

//     /// Number of nibbles written so far.
//     #[allow(dead_code)]
//     pub(crate) const fn len(&self) -> usize {
//         if self.high {
//             self.bytes.len() * 2
//         } else {
//             self.bytes.len() * 2 - 1
//         }
//     }

//     /// Returns `true` if no nibbles have been written.
//     #[allow(dead_code)]
//     pub(crate) const fn is_empty(&self) -> bool {
//         self.bytes.is_empty()
//     }
// }

// /// Nibble reader over a generic byte source.
// ///
// /// Returns [`io::ErrorKind::UnexpectedEof`] when reading past the end.
// #[derive(Debug)]
// pub(crate) struct NibbleReader<R> {
//     reader: R,
//     /// Low nibble of the last byte read, waiting to be consumed.
//     pending: Option<u8>,
//     /// Nibble count (for diagnostics).
//     pos: usize,
// }

// impl<R: Read> NibbleReader<R> {
//     /// Create a reader starting at nibble 0.
//     pub(crate) const fn new(reader: R) -> Self {
//         Self {
//             reader,
//             pending: None,
//             pos: 0,
//         }
//     }

//     /// Read the next nibble.
//     #[inline]
//     pub(crate) fn read(&mut self) -> io::Result<u8> {
//         let nibble = if let Some(low) = self.pending.take() {
//             low
//         } else {
//             let mut buf = [0u8; 1];
//             self.reader.read_exact(&mut buf)?;
//             let byte = buf[0];
//             self.pending = Some(byte & 0xF);
//             byte >> 4
//         };
//         self.pos += 1;
//         Ok(nibble)
//     }

//     /// Read a literal byte: low nibble first, high nibble second.
//     #[inline]
//     pub(crate) fn read_byte(&mut self) -> io::Result<u8> {
//         if self.pending.is_none() {
//             // Fast path: aligned — single read, no Option checks.
//             let mut buf = [0u8; 1];
//             self.reader.read_exact(&mut buf)?;
//             self.pos += 2;
//             Ok(buf[0].rotate_right(4))
//         } else {
//             let lo = self.read()?;
//             let hi = self.read()?;
//             Ok((hi << 4) | lo)
//         }
//     }

//     /// Read all remaining bytes at once.
//     ///
//     /// When aligned (no pending nibble), reads the entire remaining stream
//     /// and nibble-swaps in place, avoiding per-byte overhead.
//     pub(crate) fn read_all_bytes(&mut self) -> io::Result<Vec<u8>> {
//         if self.pending.is_some() {
//             let mut result = Vec::new();
//             while let Ok(byte) = self.read_byte() {
//                 result.push(byte);
//             }
//             return Ok(result);
//         }
//         let mut raw = Vec::new();
//         self.reader.read_to_end(&mut raw)?;
//         self.pos += raw.len() * 2;
//         for byte in &mut raw {
//             *byte = byte.rotate_right(4);
//         }
//         Ok(raw)
//     }

//     /// If there is a pending nibble, discard it.
//     #[allow(dead_code)]
//     pub(crate) const fn skip_padding(&mut self) {
//         if self.pending.is_some() {
//             self.pending = None;
//             self.pos += 1;
//         }
//     }

//     /// Current nibble index.
//     #[allow(dead_code)]
//     pub(crate) const fn position(&self) -> usize {
//         self.pos
//     }

//     /// Consume the reader, returning the inner source.
//     ///
//     /// Debug-asserts that no pending nibble remains.
//     #[allow(dead_code)]
//     pub(crate) fn into_inner(self) -> R {
//         debug_assert!(self.pending.is_none(), "into_inner called with pending nibble");
//         self.reader
//     }
// }

// #[cfg(test)]
// mod test {
//     use pretty_assertions::assert_eq;
//     use proptest::prelude::*;

//     use super::*;

//     #[test]
//     fn single_nibble_round_trip() {
//         let mut w = NibbleWriter::new();
//         w.write(0x5);
//         let bytes = w.into_bytes();
//         let mut r = NibbleReader::new(&bytes[..]);
//         assert_eq!(r.read().unwrap(), 0x5);
//     }

//     #[test]
//     fn nibble_packing_odd() {
//         let mut w = NibbleWriter::new();
//         for n in [0xA, 0xB, 0xC] {
//             w.write(n);
//         }
//         assert_eq!(w.len(), 3);
//         let bytes = w.into_bytes();
//         assert_eq!(bytes, vec![0xAB, 0xC0]);
//     }

//     #[test]
//     fn nibble_packing_even() {
//         let mut w = NibbleWriter::new();
//         w.write(0xA);
//         w.write(0xB);
//         assert_eq!(w.len(), 2);
//         let bytes = w.into_bytes();
//         assert_eq!(bytes, vec![0xAB]);
//     }

//     #[test]
//     fn write_byte_layout() {
//         // write_byte(0x48) → low nibble 0x8 first, high nibble 0x4 second → [0x84]
//         let mut w = NibbleWriter::new();
//         w.write_byte(0x48);
//         let bytes = w.into_bytes();
//         assert_eq!(bytes, vec![0x84]);
//     }

//     #[test]
//     fn read_byte_from_packed() {
//         // [0x84] → low nibble 8 (high nib of byte), high nibble 4 (low nib of byte) → 0x48
//         let mut r = NibbleReader::new(&[0x84][..]);
//         assert_eq!(r.read_byte().unwrap(), 0x48);
//     }

//     #[test]
//     fn hi_literal_sequence() {
//         // "Hi" = 0x48, 0x69
//         // write_byte(0x48) → nibbles [8, 4], write_byte(0x69) → nibbles [9, 6]
//         // packed: [0x84, 0x96]
//         let mut w = NibbleWriter::new();
//         w.write_byte(0x48);
//         w.write_byte(0x69);
//         let bytes = w.into_bytes();
//         assert_eq!(bytes, vec![0x84, 0x96]);

//         let mut r = NibbleReader::new(&bytes[..]);
//         assert_eq!(r.read_byte().unwrap(), 0x48);
//         assert_eq!(r.read_byte().unwrap(), 0x69);
//     }

//     #[test]
//     fn empty_writer() {
//         let w = NibbleWriter::new();
//         assert!(w.is_empty());
//         assert_eq!(w.len(), 0);
//         assert_eq!(w.into_bytes(), vec![]);
//     }

//     #[test]
//     fn reader_empty_slice_eof() {
//         let mut r = NibbleReader::new(&[][..]);
//         let err = r.read().unwrap_err();
//         assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
//     }

//     #[test]
//     fn pad_even_is_noop() {
//         let mut w = NibbleWriter::new();
//         w.write(0xA);
//         w.write(0xB);
//         let len_before = w.len();
//         w.pad();
//         assert_eq!(w.len(), len_before);
//         assert_eq!(w.into_bytes(), vec![0xAB]);
//     }

//     #[test]
//     fn pad_odd_adds_zero() {
//         let mut w = NibbleWriter::new();
//         w.write(0xA);
//         assert_eq!(w.len(), 1);
//         w.pad();
//         assert_eq!(w.len(), 2);
//         assert_eq!(w.into_bytes(), vec![0xA0]);
//     }

//     #[test]
//     fn into_inner_returns_remaining() {
//         let data: &[u8] = &[0xAB, 0xCD];
//         let mut r = NibbleReader::new(data);
//         // Read all 4 nibbles (2 bytes worth)
//         for _ in 0..4 {
//             r.read().unwrap();
//         }
//         let inner = r.into_inner();
//         // Inner reader should be exhausted
//         assert!(inner.is_empty());
//     }

//     #[test]
//     fn into_inner_after_full_byte_reads() {
//         let data: &[u8] = &[0x84, 0x96];
//         let mut r = NibbleReader::new(data);
//         r.read_byte().unwrap();
//         r.read_byte().unwrap();
//         let inner = r.into_inner();
//         assert!(inner.is_empty());
//     }

//     #[test]
//     fn skip_padding_even_noop() {
//         let mut r = NibbleReader::new(&[0xAB][..]);
//         assert_eq!(r.position(), 0);
//         r.skip_padding();
//         assert_eq!(r.position(), 0);
//     }

//     #[test]
//     fn skip_padding_odd_advances() {
//         let mut r = NibbleReader::new(&[0xAB, 0xCD][..]);
//         r.read().unwrap(); // pos = 1
//         r.skip_padding();
//         assert_eq!(r.position(), 2);
//         assert_eq!(r.read().unwrap(), 0xC);
//     }

//     #[test]
//     fn with_capacity_works() {
//         let mut w = NibbleWriter::with_capacity(16);
//         w.write(0xF);
//         assert_eq!(w.len(), 1);
//     }

//     #[test]
//     fn drain_bytes_even() {
//         let mut w = NibbleWriter::new();
//         w.write_byte(0x48);
//         w.write_byte(0x69);
//         let mut out = Vec::new();
//         w.drain_bytes(&mut out).unwrap();
//         assert_eq!(out, vec![0x84, 0x96]);
//         assert!(w.is_empty());
//     }

//     #[test]
//     fn drain_bytes_odd() {
//         let mut w = NibbleWriter::new();
//         w.write(0xA);
//         w.write(0xB);
//         w.write(0xC); // 3 nibbles → 1 complete byte + 1 partial
//         let mut out = Vec::new();
//         w.drain_bytes(&mut out).unwrap();
//         assert_eq!(out, vec![0xAB]);
//         assert_eq!(w.len(), 1); // partial byte retained
//     }

//     #[test]
//     fn drain_bytes_empty() {
//         let mut w = NibbleWriter::new();
//         let mut out = Vec::new();
//         w.drain_bytes(&mut out).unwrap();
//         assert!(out.is_empty());
//         assert!(w.is_empty());
//     }

//     #[test]
//     fn drain_bytes_single_nibble() {
//         let mut w = NibbleWriter::new();
//         w.write(0xA); // 1 nibble → only a partial byte, nothing to drain
//         let mut out = Vec::new();
//         w.drain_bytes(&mut out).unwrap();
//         assert!(out.is_empty());
//         assert_eq!(w.len(), 1);
//     }

//     #[test]
//     fn write_bytes_aligned() {
//         let data = b"Hello, world!";
//         let mut w1 = NibbleWriter::new();
//         w1.write_bytes(data);
//         let mut w2 = NibbleWriter::new();
//         for &b in data.as_slice() {
//             w2.write_byte(b);
//         }
//         assert_eq!(w1.into_bytes(), w2.into_bytes());
//     }

//     #[test]
//     fn write_bytes_unaligned() {
//         let data = b"Hi";
//         // Start unaligned by writing a single nibble first.
//         let mut w1 = NibbleWriter::new();
//         w1.write(0xA);
//         w1.write_bytes(data);
//         let mut w2 = NibbleWriter::new();
//         w2.write(0xA);
//         for &b in data.as_slice() {
//             w2.write_byte(b);
//         }
//         assert_eq!(w1.into_bytes(), w2.into_bytes());
//     }

//     #[test]
//     fn read_all_bytes_aligned() {
//         let mut w = NibbleWriter::new();
//         let original = b"Hello!";
//         w.write_bytes(original);
//         let packed = w.into_bytes();
//         let mut r = NibbleReader::new(&packed[..]);
//         let decoded = r.read_all_bytes().unwrap();
//         assert_eq!(decoded, original);
//     }

//     #[test]
//     fn read_all_bytes_unaligned() {
//         // Write a nibble prefix + 2 bytes, then read: first nibble via read(), rest via read_all_bytes().
//         let mut w = NibbleWriter::new();
//         w.write(0xA);
//         w.write_byte(0x48);
//         w.write_byte(0x69);
//         let packed = w.into_bytes();

//         let mut r = NibbleReader::new(&packed[..]);
//         // Consume one nibble to put reader in unaligned state.
//         assert_eq!(r.read().unwrap(), 0xA);
//         let decoded = r.read_all_bytes().unwrap();
//         assert_eq!(decoded, vec![0x48, 0x69]);
//     }

//     proptest! {
//         #[test]
//         fn round_trip_nibbles(nibbles in prop::collection::vec(0u8..=15, 0..256)) {
//             let mut w = NibbleWriter::new();
//             for &n in &nibbles {
//                 w.write(n);
//             }
//             let bytes = w.into_bytes();
//             let mut r = NibbleReader::new(&bytes[..]);
//             for &expected in &nibbles {
//                 prop_assert_eq!(r.read().unwrap(), expected);
//             }
//         }

//         #[test]
//         fn round_trip_bytes(values in prop::collection::vec(any::<u8>(), 0..128)) {
//             let mut w = NibbleWriter::new();
//             for &b in &values {
//                 w.write_byte(b);
//             }
//             let bytes = w.into_bytes();
//             let mut r = NibbleReader::new(&bytes[..]);
//             for &expected in &values {
//                 prop_assert_eq!(r.read_byte().unwrap(), expected);
//             }
//         }

//         #[test]
//         fn round_trip_bulk(values in prop::collection::vec(any::<u8>(), 0..256)) {
//             let mut w = NibbleWriter::new();
//             w.write_bytes(&values);
//             let packed = w.into_bytes();
//             let mut r = NibbleReader::new(&packed[..]);
//             let decoded = r.read_all_bytes().unwrap();
//             prop_assert_eq!(decoded, values);
//         }

//         #[test]
//         fn into_bytes_length(nibbles in prop::collection::vec(0u8..=15, 0..256)) {
//             let count = nibbles.len();
//             let mut w = NibbleWriter::new();
//             for &n in &nibbles {
//                 w.write(n);
//             }
//             let bytes = w.into_bytes();
//             prop_assert_eq!(bytes.len(), count.div_ceil(2));
//         }
//     }
// }

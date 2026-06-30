//! Online (self-building) word dictionary — an LRU alternative to the shipped
//! static [`super::word_dict::WordDict`].
//!
//! The static dict ships a corpus-mined word list (paid as `L(D)`, counted 2×).
//! This one ships nothing: encoder and decoder build the identical dictionary as
//! they scan, so the cost of learning a word is paid once, in the stream (1×
//! `L(C)`), as that word's first literal occurrence. A word is added to the dict
//! when first seen (emitted literally that time) and coded on every later
//! occurrence; the dict is capped at `cap` entries with LRU eviction, which both
//! bounds memory and self-limits how much of the stream is tokenized (the knob
//! CDR wanted, and one a generic compressor needs regardless).
//!
//! Codes reuse the same free-byte [`code_pool`] and three-tier scheme as the
//! static dict (24 one-byte slots, then two- and three-byte `lead + index` slots),
//! so `cap` ≤ `1_057_304` — large enough to hold every distinct word on a multi-MB
//! slice with no eviction at all. A word takes the shortest free slot it qualifies
//! for, so the early (and therefore, in English, common)
//! stopwords land in the one-byte slots and stay there (LRU keeps reused words
//! alive). Frequency-optimal code assignment (promotion) is a deliberate follow-up
//! — this measures the online idea, not its tuned form.

use std::collections::HashMap;

use super::dictionary::code_pool;
use super::word_dict::min_word_len;

const SINGLE_END: usize = 24; // one-byte code slots (mirrors word_dict::N_SINGLE)
const N_LEAD2: usize = 34; // two-byte lead bytes
const N_LEAD3: usize = 16; // three-byte lead bytes (24 + 34 + 16 == pool.len() == 74)
const PAIR_END: usize = SINGLE_END + N_LEAD2 * 256; // 8728
const TRIP_END: usize = PAIR_END + N_LEAD3 * 65536; // 1_057_304
const NIL: usize = usize::MAX;

/// A capacity-bounded, LRU-evicted word dictionary built online from the stream.
pub(crate) struct OnlineDict {
    pool: Vec<u8>,
    slot_of: HashMap<Vec<u8>, usize>,
    word_at: Vec<Option<Vec<u8>>>,
    // intrusive LRU over occupied slots (MRU at `head`, victim at `tail`)
    prev: Vec<usize>,
    next: Vec<usize>,
    head: usize,
    tail: usize,
    free1: Vec<usize>, // free one-byte slots
    free2: Vec<usize>, // free two-byte slots
    free3: Vec<usize>, // free three-byte slots
    evict: bool,       // full + evict: LRU-recycle a slot (reuses IDs); else freeze
    // static byte classification for decode
    tier1_slot: Vec<Option<usize>>, // byte -> one-byte-code slot
    tier2_l: Vec<Option<usize>>,    // byte -> two-byte lead index `l`
    tier3_l: Vec<Option<usize>>,    // byte -> three-byte lead index `m`
}

impl OnlineDict {
    /// LRU-evicting dictionary: when full, recycles the least-recently-used slot
    /// for a new word (reuses code IDs — non-stationary, measured to be costly).
    pub(crate) fn new(cap: usize) -> Self {
        Self::build(cap, true)
    }

    /// Fill-then-freeze dictionary: builds online up to `cap` entries, then stops
    /// adding (a code ID, once minted, is never reassigned). The decoder-rebuildable
    /// analog of a fixed-size static dict, mined from the stream instead of shipped.
    pub(crate) fn frozen(cap: usize) -> Self {
        Self::build(cap, false)
    }

    fn build(cap: usize, evict: bool) -> Self {
        let cap = cap.clamp(1, TRIP_END);
        let pool = code_pool();
        let mut tier1_slot = vec![None; 256];
        let mut tier2_l = vec![None; 256];
        let mut tier3_l = vec![None; 256];
        for (j, &b) in pool.iter().take(SINGLE_END).enumerate() {
            tier1_slot[b as usize] = Some(j);
        }
        for (l, &b) in pool[SINGLE_END..SINGLE_END + N_LEAD2].iter().enumerate() {
            tier2_l[b as usize] = Some(l);
        }
        for (m, &b) in pool[SINGLE_END + N_LEAD2..SINGLE_END + N_LEAD2 + N_LEAD3]
            .iter()
            .enumerate()
        {
            tier3_l[b as usize] = Some(m);
        }
        let free1: Vec<usize> = (0..cap.min(SINGLE_END)).collect();
        let free2: Vec<usize> = (SINGLE_END.min(cap)..PAIR_END.min(cap)).collect();
        let free3: Vec<usize> = (PAIR_END.min(cap)..cap).collect();
        Self {
            pool,
            slot_of: HashMap::new(),
            word_at: vec![None; cap],
            prev: vec![NIL; cap],
            next: vec![NIL; cap],
            head: NIL,
            tail: NIL,
            free1,
            free2,
            free3,
            evict,
            tier1_slot,
            tier2_l,
            tier3_l,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn emit_code(&self, slot: usize, out: &mut Vec<u8>) {
        if slot < SINGLE_END {
            out.push(self.pool[slot]);
        } else if slot < PAIR_END {
            let p = slot - SINGLE_END;
            out.push(self.pool[SINGLE_END + p / 256]);
            out.push((p % 256) as u8);
        } else {
            let q = slot - PAIR_END;
            out.push(self.pool[SINGLE_END + N_LEAD2 + q / 65536]);
            out.push(((q >> 8) & 0xff) as u8);
            out.push((q & 0xff) as u8);
        }
    }

    fn push_front(&mut self, s: usize) {
        self.prev[s] = NIL;
        self.next[s] = self.head;
        if self.head != NIL {
            self.prev[self.head] = s;
        }
        self.head = s;
        if self.tail == NIL {
            self.tail = s;
        }
    }

    fn unlink(&mut self, s: usize) {
        let (p, n) = (self.prev[s], self.next[s]);
        if p == NIL {
            self.head = n;
        } else {
            self.next[p] = n;
        }
        if n == NIL {
            self.tail = p;
        } else {
            self.prev[n] = p;
        }
    }

    fn touch(&mut self, s: usize) {
        self.unlink(s);
        self.push_front(s);
    }

    /// Add `w` (a freshly-seen word) to the dictionary if a qualifying slot is
    /// available, evicting the LRU entry when full. A no-op if `w` cannot be placed
    /// (e.g. a two-char word once the one-byte slots are gone) — it stays literal.
    fn insert(&mut self, w: &[u8]) {
        let need = w.len();
        let slot = if need >= 2 && !self.free1.is_empty() {
            self.free1.pop().unwrap()
        } else if need >= 3 && !self.free2.is_empty() {
            self.free2.pop().unwrap()
        } else if need >= 4 && !self.free3.is_empty() {
            self.free3.pop().unwrap()
        } else if self.free1.is_empty() && self.free2.is_empty() && self.free3.is_empty() {
            if !self.evict {
                return; // frozen: full, never reassign a code ID
            }
            let v = self.tail;
            if v == NIL || min_word_len(v) > need {
                return;
            }
            let old = self.word_at[v].take().unwrap();
            self.slot_of.remove(&old);
            self.unlink(v);
            v
        } else {
            return; // word too short for any remaining free tier
        };
        self.slot_of.insert(w.to_vec(), slot);
        self.word_at[slot] = Some(w.to_vec());
        self.push_front(slot);
    }

    /// Encode side: replace known words with their code, learn new words.
    pub(crate) fn forward(&mut self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len());
        let mut i = 0;
        while i < input.len() {
            if input[i].is_ascii_lowercase() {
                let s = i;
                while i < input.len() && input[i].is_ascii_lowercase() {
                    i += 1;
                }
                let w = &input[s..i];
                if let Some(&slot) = self.slot_of.get(w) {
                    self.emit_code(slot, &mut out);
                    self.touch(slot);
                } else {
                    out.extend_from_slice(w);
                    self.insert(w);
                }
            } else {
                out.push(input[i]);
                i += 1;
            }
        }
        out
    }

    /// Decode side: rebuild the identical dictionary and expand codes.
    pub(crate) fn inverse(&mut self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len() * 2);
        let mut i = 0;
        while i < input.len() {
            let b = input[i];
            if let Some(slot) = self.tier1_slot[b as usize] {
                out.extend_from_slice(self.word_at[slot].as_ref().unwrap());
                self.touch(slot);
                i += 1;
            } else if let Some(l) = self.tier2_l[b as usize] {
                let slot = SINGLE_END + l * 256 + usize::from(input[i + 1]);
                out.extend_from_slice(self.word_at[slot].as_ref().unwrap());
                self.touch(slot);
                i += 2;
            } else if let Some(m) = self.tier3_l[b as usize] {
                let slot = PAIR_END
                    + m * 65536
                    + (usize::from(input[i + 1]) << 8)
                    + usize::from(input[i + 2]);
                out.extend_from_slice(self.word_at[slot].as_ref().unwrap());
                self.touch(slot);
                i += 3;
            } else if b.is_ascii_lowercase() {
                let s = i;
                while i < input.len() && input[i].is_ascii_lowercase() {
                    i += 1;
                }
                let w = input[s..i].to_vec();
                out.extend_from_slice(&w);
                self.insert(&w);
            } else {
                out.push(b);
                i += 1;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_repeated_words() {
        let data = b"\x00the category the theory the category and the end the\n";
        let mut enc = OnlineDict::new(2000);
        let fwd = enc.forward(data);
        assert!(fwd.len() < data.len(), "repeated words should shrink");
        let mut dec = OnlineDict::new(2000);
        assert_eq!(dec.inverse(&fwd), data);
    }

    #[test]
    fn roundtrips_under_eviction() {
        // cap=2 forces LRU churn; round-trip must still be exact.
        let data =
            b"alpha beta gamma alpha delta beta gamma alpha epsilon beta alpha gamma delta beta";
        let mut enc = OnlineDict::new(2);
        let fwd = enc.forward(data);
        let mut dec = OnlineDict::new(2);
        assert_eq!(dec.inverse(&fwd), data);
    }

    #[test]
    fn frozen_roundtrips_and_never_reuses() {
        // cap=2 freezes after two entries; later new words stay literal forever
        // (no ID reuse), and the round-trip is still exact.
        let data = b"alpha beta alpha gamma beta delta alpha gamma beta alpha";
        let mut enc = OnlineDict::frozen(2);
        let fwd = enc.forward(data);
        let mut dec = OnlineDict::frozen(2);
        assert_eq!(dec.inverse(&fwd), data);
    }

    #[test]
    fn first_word_is_literal_second_is_coded() {
        let data = b"hello hello";
        let mut enc = OnlineDict::new(2000);
        let fwd = enc.forward(data);
        // "hello" (5) literal + space + 1-byte code = 7 bytes, vs 11.
        assert_eq!(fwd.len(), 7);
        let mut dec = OnlineDict::new(2000);
        assert_eq!(dec.inverse(&fwd), data);
    }
}

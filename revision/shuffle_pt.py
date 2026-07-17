#!/usr/bin/env python3
"""
shuffle_pt.py — globally shuffle the RECORD ORDER of a pre-built UER `.pt` dataset.

Why this is safe (and why it fixes the acc_sz spike-then-drop):
  A `.pt` file here is NOT a single torch.save object. It is a flat stream of
  back-to-back `pickle.dump(instance, f)` records (uer/utils/data.py: workers
  dump per-instance, merge_dataset() byte-concatenates them, and DataLoader._fill_buf
  reads them with `pickle.load` in a loop until EOFError).

  Each record is ONE self-contained flow sample:
    - multimodal .pt : (raw_src, raw_packet_ids, raw_directions_seq,
                        size_src, iat_src, [tgt_mlm_size, tgt_mlm_temporal])
                       -> raw<->size<->IAT alignment lives INSIDE the record.
    - stage-1 .pt    : a single unimodal sample.
  Because nothing in a record references another record, permuting the record
  ORDER cannot break any alignment. No shared cross-file permutation is needed.

  The corpus is class-blocked and the trainer only does a buffer-LOCAL shuffle
  (instances_buffer_size sliding window). So a report window that sits inside a
  homogeneous class block sees trivially-predictable masked tokens (acc_sz ~0.97)
  and collapses (~0.6) at the block boundary. A one-time GLOBAL record shuffle
  de-correlates the stream so every window is class-mixed -> acc_sz smooths out
  and gradients stop being block-correlated. No regeneration required.

How it works (byte-exact, no re-pickling):
  Pass 1  index every record's [start, end) byte span via pickle.load boundaries.
          Policy: SKIP ONLY THE BROKEN RECORD, KEEP EVERY OTHER SAMPLE WHOLE.
          A truncated/corrupt record (e.g. a crashed preprocess worker whose
          half-written /tmp/<i>.pt was byte-concatenated into the merged .pt by
          merge_dataset) is skipped by RESYNCING to the next confirmed pickle frame
          (a position from which several records load cleanly, so we never lock onto
          a false in-payload boundary and mangle the good samples that follow). Thus
          every record AFTER a bad one is RECOVERED rather than silently dropped —
          unlike a plain pickle.load loop, which stops at the first bad record and
          loses the whole remainder. Each KEPT span is one that unpickled cleanly, so
          every output sample is complete; --verify reloads them all to prove it.
          Use --inspect to report damage without writing.
  Shuffle the spans with a seeded RNG.
  Pass 2  copy each record's raw bytes, in shuffled order, to the output.
  The unpickled objects are never re-serialized, so contents are byte-identical;
  only their order changes. Pure stdlib — no torch / numpy / UER imports.

NOTE: a MID-FILE corruption point (valid records exist after it) means the merged
  .pt is damaged and also breaks training, because DataLoader._fill_buf only catches
  EOFError; the shuffled output this writes is the repaired (gap-free) file.

Usage:
  # Write a shuffled copy next to the original (safest; original untouched):
  python data_generation/shuffle_pt.py /path/to/dataset.pt --seed 42
      -> /path/to/dataset.pt.shuffled

  # In place, keeping the trainer's path unchanged (original saved as .bak):
  python data_generation/shuffle_pt.py /path/to/dataset.pt --inplace

  # Faster when the file fits in RAM (sequential write instead of random seeks):
  python data_generation/shuffle_pt.py /path/to/dataset.pt --inplace --in-memory

  # Stage-1 has two independent unimodal .pt files; just run it on each:
  python data_generation/shuffle_pt.py raw_dataset.pt  --inplace
  python data_generation/shuffle_pt.py size_dataset.pt --inplace
"""

import argparse
import mmap
import os
import pickle
import random
import time


_PROTO = b"\x80"  # pickle protocol>=2: every pickle.dump record starts with PROTO (\x80 + ver)


def _try_load_at(mm, pos):
    """Attempt to unpickle one record starting at byte `pos`.
    Returns the end offset on success, or None if it does not parse cleanly there
    (EOFError, UnpicklingError, ValueError, KeyError, ... — any malformed record)."""
    mm.seek(pos)
    try:
        pickle.load(mm)
    except Exception:  # noqa: BLE001 - probing; any failure means "not a record here"
        return None
    return mm.tell()


_CONFIRM = 3  # consecutive clean records required to accept a resync point


def _loads_n_from(mm, pos, size, need):
    """True iff `need` consecutive records (or however many remain before EOF) all
    unpickle cleanly starting at byte `pos`.

    A stray b'\\x80' inside a record's payload can occasionally parse as ONE bogus
    frame, which would misalign — and thus silently corrupt — every good record
    after it. Requiring a short RUN of clean frames rejects those false boundaries:
    we only resync onto a position that is really the start of a record, so the
    good samples that follow are kept intact. (This assumes corruption points are
    more than `need` records apart, which holds for per-worker truncation seams.)"""
    cur = pos
    for _ in range(need):
        end = _try_load_at(mm, cur)
        if end is None:
            return False
        cur = end
        if cur >= size:      # fewer than `need` records remain — that's fine
            break
    return True


def _resync(mm, start, size):
    """Find the next byte offset >= `start` that is a REAL record boundary.

    Records are pickle.dump frames, each beginning with the PROTO opcode b'\\x80'.
    We scan for b'\\x80' candidates and accept the first from which `_CONFIRM`
    consecutive records unpickle cleanly, jumping over the partial bytes of a
    truncated/corrupt record without ever locking onto a false in-payload b'\\x80'.
    Returns the offset, or None if no valid frame remains."""
    pos = start
    while pos < size:
        cand = mm.find(_PROTO, pos)
        if cand < 0:
            return None
        if _loads_n_from(mm, cand, size, _CONFIRM):
            return cand
        pos = cand + 1
    return None


def index_records(mm, size):
    """Index every COMPLETE pickle record's [start, end) byte span in an mmap.

    Returns (spans, gaps):
      * spans — [(start, end), ...] of every record that unpickles cleanly.
      * gaps  — [(bad_start, resume), ...] byte ranges skipped because a record was
                truncated/corrupt. A clean file gives gaps == []. A gap with
                resume == size is a truncated FINAL record (benign). A gap with
                resume < size is MID-FILE corruption: the merged .pt is damaged (a
                crashed/truncated preprocess worker) — which also breaks training,
                since DataLoader._fill_buf only catches EOFError.

    Resyncs past corruption, so records after a bad one are RECOVERED, not lost.
    """
    spans, gaps = [], []
    pos = 0
    while pos < size:
        end = _try_load_at(mm, pos)
        if end is not None:
            spans.append((pos, end))
            pos = end
            continue
        resume = _resync(mm, pos + 1, size)
        if resume is None:
            gaps.append((pos, size))      # nothing valid remains -> truncated tail
            break
        gaps.append((pos, resume))        # corrupt record; recovered from `resume`
        pos = resume
    if not spans:
        raise RuntimeError("Could not parse any pickle record (is this a UER .pt?).")
    return spans, gaps


def _copy_span(reader, writer, start, end):
    reader.seek(start)
    writer.write(reader.read(end - start))


def shuffle_offset_mode(in_path, out_path, spans, seed):
    """Memory-light: hold only byte spans; pass 2 seeks + copies raw bytes."""
    order = list(range(len(spans)))
    random.Random(seed).shuffle(order)
    with open(in_path, "rb") as reader, open(out_path, "wb") as writer:
        for idx in order:
            s, e = spans[idx]
            _copy_span(reader, writer, s, e)
        writer.flush()
        os.fsync(writer.fileno())


def shuffle_in_memory_mode(buf, out_path, spans, seed):
    """Faster when the file fits in RAM: hold record blobs, write sequentially."""
    blobs = [buf[s:e] for (s, e) in spans]
    random.Random(seed).shuffle(blobs)
    with open(out_path, "wb") as writer:
        for blob in blobs:
            writer.write(blob)
        writer.flush()
        os.fsync(writer.fileno())


def verify(out_path, expected_records, expected_bytes):
    """Stream the output back and assert record count + total size are preserved."""
    size = os.path.getsize(out_path)
    if size != expected_bytes:
        raise RuntimeError(
            f"Verify FAILED: output size {size} != input size {expected_bytes}"
        )
    count = 0
    with open(out_path, "rb") as f:
        while True:
            try:
                pickle.load(f)
            except EOFError:
                break
            count += 1
    if count != expected_records:
        raise RuntimeError(
            f"Verify FAILED: output has {count} records, expected {expected_records}"
        )
    return count


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024


def main():
    p = argparse.ArgumentParser(
        description="Globally shuffle the record order of a UER .pt dataset (byte-exact, no regen).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Each record is one self-contained flow sample, so reordering records "
        "is always safe; raw/size/IAT alignment is intra-record.",
    )
    p.add_argument("input", help="path to the .pt dataset to shuffle")
    p.add_argument("-o", "--output", help="output path (default: <input>.shuffled)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument("--inplace", action="store_true",
                   help="replace the input file; original saved as <input>.bak")
    p.add_argument("--no-backup", action="store_true",
                   help="with --inplace, do NOT keep a .bak of the original")
    p.add_argument("--in-memory", action="store_true",
                   help="load the whole file into RAM (faster I/O; needs ~filesize RAM)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the post-write verification pass")
    p.add_argument("--inspect", "--dry-run", action="store_true", dest="inspect",
                   help="only index + report record count and any corruption points; "
                        "do NOT shuffle or write output")
    args = p.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        p.error(f"input not found: {in_path}")
    if args.inplace and args.output:
        p.error("--inplace and --output are mutually exclusive")

    in_bytes = os.path.getsize(in_path)
    final_path = in_path if args.inplace else (args.output or in_path + ".shuffled")
    # Write to a temp file first; never clobber until verified.
    tmp_path = final_path + ".shuffling.tmp"

    t0 = time.time()
    print(f"[shuffle_pt] input : {in_path}  ({human(in_bytes)})")
    print(f"[shuffle_pt] seed  : {args.seed}   mode: "
          f"{'in-memory' if args.in_memory else 'offset'}")

    # ---- Pass 1: index record boundaries (mmap; recovers past mid-file corruption)
    if in_bytes == 0:
        p.error("input file is empty — nothing to shuffle")
    with open(in_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            spans, gaps = index_records(mm, mm.size())
        finally:
            mm.close()
    n = len(spans)
    out_bytes = sum(e - s for s, e in spans)
    midfile = [(s, e) for (s, e) in gaps if e < in_bytes]   # corruption with records after it
    tail = [(s, e) for (s, e) in gaps if e >= in_bytes]     # truncated final record (benign)
    dropped = sum(e - s for s, e in gaps)
    print(f"[shuffle_pt] records: {n:,} complete samples kept  ({human(out_bytes)}; "
          f"{time.time()-t0:.1f}s to index)")
    if midfile:
        print(f"[shuffle_pt] WARN  : {len(midfile)} MID-FILE corruption point(s) — the "
              f"merged .pt is damaged (a truncated/crashed preprocess worker). The bad "
              f"records are dropped; everything after is RECOVERED by resync:")
        for (bs, be) in midfile[:10]:
            print(f"               corrupt @ byte {bs:,} -> resynced @ {be:,} "
                  f"(skipped {human(be - bs)})")
        if len(midfile) > 10:
            print(f"               ... and {len(midfile) - 10} more point(s)")
        print(f"[shuffle_pt] WARN  : the SAME corruption breaks training — "
              f"DataLoader._fill_buf only catches EOFError. Re-check preprocessing; "
              f"meanwhile train on this shuffled output (it is gap-free/repaired).")
    if tail:
        print(f"[shuffle_pt] note  : ignoring a truncated final record "
              f"({human(sum(e - s for s, e in tail))}) — benign.")
    if dropped:
        print(f"[shuffle_pt] total : kept {n:,} records, dropped {human(dropped)} corrupt "
              f"({100.0 * dropped / in_bytes:.4f}% of file).")

    if args.inspect:
        print("[shuffle_pt] inspect-only: no output written "
              f"({'CLEAN' if not gaps else 'MID-FILE CORRUPTION' if midfile else 'benign tail only'}).")
        return

    # ---- Shuffle + write ----------------------------------------------------
    if args.in_memory:
        with open(in_path, "rb") as f:
            buf = f.read()
        shuffle_in_memory_mode(buf, tmp_path, spans, args.seed)
    else:
        shuffle_offset_mode(in_path, tmp_path, spans, args.seed)
    print(f"[shuffle_pt] wrote : {tmp_path}  ({time.time()-t0:.1f}s elapsed)")

    # ---- Verify before committing ------------------------------------------
    if not args.no_verify:
        got = verify(tmp_path, n, out_bytes)
        print(f"[shuffle_pt] verify: OK  ({got:,} records, byte-count matches)")

    # ---- Commit (atomic rename) --------------------------------------------
    if args.inplace and not args.no_backup:
        bak = in_path + ".bak"
        os.replace(in_path, bak)
        print(f"[shuffle_pt] backup: {bak}")
    os.replace(tmp_path, final_path)
    print(f"[shuffle_pt] DONE  : {final_path}  ({time.time()-t0:.1f}s total)")
    if not args.inplace:
        print("[shuffle_pt] point your training config's dataset_path at this file, "
              "or rerun with --inplace to keep the same path.")


if __name__ == "__main__":
    main()

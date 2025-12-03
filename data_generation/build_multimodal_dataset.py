"""
Build Multi-Modal Dataset from Corpus Files

Converts paired corpus files (Raw + Size) to .pt dataset format for Stage 2 training.

Usage:
    python data_generation/build_multimodal_dataset.py \
        --corpus_path_raw data/corpus_raw.txt \
        --corpus_path_size data/corpus_size.txt \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --dataset_path data/multimodal_dataset.pt \
        --seq_length_raw 512 \
        --seq_length_size 256 \
        --processes_num 64
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uer.utils.vocab import Vocab
from uer.utils.tokenizers import SpaceTokenizer
from uer.utils.data import MultiModalDataset


def main():
    parser = argparse.ArgumentParser(description='Build Multi-Modal Dataset')

    # ===== Required Arguments =====
    parser.add_argument("--corpus_path_raw", type=str, required=True,
                       help="Path to raw packet corpus file")
    parser.add_argument("--corpus_path_size", type=str, required=True,
                       help="Path to packet size corpus file")
    parser.add_argument("--vocab_path_raw", type=str, required=True,
                       help="Path to raw packet vocabulary")
    parser.add_argument("--vocab_path_size", type=str, required=True,
                       help="Path to packet size vocabulary")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Output dataset path (.pt file)")

    # ===== Sequence Lengths =====
    parser.add_argument("--seq_length_raw", type=int, default=512,
                       help="Max sequence length for raw packets (default: 512)")
    parser.add_argument("--seq_length_size", type=int, default=256,
                       help="Max sequence length for packet sizes (default: 256)")

    # ===== Processing Options =====
    parser.add_argument("--processes_num", type=int, default=1,
                       help="Number of worker processes (default: 1)")
    parser.add_argument("--docs_buffer_size", type=int, default=100000,
                       help="Document buffer size (default: 100000)")
    parser.add_argument("--dup_factor", type=int, default=2,
                       help="Duplication factor for data augmentation (default: 1)")

    # ===== Masking Options =====
    parser.add_argument("--dynamic_masking", action="store_true",
                       help="Enable dynamic masking (recommended)")
    parser.add_argument("--whole_word_masking", action="store_true",
                       help="Enable whole word masking")
    parser.add_argument("--span_masking", action="store_true",
                       help="Enable span masking")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                       help="Geometric probability for span masking (default: 0.2)")
    parser.add_argument("--span_max_length", type=int, default=10,
                       help="Maximum span length (default: 10)")

    # ===== Other Options =====
    parser.add_argument("--short_seq_prob", type=float, default=0.0,
                       help="Probability of short sequences (default: 0.0)")
    parser.add_argument("--seed", type=int, default=7,
                       help="Random seed (default: 7)")

    args = parser.parse_args()

    if args.dynamic_masking:
        print("Dynamic masking is enabled.")
        args.dup_factor = 1

    # Print configuration
    print("=" * 80)
    print("Building Multi-Modal Dataset")
    print("=" * 80)
    print(f"Raw Corpus:  {args.corpus_path_raw}")
    print(f"Size Corpus: {args.corpus_path_size}")
    print(f"Raw Vocab:   {args.vocab_path_raw}")
    print(f"Size Vocab:  {args.vocab_path_size}")
    print(f"Output:      {args.dataset_path}")
    print(f"Seq Lengths: Raw={args.seq_length_raw}, Size={args.seq_length_size}")
    print(f"Processes:   {args.processes_num}")
    print("=" * 80)
    print()

    # ===== Load Vocabularies =====
    print("[1/3] Loading vocabularies...")

    vocab_raw = Vocab()
    vocab_raw.load(args.vocab_path_raw)
    print(f"  Raw vocab size: {len(vocab_raw)}")

    vocab_size = Vocab()
    vocab_size.load(args.vocab_path_size)
    print(f"  Size vocab size: {len(vocab_size)}")
    print()

    # ===== Create Tokenizers =====
    print("[2/3] Creating tokenizers...")
    tokenizer_raw = SpaceTokenizer(args)
    tokenizer_size = SpaceTokenizer(args)
    print()

    # Store vocabularies and tokenizers in args for dataset
    args.vocab_raw = vocab_raw.w2i
    args.vocab_size = vocab_size.w2i
    args.tokenizer_raw = tokenizer_raw
    args.tokenizer_size = tokenizer_size

    # ===== Build Dataset =====
    print("[3/3] Building dataset...")
    if args.processes_num > 1:
        print(f"  This may take a while with {args.processes_num} processes...")
    else:
        print("  This may take a while...")

    dataset = MultiModalDataset(
        args,
        vocab_raw=args.vocab_raw,
        vocab_size=args.vocab_size,
        tokenizer_raw=tokenizer_raw,
        tokenizer_size=tokenizer_size
    )

    dataset.build_and_save(args.processes_num)

    print()
    print("=" * 80)
    print(f"✓ Dataset successfully saved to: {args.dataset_path}")

    # Print file size
    if os.path.exists(args.dataset_path):
        file_size = os.path.getsize(args.dataset_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()

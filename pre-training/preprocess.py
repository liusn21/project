#!/usr/bin/python3
#-*- coding:utf-8 -*-
"""
Preprocess corpus files to .pt dataset format for pretraining.

Supports multiple targets:
- bert, mlm, lm, etc.: Standard text pretraining
- raw_packet: Raw packet pretraining (Stage 1)
- packet_size: Packet size + IAT pretraining (Stage 1)
- multimodal: Multi-modal pretraining (Stage 2) - requires paired Raw + Size corpora

Usage for multimodal:
    python pre-training/preprocess.py \
        --target multimodal \
        --corpus_path_raw data/corpus_raw.txt \
        --corpus_path_size data/corpus_size.txt \
        --vocab_path_raw models/vocab_raw.txt \
        --vocab_path_size models/vocab_size.txt \
        --vocab_path_temporal models/vocab_temporal.txt \
        --dataset_path data/multimodal_dataset.pt \
        --seq_length_raw 512 \
        --seq_length_size 256 \
        --processes_num 64 \
        --dynamic_masking
"""
import argparse
import os
import sys
sys.path.append(os.getcwd())
import six
from packaging import version
from uer.utils.data import *
from uer.utils import *


assert version.parse(six.__version__) >= version.parse("1.12.0")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--corpus_path", type=str, default=None,
                        help="Path of the corpus for pretraining. "
                             "Required for non-multimodal targets. "
                             "For multimodal target, use --corpus_path_raw and --corpus_path_size instead.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
    parser.add_argument("--vocab_path_temporal", default=None, type=str,
                        help="Path of the temporal vocabulary file (for IAT tokens in packet_size/multimodal target).")
    parser.add_argument("--dataset_path", type=str, default="dataset.pt",
                        help="Path of the preprocessed dataset.")

    # Multi-modal specific options (Stage 2)
    parser.add_argument("--corpus_path_raw", default=None, type=str,
                        help="Path of the raw packet corpus (for multimodal target).")
    parser.add_argument("--corpus_path_size", default=None, type=str,
                        help="Path of the packet size corpus with IAT (for multimodal target).")
    parser.add_argument("--vocab_path_raw", default=None, type=str,
                        help="Path of the raw packet vocabulary (for multimodal target).")
    parser.add_argument("--vocab_path_size", default=None, type=str,
                        help="Path of the packet size vocabulary (for multimodal target).")
    parser.add_argument("--seq_length_raw", type=int, default=512,
                        help="Sequence length for raw packets (for multimodal target).")
    parser.add_argument("--seq_length_size", type=int, default=256,
                        help="Sequence length for packet sizes (for multimodal target).")

    # Preprocess options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )
    parser.add_argument("--tgt_tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")
    parser.add_argument("--processes_num", type=int, default=1,
                        help="Split the whole dataset into `processes_num` parts, "
                             "and each part is fed to a single process in training step.")
    parser.add_argument("--target", choices=["bert", "bertflow","lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm","raw_packet","packet_size","multimodal"], default="bert",
                        help="The training dataset target.")
    parser.add_argument("--docs_buffer_size", type=int, default=100000,
                        help="The buffer size of documents in memory, specific to targets that require negative sampling.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length of instances.")
    parser.add_argument("--tgt_seq_length", type=int, default=128, help="Target sequence length of instances.")
    parser.add_argument("--dup_factor", type=int, default=5,
                        help="Duplicate instances multiple times.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of truncating sequence."
                             "The larger value, the higher probability of using short (truncated) sequence.")
    parser.add_argument("--full_sentences", action="store_true", help="Full sentences.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Masking options.
    parser.add_argument("--dynamic_masking", action="store_true", help="Dynamic masking.")
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")

    args = parser.parse_args()

    # Dynamic masking.
    if args.dynamic_masking:
        args.dup_factor = 1

    # ===== Handle multimodal target separately =====
    if args.target == "multimodal":
        build_multimodal_dataset(args)
        return


    # Build tokenizer.
    tokenizer = str2tokenizer[args.tokenizer](args)
    if args.target == "seq2seq":
        args.tgt_tokenizer = str2tokenizer[args.tgt_tokenizer](args, False)

    # Build temporal tokenizer for packet_size target (if vocab_path_temporal is provided)
    tokenizer_temporal = None
    if args.target == "packet_size" and hasattr(args, 'vocab_path_temporal') and args.vocab_path_temporal:
        # Create a temporary args object for temporal tokenizer
        import copy
        args_temporal = copy.copy(args)
        args_temporal.vocab_path = args.vocab_path_temporal
        tokenizer_temporal = str2tokenizer[args.tokenizer](args_temporal)
        print(f"Loaded temporal vocab from {args.vocab_path_temporal} (size: {len(tokenizer_temporal.vocab)})")

    # Build and save dataset.
    if args.target == "packet_size" and tokenizer_temporal is not None:
        dataset = str2dataset[args.target](args, tokenizer.vocab, tokenizer, tokenizer_temporal.vocab, tokenizer_temporal)
    else:
        dataset = str2dataset[args.target](args, tokenizer.vocab, tokenizer)
    dataset.build_and_save(args.processes_num, split_by_flow=True)


def build_multimodal_dataset(args):
    """
    Build multi-modal dataset for Stage 2 pretraining.

    Requires paired Raw + Size corpora with matching flow counts.
    """
    from uer.utils.vocab import Vocab
    from uer.utils.tokenizers import BertTokenizer
    from uer.utils.data import MultiModalDataset

    # Validate required arguments
    if not args.corpus_path_raw:
        raise ValueError("--corpus_path_raw is required for multimodal target")
    if not args.corpus_path_size:
        raise ValueError("--corpus_path_size is required for multimodal target")
    if not args.vocab_path_raw:
        raise ValueError("--vocab_path_raw is required for multimodal target")
    if not args.vocab_path_size:
        raise ValueError("--vocab_path_size is required for multimodal target")

    # Print configuration
    print("=" * 80)
    print("Building Multi-Modal Dataset (Stage 2)")
    print("=" * 80)
    print(f"Raw Corpus:     {args.corpus_path_raw}")
    print(f"Size Corpus:    {args.corpus_path_size}")
    print(f"Raw Vocab:      {args.vocab_path_raw}")
    print(f"Size Vocab:     {args.vocab_path_size}")
    print(f"Temporal Vocab: {args.vocab_path_temporal or '(using size vocab)'}")
    print(f"Output:         {args.dataset_path}")
    print(f"Seq Lengths:    Raw={args.seq_length_raw}, Size={args.seq_length_size}")
    print(f"Processes:      {args.processes_num}")
    print(f"Dynamic Mask:   {args.dynamic_masking}")
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

    # Load temporal vocabulary (for IAT tokens)
    if args.vocab_path_temporal:
        vocab_temporal = Vocab()
        vocab_temporal.load(args.vocab_path_temporal)
        print(f"  Temporal vocab size: {len(vocab_temporal)}")
    else:
        vocab_temporal = vocab_size
        print(f"  Temporal vocab: using size vocab ({len(vocab_temporal)})")
    print()

    # ===== Create Tokenizers =====
    print("[2/3] Creating tokenizers...")

    # Raw tokenizer
    args.vocab_path = args.vocab_path_raw
    tokenizer_raw = BertTokenizer(args)

    # Size tokenizer
    args.vocab_path = args.vocab_path_size
    tokenizer_size = BertTokenizer(args)

    # Temporal tokenizer
    args.vocab_path = args.vocab_path_temporal if args.vocab_path_temporal else args.vocab_path_size
    tokenizer_temporal = BertTokenizer(args)

    # Store vocabularies and tokenizers in args for dataset
    args.vocab_raw = vocab_raw.w2i
    args.vocab_size = vocab_size.w2i
    args.vocab_temporal = vocab_temporal.w2i
    args.tokenizer_raw = tokenizer_raw
    args.tokenizer_size = tokenizer_size
    args.tokenizer_temporal = tokenizer_temporal

    # ===== Build Dataset =====
    print("[3/3] Building dataset...")
    if args.processes_num > 1:
        print(f"  Using {args.processes_num} parallel processes...")
    else:
        print("  Using single process...")

    dataset = MultiModalDataset(
        args,
        vocab_raw=args.vocab_raw,
        vocab_size=args.vocab_size,
        tokenizer_raw=tokenizer_raw,
        tokenizer_size=tokenizer_size,
        vocab_temporal=args.vocab_temporal,
        tokenizer_temporal=tokenizer_temporal
    )

    dataset.build_and_save(args.processes_num)

    print()
    print("=" * 80)
    print(f"Dataset successfully saved to: {args.dataset_path}")


if __name__ == "__main__":
    main()

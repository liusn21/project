"""
Multi-Modal Traffic Pre-training Script

Usage example:
    python pretrain_multimodal.py \
        --json_path data/multimodal_flows.json \
        --output_model_path models/multimodal_pretrain.bin \
        --config_path models/bert/base_config.json \
        --embedding multimodal \
        --encoder transformer \
        --target simsiam_mbm \
        --seq_length 512 \
        --batch_size 32 \
        --learning_rate 2e-4 \
        --total_steps 100000
"""

import os
import sys
sys.path.append(os.getcwd())
import argparse
import torch
import uer.trainer as trainer
from uer.utils.config import load_hyperparam
from uer.opts import *


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to the multi-modal JSON file (output from multimodal_data_gen.py)")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path of the output model.")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model (for continued training).")
    parser.add_argument("--config_path", type=str, default="models/bert/base_config.json",
                        help="Config file of model hyper-parameters.")

    # Training options
    parser.add_argument("--total_steps", type=int, default=100000,
                        help="Total training steps.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print progress.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Maximum sequence length.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Model options
    model_opts(parser)
    parser.add_argument("--embedding", type=str, default="multimodal",
                        help="Embedding type (must be 'multimodal')")
    parser.add_argument("--encoder", type=str, default="transformer",
                        help="Encoder type")
    parser.add_argument("--target", type=str, default="simsiam_mbm",
                        help="Target type (must be 'simsiam_mbm')")

    # Multi-modal specific options
    parser.add_argument("--mask_ratio", type=float, default=0.15,
                        help="Masking ratio for MBM (default: 0.15)")
    parser.add_argument("--mbm_weight", type=float, default=0.1,
                        help="Weight for MBM auxiliary loss (default: 0.1)")
    parser.add_argument("--projection_dim", type=int, default=2048,
                        help="Projection dimension for SimSiam (default: 2048)")

    # Optimizer options
    optimization_opts(parser)

    # GPU options
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int,
                        help="List of GPU ranks for distributed training.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str,
                        help="IP-Port of master for distributed training.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str,
                        help="Distributed backend.")

    args = parser.parse_args()

    # Validate arguments
    if args.embedding != "multimodal":
        raise ValueError("For multi-modal training, --embedding must be 'multimodal'")
    if args.target != "simsiam_mbm":
        raise ValueError("For multi-modal training, --target must be 'simsiam_mbm'")

    # Load hyper-parameters from config file
    if args.config_path:
        load_hyperparam(args)

    # Set dataset_path to json_path for compatibility with trainer
    args.dataset_path = args.json_path

    # Multi-modal doesn't need vocabulary (fixed token spaces)
    args.vocab = None
    args.vocab_path = None
    args.spm_model_path = None

    # Set tokenizer to None (not used for multi-modal)
    args.tokenizer = None

    # GPU configuration
    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        # Multiprocessing distributed mode
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= args.world_size, "Started processes exceed `world_size`."
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceed available GPUs."
        args.dist_train = True
        args.ranks_num = ranks_num
        print("Using distributed mode for training.")
    elif args.world_size == 1 and ranks_num == 1:
        # Single GPU mode
        assert torch.cuda.is_available(), "No available GPUs."
        args.gpu_id = args.gpu_ranks[0]
        assert args.gpu_id < torch.cuda.device_count(), "Invalid GPU device."
        args.dist_train = False
        args.single_gpu = True
        print(f"Using GPU {args.gpu_id} for training.")
    else:
        # CPU mode
        assert ranks_num == 0, "GPUs are specified, please check the arguments."
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode for training.")

    print("\n" + "="*60)
    print("Multi-Modal Traffic Pre-training Configuration")
    print("="*60)
    print(f"JSON Path: {args.json_path}")
    print(f"Output Model: {args.output_model_path}")
    print(f"Embedding: {args.embedding}")
    print(f"Encoder: {args.encoder}")
    print(f"Target: {args.target}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Total Steps: {args.total_steps}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"MBM Weight: {args.mbm_weight}")
    print(f"Projection Dim: {args.projection_dim}")
    print("="*60 + "\n")

    # Start training
    trainer.train_and_validate(args)


if __name__ == "__main__":
    main()

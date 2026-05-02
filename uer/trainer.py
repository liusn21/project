import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.utils.constants import PAD_ID
# Note: Old compute_balance_loss removed in ALBEF-style implementation
from tqdm import tqdm

def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    print("Load vocabulary.")
    if args.target == "multimodal":
        # Multi-modal requires three vocabularies: Raw, Size, Temporal
        print("Loading multi-modal vocabularies...")

        # Raw vocabulary
        vocab_raw = Vocab()
        vocab_raw.load(args.vocab_path_raw)
        args.vocab_raw = vocab_raw.w2i

        # Size vocabulary
        vocab_size = Vocab()
        vocab_size.load(args.vocab_path_size)
        args.vocab_size = vocab_size.w2i

        # Temporal vocabulary (for IAT tokens)
        if hasattr(args, 'vocab_path_temporal') and args.vocab_path_temporal:
            import copy
            vocab_temporal = Vocab()
            vocab_temporal.load(args.vocab_path_temporal)
            args.vocab_temporal = vocab_temporal.w2i
            print(f"  Temporal vocab loaded from {args.vocab_path_temporal} (size: {len(args.vocab_temporal)})")
        else:
            # Fallback: use size vocab for temporal (not recommended)
            args.vocab_temporal = args.vocab_size
            print("  WARNING: No temporal vocab specified, using size vocab as fallback")

        # Create tokenizers for all modalities
        print("Raw tokenizer")
        args.vocab_path = args.vocab_path_raw
        args.tokenizer_raw = str2tokenizer[args.tokenizer](args)

        print("Size tokenizer")
        args.vocab_path = args.vocab_path_size
        args.tokenizer_size = str2tokenizer[args.tokenizer](args)

        print("Temporal tokenizer")
        if hasattr(args, 'vocab_path_temporal') and args.vocab_path_temporal:
            args.vocab_path = args.vocab_path_temporal
            args.tokenizer_temporal = str2tokenizer[args.tokenizer](args)
        else:
            args.tokenizer_temporal = args.tokenizer_size

        # Set main vocab/tokenizer for compatibility
        args.vocab = vocab_raw.w2i
        args.tokenizer = args.tokenizer_raw

    else:
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        args.vocab = args.tokenizer.vocab
        # Load temporal vocabulary for packet_size target (if provided)
        if args.target == "packet_size" and hasattr(args, 'vocab_path_temporal') and args.vocab_path_temporal:
            # Temporarily change vocab_path to load temporal vocab
            original_vocab_path = args.vocab_path
            args.vocab_path = args.vocab_path_temporal
            args.tokenizer_temporal = BertTokenizer(args)
            args.vocab_temporal = args.tokenizer_temporal.vocab
            # Restore original vocab_path
            args.vocab_path = original_vocab_path
            print(f"Loaded temporal vocab from {args.vocab_path_temporal} (size: {len(args.vocab_temporal)})")
        else:
            args.vocab_temporal = None
            args.tokenizer_temporal = None

    # Build model.
    print("Build model.")
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model = load_model(model, args.pretrained_model_path)
    elif args.target == "multimodal":
        # Multimodal model loads Stage 1 weights inside build_model(),
        # only randomly initialize non-pretrained components (fusion, target, projections).
        pretrained_prefixes = ('embedding_raw.', 'encoder_raw.', 'embedding_size.', 'encoder_size.',
                               'embedding_raw_m.', 'encoder_raw_m.', 'embedding_size_m.', 'encoder_size_m.',
                               'itc_proj_raw_m.', 'itc_proj_size_m.')
        # ITGCA gate parameters have specific initializations for implicit curriculum learning
        # (alpha=-2.0 → sigmoid≈0.12, g_default_logit=0.0, bilinear xavier gain=0.1)
        # These must NOT be overwritten by blanket normal_(0, 0.02)
        itgca_param_names = ('alpha_modality', 'g_default_logit',
                             'bilinear_W', 'bilinear_bias',
                             'stat_scale', 'stat_shift',
                             'local_stat_scale', 'local_stat_shift',
                             'token_gate_bias')
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                if not n.startswith(pretrained_prefixes):
                    if any(ip in n for ip in itgca_param_names):
                        continue  # Preserve ITGCA-specific initialization
                    p.data.normal_(0, 0.02)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    if args.dist_train:
        # Multiprocessing distributed mode.
        print("Multiprocessing distributed mode.")
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        print("Single GPU mode.")
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        print("CPU mode.")
        worker(None, None, args, model)


class Trainer(object):
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps

        self.output_model_path = args.output_model_path

        self.start_time = time.time()
        self.total_loss = 0.0

        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size

    def forward_propagation(self, batch, model):
        raise NotImplementedError

    def report_and_reset_stats(self):
        raise NotImplementedError

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        self.model = model
        model.train()
        loader_iter = iter(loader)

        # Native PyTorch AMP
        use_amp = getattr(args, 'fp16', False) and gpu_id is not None
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        while True:
            if self.current_step == self.total_steps + 1:
                break
            batch = list(next(loader_iter))
            self.seq_length = batch[0].size(1)
            if gpu_id is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.forward_propagation(batch, model)

            scaler.scale(loss).backward()

            if self.current_step % self.accumulation_steps == 0:
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm
                    )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()

            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                self.report_and_reset_stats()
                self.start_time = time.time()

            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                save_model(model, self.output_model_path + "-" + str(self.current_step))

            self.current_step += 1



class RawPacketMlmTrainer(Trainer):
    """Trainer for Raw Packet modality - MLM only"""

    def __init__(self, args):
        super(RawPacketMlmTrainer, self).__init__(args)
        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_mlm, packet_ids, directions = batch
        loss_mlm, correct_mlm, denominator = model(src, tgt_mlm, packet_ids, directions)

        self.total_loss += loss_mlm.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_denominator += denominator.item()

        loss = loss_mlm / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print("| {:8d}/{:8d} steps"
              "| {:3.3f} s"
              "| {:8.2f} tokens/s"
              "| loss_mlm: {:7.2f}"
              "| acc_mlm: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            (time.time() - self.start_time),
            done_tokens / (time.time() - self.start_time),
            self.total_loss_mlm / self.report_steps,
            self.total_correct_mlm / self.total_denominator))

        self.total_loss = 0.0
        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0


class PacketSizeMlmTrainer(Trainer):
    """
    Trainer for Packet Size modality with Temporal Information - Dual MLM (Size + Temporal)

    Size MLM: Uses CrossEntropy loss (hard labels)
    Temporal MLM: Uses soft-label KL divergence loss (Gaussian soft labels)
                  Reports both exact and range (±sigma) accuracy
    """

    def __init__(self, args):
        super(PacketSizeMlmTrainer, self).__init__(args)
        # Size MLM stats
        self.total_loss_mlm_size = 0.0
        self.total_correct_mlm_size = 0.0
        self.total_denominator_size = 0.0
        # Temporal MLM stats (with both exact and range accuracy)
        self.total_loss_mlm_temporal = 0.0
        self.total_correct_mlm_temporal_exact = 0.0
        self.total_correct_mlm_temporal_range = 0.0
        self.total_denominator_temporal = 0.0
        # Loss weights
        self.lambda_mlm_size = getattr(args, 'lambda_mlm_size', 1.0)
        self.lambda_mlm_temporal = getattr(args, 'lambda_mlm_temporal', 1.0)

    def forward_propagation(self, batch, model):
        # Check if batch has temporal info (4 tensors) or legacy format (2 tensors)
        if len(batch) == 4:
            # New format with temporal: (src_size, src_iat, tgt_mlm_size, tgt_mlm_temporal)
            src_size, src_iat, tgt_mlm_size, tgt_mlm_temporal = batch

            # Model returns 7 values now (with correct_temporal_range)
            (loss_size, correct_size, denominator_size,
             loss_temporal, correct_temporal_exact, correct_temporal_range, denominator_temporal) = model(
                src_size, src_iat, tgt_mlm_size, tgt_mlm_temporal
            )

            # Weighted combined loss
            loss_mlm = self.lambda_mlm_size * loss_size + self.lambda_mlm_temporal * loss_temporal

            self.total_loss += loss_mlm.item()
            self.total_loss_mlm_size += loss_size.item()
            self.total_loss_mlm_temporal += loss_temporal.item()
            self.total_correct_mlm_size += correct_size.item()
            self.total_correct_mlm_temporal_exact += correct_temporal_exact.item()
            self.total_correct_mlm_temporal_range += correct_temporal_range.item()
            self.total_denominator_size += denominator_size.item()
            self.total_denominator_temporal += denominator_temporal.item()

        else:
            # Legacy format (2 tensors): (src, tgt_mlm) - single MLM
            src, tgt_mlm = batch
            loss_mlm, correct_mlm, denominator = model(src, tgt_mlm)

            self.total_loss += loss_mlm.item()
            self.total_loss_mlm_size += loss_mlm.item()
            self.total_correct_mlm_size += correct_mlm.item()
            self.total_denominator_size += denominator.item()
            loss_mlm = loss_mlm  # Keep for compatibility

        loss = loss_mlm / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        # Report Size and Temporal MLM stats
        # Temporal reports both exact accuracy (for reference) and range accuracy (primary metric)
        print("| {:8d}/{:8d} steps"
              "| {:3.3f} s"
              "| {:8.2f} tokens/s"
              "| loss_sz: {:5.2f}"
              "| acc_sz: {:5.3f}"
              "| loss_tp: {:5.2f}"
              "| acc_tp_ex: {:5.3f}"
              "| acc_tp_rg: {:5.3f}".format(
            self.current_step,
            self.total_steps,
            (time.time() - self.start_time),
            done_tokens / (time.time() - self.start_time),
            self.total_loss_mlm_size / self.report_steps,
            self.total_correct_mlm_size / (self.total_denominator_size + 1e-6),
            self.total_loss_mlm_temporal / self.report_steps,
            self.total_correct_mlm_temporal_exact / (self.total_denominator_temporal + 1e-6),
            self.total_correct_mlm_temporal_range / (self.total_denominator_temporal + 1e-6)))

        self.total_loss = 0.0
        self.total_loss_mlm_size = 0.0
        self.total_loss_mlm_temporal = 0.0
        self.total_correct_mlm_size = 0.0
        self.total_correct_mlm_temporal_exact = 0.0
        self.total_correct_mlm_temporal_range = 0.0
        self.total_denominator_size = 0.0
        self.total_denominator_temporal = 0.0


class MultiModalTrainer(Trainer):
    """
    Trainer for Multi-Modal Pretraining (Stage 2) - ALBEF-style with Masked Reconstruction

    Tasks: ITC + ITM + Masked Reconstruction (Size + Temporal)

    Features:
    - Momentum distillation with EMA update
    - Feature queues for contrastive learning
    - Hard negative mining for ITM
    - Masked Reconstruction with soft-label KL for Temporal
    - Full parameter training with differential LR
    """

    def __init__(self, args):
        super(MultiModalTrainer, self).__init__(args)
        self.args = args

        # Loss tracking
        self.total_loss_itc = 0.0
        self.total_loss_itm = 0.0
        self.total_loss_recon_size = 0.0
        self.total_loss_recon_temporal = 0.0

        # Accuracy tracking
        self.total_acc_itm = 0.0
        self.total_correct_recon_size = 0.0
        self.total_denominator_recon_size = 0.0
        self.total_correct_recon_temporal_exact = 0.0
        self.total_correct_recon_temporal_range = 0.0
        self.total_denominator_recon_temporal = 0.0

        # Loss weights
        self.lambda_itc = getattr(args, 'lambda_itc', 1.0)
        self.lambda_itm = getattr(args, 'lambda_itm', 1.0)
        self.lambda_recon_size = getattr(args, 'lambda_recon_size', 1.0)
        self.lambda_recon_temporal = getattr(args, 'lambda_recon_temporal', 1.0)

        # Count batches for averaging
        self.batch_count = 0

    def forward_propagation(self, batch, model):
        """
        Forward pass for ALBEF-style multimodal pretraining with Masked Reconstruction

        Batch format (9 tensors):
            (raw_src, raw_packet_ids, raw_directions,
             size_src_clean, iat_src_clean, size_src_masked, iat_src_masked,
             tgt_mlm_size, tgt_mlm_temporal)

        Design: ITC/ITM use clean Size inputs; Masked Reconstruction uses masked inputs.
        Local entropy for ITGCA is computed on-the-fly in model forward (vectorized GPU).
        """
        (raw_src, raw_packet_ids, raw_directions,
         size_src_clean, iat_src_clean, size_src_masked, iat_src_masked,
         tgt_mlm_size, tgt_mlm_temporal) = batch

        # Forward through model (must call through DDP wrapper for gradient synchronization)
        loss_dict = model(
            raw_src, raw_packet_ids, raw_directions,
            size_src_clean, iat_src_clean,
            size_src_masked, iat_src_masked,
            tgt_mlm_size, tgt_mlm_temporal
        )

        # Extract losses
        itc_loss = loss_dict['itc_loss']
        itm_loss = loss_dict['itm_loss']
        recon_size_loss = loss_dict['recon_size_loss']
        recon_temporal_loss = loss_dict['recon_temporal_loss']

        # Combined loss
        loss = (self.lambda_itc * itc_loss +
                self.lambda_itm * itm_loss +
                self.lambda_recon_size * recon_size_loss +
                self.lambda_recon_temporal * recon_temporal_loss)

        # Update statistics
        self.total_loss += loss.item()
        self.total_loss_itc += itc_loss.item()
        self.total_loss_itm += itm_loss.item()
        self.total_loss_recon_size += recon_size_loss.item()
        self.total_loss_recon_temporal += recon_temporal_loss.item()

        self.total_acc_itm += loss_dict['itm_acc'].item()
        self.total_correct_recon_size += loss_dict['recon_size_correct'].item()
        self.total_denominator_recon_size += loss_dict['recon_size_denom'].item()
        self.total_correct_recon_temporal_exact += loss_dict['recon_temporal_correct_exact'].item()
        self.total_correct_recon_temporal_range += loss_dict['recon_temporal_correct_range'].item()
        self.total_denominator_recon_temporal += loss_dict['recon_temporal_denom'].item()

        self.batch_count += 1

        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        n = self.batch_count if self.batch_count > 0 else 1

        avg_loss = self.total_loss / n
        avg_loss_itc = self.total_loss_itc / n
        avg_loss_itm = self.total_loss_itm / n
        avg_loss_recon_sz = self.total_loss_recon_size / n
        avg_loss_recon_tp = self.total_loss_recon_temporal / n

        avg_acc_itm = self.total_acc_itm / n
        acc_recon_sz = self.total_correct_recon_size / self.total_denominator_recon_size if self.total_denominator_recon_size > 0 else 0.0
        acc_recon_tp_ex = self.total_correct_recon_temporal_exact / self.total_denominator_recon_temporal if self.total_denominator_recon_temporal > 0 else 0.0
        acc_recon_tp_rg = self.total_correct_recon_temporal_range / self.total_denominator_recon_temporal if self.total_denominator_recon_temporal > 0 else 0.0

        # Build log line
        log_parts = [
            "| {:8d}/{:8d} steps".format(self.current_step, self.total_steps),
            " | {:3.1f}s".format(time.time() - self.start_time),
            " | loss {:5.2f}".format(avg_loss),
            " | itc: {:4.2f}".format(avg_loss_itc),
            " | itm: {:4.2f}".format(avg_loss_itm),
            " | rc_sz: {:4.2f}".format(avg_loss_recon_sz),
            " | rc_tp: {:4.2f}".format(avg_loss_recon_tp),
            " | itm_acc: {:4.2f}".format(avg_acc_itm),
            " | sz_acc: {:4.2f}".format(acc_recon_sz),
            " | tp_ex: {:4.2f}".format(acc_recon_tp_ex),
            " | tp_rg: {:4.2f}".format(acc_recon_tp_rg),
        ]
        print("".join(log_parts))

        # Print ITGCA calibration parameters (gate health monitoring)
        if getattr(self.args, 'use_itgca', False):
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_ref, 'fusion'):
                for i, layer in enumerate(model_ref.fusion.fusion_layers):
                    if hasattr(layer, 'gate_size') and hasattr(layer.gate_size, 'stat_scale'):
                        g = layer.gate_size
                        print(f"  ITGCA L{i}: scale={g.stat_scale.item():.3f} "
                              f"shift={g.stat_shift.item():.3f} "
                              f"alpha={g.alpha_modality.item():.3f} "
                              f"beta={torch.sigmoid(g.alpha_modality).item():.3f}")
                        break  

        # Reset statistics
        self.total_loss = 0.0
        self.total_loss_itc = 0.0
        self.total_loss_itm = 0.0
        self.total_loss_recon_size = 0.0
        self.total_loss_recon_temporal = 0.0
        self.total_acc_itm = 0.0
        self.total_correct_recon_size = 0.0
        self.total_denominator_recon_size = 0.0
        self.total_correct_recon_temporal_exact = 0.0
        self.total_correct_recon_temporal_range = 0.0
        self.total_denominator_recon_temporal = 0.0
        self.batch_count = 0


str2trainer = {
    "raw_packet": RawPacketMlmTrainer,
    "packet_size": PacketSizeMlmTrainer,
    "multimodal": MultiModalTrainer
}


def worker(proc_id, gpu_ranks, args, model):    
    """
    Worker function for training

    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)
    
    print("Starting worker function")

    if args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    print("train loader constructing...")
    if args.dist_train:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, rank, args.world_size, True)
    else:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    # Build optimizer.
    print("Building optimizer...")
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]

    # Multimodal: Differential learning rate (ALBEF-style)
    if args.target == "multimodal":
        encoder_lr_ratio = getattr(args, 'encoder_lr_ratio', 0.1)
        encoder_lr = args.learning_rate * encoder_lr_ratio

        print(f"  Encoder lr: {encoder_lr:.2e} (ratio={encoder_lr_ratio})")
        print(f"  Fusion/Target lr: {args.learning_rate:.2e}")

        # Group parameters: encoder vs non-encoder (fusion + target)
        # Note: Momentum encoders are not trained (requires_grad=False)
        encoder_params_decay = []
        encoder_params_no_decay = []
        other_params_decay = []
        other_params_no_decay = []
        calibration_params = []

        # ITGCA calibration parameter names — need higher LR, no weight decay
        calibration_names = ('stat_scale', 'stat_shift',
                             'local_stat_scale', 'local_stat_shift')

        for n, p in param_optimizer:
            if not p.requires_grad:
                continue  # Skip momentum encoder parameters

            # Check if this is a main encoder parameter (not momentum)
            is_main_encoder = (("encoder_raw" in n or "encoder_size" in n or
                               "embedding_raw" in n or "embedding_size" in n) and
                              "_m" not in n)  # Exclude momentum encoders
            is_no_decay = any(nd in n for nd in no_decay)
            is_calibration = any(cn in n for cn in calibration_names)

            if is_calibration:
                calibration_params.append(p)
            elif is_main_encoder:
                if is_no_decay:
                    encoder_params_no_decay.append(p)
                else:
                    encoder_params_decay.append(p)
            else:
                if is_no_decay:
                    other_params_no_decay.append(p)
                else:
                    other_params_decay.append(p)

        calibration_lr = args.learning_rate * 10  # 10× higher LR for calibration
        optimizer_grouped_parameters = [
            {"params": encoder_params_decay, "lr": encoder_lr, "weight_decay": 0.01},
            {"params": encoder_params_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
            {"params": other_params_decay, "lr": args.learning_rate, "weight_decay": 0.01},
            {"params": other_params_no_decay, "lr": args.learning_rate, "weight_decay": 0.0},
            {"params": calibration_params, "lr": calibration_lr, "weight_decay": 0.0},
        ]

        print(f"  Encoder params (lr={encoder_lr:.2e}): "
              f"{len(encoder_params_decay)} decay, {len(encoder_params_no_decay)} no_decay")
        print(f"  Other params (lr={args.learning_rate:.2e}): "
              f"{len(other_params_decay)} decay, {len(other_params_no_decay)} no_decay")
        print(f"  Calibration params (lr={calibration_lr:.2e}, no wd): {len(calibration_params)}")
    else:
        # Non-multimodal: all parameters use same learning rate
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

    optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)

    # Create scheduler (same for both phases)
    scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps*args.warmup, args.total_steps)

    debug = True
    if args.dist_train:
        print("Initialize multiprocessing distributed training environment...")
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        if debug:
            print("Debug: dist init finished")
            torch.cuda.synchronize()
            print(f"Rank{rank}: CUDA synchronized")
            dist.barrier()
            print(f"Rank{rank}: Barrier passed")
            
        # broadcast_buffers=False for multimodal: let each rank maintain its own
        # feature queue independently, avoiding all_gather hangs while preserving
        # queue diversity (each rank accumulates features from its own data stream).
        broadcast_bufs = (args.target != "multimodal")
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=False,
                                        broadcast_buffers=broadcast_bufs)
        print("Worker %d is training ... " % rank)
    else:
        print("Worker is training ...")

    trainer = str2trainer[args.target](args)
    trainer.train(args, gpu_id, rank, train_loader, model, optimizer, scheduler)

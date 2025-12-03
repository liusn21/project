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
from tqdm import tqdm

def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    print("Load vocabulary.")
    if args.target == "multimodal":
        # Multi-modal requires two vocabularies
        print("Loading multi-modal vocabularies...")
        vocab_raw = Vocab()
        vocab_raw.load(args.vocab_path_raw)
        args.vocab_raw = vocab_raw.w2i

        vocab_size = Vocab()
        vocab_size.load(args.vocab_path_size)
        args.vocab_size = vocab_size.w2i

        # Create tokenizers for both modalities
        args.tokenizer_raw = str2tokenizer[args.tokenizer](args)
        args.tokenizer_raw.vocab = vocab_raw.w2i

        args.tokenizer_size = str2tokenizer[args.tokenizer](args)
        args.tokenizer_size.vocab = vocab_size.w2i

        # Set main vocab/tokenizer for compatibility
        args.vocab = vocab_raw.w2i
        args.tokenizer = args.tokenizer_raw

    elif args.spm_model_path:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                              "pip install sentencepiece")
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.spm_model_path)
        args.vocab = {sp_model.IdToPiece(i): i for i
                      in range(sp_model.GetPieceSize())}
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        if args.target == "seq2seq":
            tgt_sp_model = spm.SentencePieceProcessor()
            tgt_sp_model.Load(args.tgt_spm_model_path)
            args.tgt_vocab = {tgt_sp_model.IdToPiece(i): i for i
                              in range(tgt_sp_model.GetPieceSize())}
    else:
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        args.vocab = args.tokenizer.vocab
        if args.target == "seq2seq":
            tgt_vocab = Vocab()
            tgt_vocab.load(args.tgt_vocab_path)
            args.tgt_vocab = tgt_vocab.w2i

    # Build model.
    print("Build model.")
    model = build_model(args)
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model = load_model(model, args.pretrained_model_path) 
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
        model.train()
        loader_iter = iter(loader)
        while True:
            if self.current_step == self.total_steps + 1:
                break
            batch = list(next(loader_iter))
            self.seq_length = batch[0].size(1)
            if gpu_id is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id)

            loss = self.forward_propagation(batch, model)

            if args.fp16:
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.current_step % self.accumulation_steps == 0:
                # Gradient clipping to prevent gradient explosion
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            args.amp.master_params(optimizer), args.clip_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.clip_grad_norm
                        )

                optimizer.step()
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
        

class BertTrainer(Trainer):
    def __init__(self, args):
        super(BertTrainer, self).__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0
        self.load_balance_alpha = args.moebert_load_balance
        self.is_moe = args.is_moe

    def forward_propagation(self, batch, model):
        debug_mode = False
        if debug_mode:
            print("In function forward_propagation(self, batch, model):")
            print("type of batch:",type(batch))
            print("type of the content of batch:",[type(elem) for elem in batch])
        if len(batch)==5:
            src, tgt_mlm, tgt_sp, seg, proto = batch
            loss_info = model(src, (tgt_mlm, tgt_sp), seg, proto)
        else:
            src, tgt_mlm, tgt_sp, seg = batch
            loss_info = model(src, (tgt_mlm, tgt_sp), seg)
        
        if self.is_moe:
            loss_mlm, loss_sp, correct_mlm, correct_sp, denominator, gate_loss = loss_info
        else:
            loss_mlm, loss_sp, correct_mlm, correct_sp, denominator = loss_info
            gate_loss = 0.0
        loss = loss_mlm/10 + loss_sp + self.load_balance_alpha * gate_loss
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print("| {:8d}/{:8d} steps"
              "| {:3.3f} s"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_mlm: {:3.3f}"
              "| loss_sp: {:3.3f}"
              "| acc_mlm: {:3.3f}"
              "| acc_sp: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            (time.time() - self.start_time),
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_loss_mlm / self.report_steps,
            self.total_loss_sp / self.report_steps,
            self.total_correct_mlm / self.total_denominator,
            self.total_correct_sp / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_sp, self.total_instances = 0.0, 0.0

class RawPacketMlmTrainer(Trainer):
    """
    Trainer for Raw Packet modality - MLM only

    New implementation using:
    - token embedding
    - position embedding
    - packet embedding (0-7 for packets, 8 for special/padding)
    - direction embedding (0=downlink, 1=neutral, 2=uplink)
    """

    def __init__(self, args):
        super(RawPacketMlmTrainer, self).__init__(args)
        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        """
        Batch format from RawPacketDataLoader:
        (src, tgt_mlm, packet_ids, directions)

        src: [batch, seq_len] - token IDs
        tgt_mlm: [batch, seq_len] - MLM targets
        packet_ids: [batch, seq_len] - packet indices (0-7 for packets, 8 for special/padding)
        directions: [batch, seq_len] - direction indices (0=downlink, 1=neutral, 2=uplink)
        """
        src, tgt_mlm, packet_ids, directions = batch

        # Forward pass with packet_ids and directions
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
    Trainer for Packet Size modality - MLM only

    New implementation using:
    - token embedding (direction already encoded in size tokens)
    - position embedding
    """

    def __init__(self, args):
        super(PacketSizeMlmTrainer, self).__init__(args)
        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        """
        Batch format from PacketSizeDataLoader:
        (src, tgt_mlm)

        src: [batch, seq_len] - size token IDs (direction encoded)
        tgt_mlm: [batch, seq_len] - MLM targets
        """
        src, tgt_mlm = batch

        # Forward pass (no additional inputs needed)
        loss_mlm, correct_mlm, denominator = model(src, tgt_mlm)

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


class MultiModalTrainer(Trainer):
    """
    Trainer for Multi-Modal Pretraining (Stage 2)

    Implements two-phase training:
    - Phase 1 (0-70K steps): Freeze encoders, train fusion only
    - Phase 2 (70K-100K steps): Joint training with differential LR

    Tasks: CMM + CMMP + Balance Loss
    """

    def __init__(self, args):
        super(MultiModalTrainer, self).__init__(args)

        # Loss tracking
        self.total_loss_cmm = 0.0
        self.total_loss_cmmp = 0.0
        self.total_loss_balance = 0.0

        # Accuracy tracking
        self.total_correct_cmm = 0.0
        self.total_instances_cmm = 0.0
        self.total_correct_cmmp = 0.0
        self.total_denominator_cmmp = 0.0

        # Gate weight tracking
        self.total_g_raw = 0.0
        self.total_g_size = 0.0

        # Two-phase training parameters
        self.phase1_steps = args.phase1_steps if hasattr(args, 'phase1_steps') else 70000
        self.balance_loss_alpha = args.balance_loss_alpha if hasattr(args, 'balance_loss_alpha') else 0.1

        # CMM temperature parameter (可配置，默认0.07)
        self.cmm_temperature = getattr(args, 'cmm_temperature', 0.07)

        # Phase transition flag
        self.phase_transitioned = False

    def forward_propagation(self, batch, model):
        """
        NEW ARCHITECTURE v2 (Both CMM and CMMP after Fusion)

        Batch format from MultiModalDataLoader (all positive samples):
        (raw_src, raw_packet_ids, raw_directions, size_src, tgt_cmmp_size)

        Key changes (v2):
        1. Forward encoders (all samples are positive pairs)
        2. Fusion on all positive samples
        3. CMMP task AFTER Fusion (MLM on fused features)
        4. CMM task AFTER Fusion (ITM with dynamic negative sampling)

        This ensures:
        - Phase 1 (freeze encoder): All losses can train fusion/target ✅
        - CMM uses fused features (higher quality) ✅
        - CMMP uses fused features (semantically correct) ✅
        - Negative samples: Dynamic hard negative mining in batch ✅

        Flow:
        1. Forward encoders (only Size is masked, Raw is not masked)
        2. Fusion on positive samples
        3. CMMP task (MLM) using fused features
        4. CMM task (ITM) using fused [CLS] with dynamic negative sampling
        """
        raw_src, raw_packet_ids, raw_directions, size_src, tgt_cmmp_size = batch
        batch_size = raw_src.size(0)

        # ===== Step 1: Forward Encoders =====
        # Get embeddings
        if hasattr(model, 'module'):  # DistributedDataParallel
            raw_emb = model.module.embedding_raw(raw_src, raw_packet_ids, raw_directions)
            size_emb = model.module.embedding_size(size_src)
        else:
            raw_emb = model.embedding_raw(raw_src, raw_packet_ids, raw_directions)
            size_emb = model.embedding_size(size_src)

        # Create attention masks
        from uer.utils.constants import PAD_ID
        raw_seg = (raw_src != PAD_ID).long()
        size_seg = (size_src != PAD_ID).long()

        # Forward encoders (only Size is masked, Raw is not masked)
        if hasattr(model, 'module'):
            raw_output = model.module.encoder_raw(raw_emb, raw_seg)
            size_output = model.module.encoder_size(size_emb, size_seg)
        else:
            raw_output = model.encoder_raw(raw_emb, raw_seg)
            size_output = model.encoder_size(size_emb, size_seg)

        # ===== Step 2: Fusion (All positive samples) =====
        if hasattr(model, 'module'):
            fusion = model.module.fusion
            target = model.module.target
        else:
            fusion = model.fusion
            target = model.target

        # Fusion processes all positive pairs
        # 传递seg信息以正确mask padding positions
        raw_fused, size_fused, (g_raw, g_size) = fusion(
            raw_output, size_output,
            raw_seg=raw_seg,
            size_seg=size_seg,
            return_gate_weights=True
        )

        # ===== Step 3: CMMP Task (AFTER Fusion, MLM on positive samples) =====
        cmmp_loss, cmmp_correct, cmmp_denominator = target.forward_cmmp_only(
            size_fused, tgt_cmmp_size
        )

        # ===== Step 4: CMM Task (AFTER Fusion, ITM with dynamic negatives) =====
        # 使用fused [CLS] features，动态构建负样本
        cmm_loss, cmm_correct = target.forward_cmm_itm(
            raw_fused, size_fused, temperature=self.cmm_temperature
        )

        # ===== Step 5: Balance Loss =====
        from uer.layers.multimodal_fusion import compute_balance_loss
        balance_loss = compute_balance_loss(g_raw, g_size, target_ratio=0.5)

        # ===== Step 6: Combined Loss =====
        loss = cmm_loss + cmmp_loss + self.balance_loss_alpha * balance_loss

        # ===== Update Statistics =====
        self.total_loss += loss.item()
        self.total_loss_cmm += cmm_loss.item()
        self.total_loss_cmmp += cmmp_loss.item()
        self.total_loss_balance += balance_loss.item()

        # CMM: batch_size samples (random 50% positive + 50% negative)
        self.total_correct_cmm += cmm_correct.item()
        self.total_instances_cmm += batch_size

        # CMMP: only on positive samples
        self.total_correct_cmmp += cmmp_correct.item()
        self.total_denominator_cmmp += cmmp_denominator.item()

        self.total_g_raw += g_raw.mean().item()
        self.total_g_size += g_size.mean().item()

        # Scale loss for gradient accumulation
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        # Compute averages
        avg_loss = self.total_loss / self.report_steps
        avg_loss_cmm = self.total_loss_cmm / self.report_steps
        avg_loss_cmmp = self.total_loss_cmmp / self.report_steps
        avg_loss_balance = self.total_loss_balance / self.report_steps

        acc_cmm = self.total_correct_cmm / self.total_instances_cmm if self.total_instances_cmm > 0 else 0.0
        acc_cmmp = self.total_correct_cmmp / self.total_denominator_cmmp if self.total_denominator_cmmp > 0 else 0.0

        avg_g_raw = self.total_g_raw / self.report_steps
        avg_g_size = self.total_g_size / self.report_steps

        # Determine current phase
        phase = "Phase1" if self.current_step <= self.phase1_steps else "Phase2"

        print("| {:8d}/{:8d} steps"
              " | {} |"
              " {:3.3f} s"
              " | {:8.2f} tokens/s"
              " | loss {:7.2f}"
              " | cmm: {:3.3f}"
              " | cmmp: {:3.3f}"
              " | bal: {:3.3f}"
              " | acc_cmm: {:3.3f}"
              " | acc_cmmp: {:3.3f}"
              " | g_raw: {:3.3f}"
              " | g_size: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            phase,
            (time.time() - self.start_time),
            done_tokens / (time.time() - self.start_time),
            avg_loss,
            avg_loss_cmm,
            avg_loss_cmmp,
            avg_loss_balance,
            acc_cmm,
            acc_cmmp,
            avg_g_raw,
            avg_g_size
        ))

        # Reset statistics
        self.total_loss = 0.0
        self.total_loss_cmm = 0.0
        self.total_loss_cmmp = 0.0
        self.total_loss_balance = 0.0

        self.total_correct_cmm = 0.0
        self.total_instances_cmm = 0.0
        self.total_correct_cmmp = 0.0
        self.total_denominator_cmmp = 0.0

        self.total_g_raw = 0.0
        self.total_g_size = 0.0

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        """
        Training loop with two-phase logic
        """
        model.train()
        loader_iter = iter(loader)

        while True:
            if self.current_step == self.total_steps + 1:
                break

            # Phase transition: Unfreeze encoders at phase1_steps
            if self.current_step == self.phase1_steps + 1 and not self.phase_transitioned:
                print("TRANSITIONING TO PHASE 2: Unfreezing encoders")

                if hasattr(model, 'module'):  # DistributedDataParallel
                    model.module.unfreeze_encoders()
                else:
                    model.unfreeze_encoders()

                self.phase_transitioned = True

            batch = list(next(loader_iter))
            self.seq_length = batch[0].size(1)  # Use raw_src length

            if gpu_id is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id)

            loss = self.forward_propagation(batch, model)

            if args.fp16:
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.current_step % self.accumulation_steps == 0:
                # Gradient clipping
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            args.amp.master_params(optimizer), args.clip_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.clip_grad_norm
                        )

                optimizer.step()
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


str2trainer = {
    "bertflow": BertTrainer,
    "raw_packet": RawPacketMlmTrainer,
    "packet_size": PacketSizeMlmTrainer,
    "multimodal": MultiModalTrainer
}

def worker(proc_id, gpu_ranks, args, model):    
    """
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
    print("build optomizer...")
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps*args.warmup, args.total_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if args.dist_train:
        print("Initialize multiprocessing distributed training environment...")
        # Initialize multiprocessing distributed training environment.
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        print("Worker %d is training ... " % rank)
    else:
        print("Worker is training ...")

    trainer = str2trainer[args.target](args)
    trainer.train(args, gpu_id, rank, train_loader, model, optimizer, scheduler)

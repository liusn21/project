import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.layer_norm import LayerNorm
from uer.utils import *


def soft_label_kl_loss(logits, target_idx, num_classes, sigma=10):
    """
    Compute KL divergence loss with soft Gaussian labels.

    Instead of hard one-hot labels, use Gaussian distribution centered at target.
    This allows "close" predictions to have lower loss than "far" predictions,
    which is appropriate for ordinal/continuous values like temporal tokens.

    Args:
        logits: [N, num_classes] - raw model outputs (before softmax)
        target_idx: [N] - ground truth token indices
        num_classes: int - vocabulary size
        sigma: float - standard deviation of Gaussian, controls tolerance range
                      (±3σ covers ~99.7% probability mass)

    Returns:
        loss: scalar - KL divergence loss (equivalent to cross-entropy with soft labels)
    """
    device = logits.device

    # Generate soft target distribution: Gaussian centered at target_idx
    # x: [num_classes]
    x = torch.arange(num_classes, device=device, dtype=torch.float32)

    # target_idx: [N] -> [N, 1], x: [num_classes] -> [1, num_classes]
    # target_soft: [N, num_classes]
    target_soft = torch.exp(
        -(x.unsqueeze(0) - target_idx.unsqueeze(1).float()) ** 2 / (2 * sigma ** 2)
    )
    # Normalize to probability distribution
    target_soft = target_soft / target_soft.sum(dim=1, keepdim=True)

    # KL(target || pred) = sum(target * log(target)) - sum(target * log(pred))
    # First term is constant (entropy of target), so we minimize:
    # -sum(target * log(pred)) = CrossEntropy with soft labels
    pred_log_prob = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(target_soft * pred_log_prob, dim=-1).mean()

    return loss


class MlmTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(MlmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.act = str2act[args.hidden_act]

        if self.factorized_embedding_parameterization:
            self.mlm_linear_1 = nn.Linear(args.hidden_size, args.emb_size)
            self.layer_norm = LayerNorm(args.emb_size)
            self.mlm_linear_2 = nn.Linear(args.emb_size, self.vocab_size)
        else:
            self.mlm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.layer_norm = LayerNorm(args.hidden_size)
            self.mlm_linear_2 = nn.Linear(args.hidden_size, self.vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)

        self.criterion = nn.NLLLoss()
        #self.criterion = nn.CrossEntropyLoss()

    def mlm(self, memory_bank, tgt_mlm):
        # Masked language modeling (MLM) with full softmax prediction.
        output_mlm = self.act(self.mlm_linear_1(memory_bank))
        output_mlm = self.layer_norm(output_mlm)
        if self.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[tgt_mlm > 0, :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        output_mlm = self.mlm_linear_2(output_mlm)
        output_mlm = self.softmax(output_mlm)
        denominator = torch.tensor(output_mlm.size(0) + 1e-6)
        if output_mlm.size(0) == 0:
            correct_mlm = torch.tensor(0.0)
        else:
            correct_mlm = torch.sum((output_mlm.argmax(dim=-1).eq(tgt_mlm)).float())
        loss_mlm = self.criterion(output_mlm, tgt_mlm)
        return loss_mlm, correct_mlm, denominator
    
    def mlm2(self, memory_bank, tgt_mlm):
        # Masked language modeling (MLM) with full softmax prediction.
        output_mlm = self.act(self.mlm_linear_1(memory_bank))
        output_mlm = self.layer_norm(output_mlm)
        if self.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[tgt_mlm > 0, :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        output_mlm = self.mlm_linear_2(output_mlm)
        output_mlm = self.softmax(output_mlm)
        denominator = torch.tensor(output_mlm.size(0) + 1e-6)
        if output_mlm.size(0) == 0:
            correct_mlm = torch.tensor(0.0)
        else:
            correct_mlm = torch.sum((output_mlm.argmax(dim=-1).eq(tgt_mlm)).float())
        loss_mlm = self.criterion(output_mlm, tgt_mlm)
        return loss_mlm, output_mlm, tgt_mlm

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Masked language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language model (MLM).
        loss, correct, denominator = self.mlm(memory_bank, tgt)

        return loss, correct, denominator


class DualMlmTarget(nn.Module):
    """
    Dual MLM Target for Packet Size Modality with Temporal Information

    Predicts both:
    1. Size MLM: Masked size tokens (CrossEntropy loss)
    2. Temporal MLM: Masked IAT tokens (Soft-label KL divergence loss)

    Temporal uses soft Gaussian labels because IAT is a continuous value discretized
    into tokens, where nearby values (e.g., 500 vs 501) should have similar loss.

    This enables the model to learn both spatial (size) and temporal (IAT) patterns.
    """

    def __init__(self, args, vocab_size_size, vocab_size_temporal):
        super(DualMlmTarget, self).__init__()
        self.vocab_size_size = vocab_size_size
        self.vocab_size_temporal = vocab_size_temporal
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.act = str2act[args.hidden_act]

        # Temporal soft-label sigma: controls tolerance range (±3σ ≈ 99.7% mass)
        # sigma=10 means predictions within ±30 tokens are considered "close"
        # Now configurable via args.temporal_sigma (default=10 for backward compatibility)
        self.temporal_sigma = getattr(args, 'temporal_sigma', 10)

        # ===== Size MLM Head =====
        if self.factorized_embedding_parameterization:
            self.mlm_size_linear_1 = nn.Linear(args.hidden_size, args.emb_size)
            self.mlm_size_layer_norm = LayerNorm(args.emb_size)
            self.mlm_size_linear_2 = nn.Linear(args.emb_size, vocab_size_size)
        else:
            self.mlm_size_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.mlm_size_layer_norm = LayerNorm(args.hidden_size)
            self.mlm_size_linear_2 = nn.Linear(args.hidden_size, vocab_size_size)

        self.mlm_size_softmax = nn.LogSoftmax(dim=-1)
        self.mlm_size_criterion = nn.NLLLoss()

        # ===== Temporal MLM Head =====
        # Note: Uses soft-label KL loss instead of CrossEntropy
        if self.factorized_embedding_parameterization:
            self.mlm_temporal_linear_1 = nn.Linear(args.hidden_size, args.emb_size)
            self.mlm_temporal_layer_norm = LayerNorm(args.emb_size)
            self.mlm_temporal_linear_2 = nn.Linear(args.emb_size, vocab_size_temporal)
        else:
            self.mlm_temporal_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.mlm_temporal_layer_norm = LayerNorm(args.hidden_size)
            self.mlm_temporal_linear_2 = nn.Linear(args.hidden_size, vocab_size_temporal)

    def mlm_size(self, memory_bank, tgt_mlm_size):
        """
        Size MLM prediction

        Args:
            memory_bank: [batch, seq_len, hidden]
            tgt_mlm_size: [batch, seq_len] - size MLM targets (0=unmasked, >0=masked token id)

        Returns:
            loss, correct, denominator
        """
        output = self.act(self.mlm_size_linear_1(memory_bank))
        output = self.mlm_size_layer_norm(output)

        if self.factorized_embedding_parameterization:
            output = output.contiguous().view(-1, self.emb_size)
        else:
            output = output.contiguous().view(-1, self.hidden_size)

        tgt = tgt_mlm_size.contiguous().view(-1)

        # Only compute loss on masked positions (tgt > 0)
        output = output[tgt > 0, :]
        tgt = tgt[tgt > 0]

        output = self.mlm_size_linear_2(output)
        output = self.mlm_size_softmax(output)

        denominator = torch.tensor(output.size(0) + 1e-6)
        if output.size(0) == 0:
            correct = torch.tensor(0.0)
            loss = torch.tensor(0.0, device=memory_bank.device)
        else:
            correct = torch.sum((output.argmax(dim=-1).eq(tgt)).float())
            loss = self.mlm_size_criterion(output, tgt)

        return loss, correct, denominator

    def mlm_temporal(self, memory_bank, tgt_mlm_temporal):
        """
        Temporal MLM prediction with soft-label KL divergence loss.

        Uses Gaussian soft labels instead of hard one-hot labels, because
        temporal tokens represent discretized continuous values where
        nearby predictions should have lower loss than distant ones.

        Args:
            memory_bank: [batch, seq_len, hidden]
            tgt_mlm_temporal: [batch, seq_len] - temporal MLM targets (0=unmasked, >0=masked token id)

        Returns:
            loss: soft-label KL divergence loss
            correct_exact: number of exact matches (for reference)
            correct_range: number of predictions within ±sigma range
            denominator: number of masked positions
        """
        output = self.act(self.mlm_temporal_linear_1(memory_bank))
        output = self.mlm_temporal_layer_norm(output)

        if self.factorized_embedding_parameterization:
            output = output.contiguous().view(-1, self.emb_size)
        else:
            output = output.contiguous().view(-1, self.hidden_size)

        tgt = tgt_mlm_temporal.contiguous().view(-1)

        # Only compute loss on masked positions (tgt > 0)
        output = output[tgt > 0, :]
        tgt = tgt[tgt > 0]

        denominator = torch.tensor(output.size(0) + 1e-6)

        if output.size(0) == 0:
            correct_exact = torch.tensor(0.0)
            correct_range = torch.tensor(0.0)
            loss = torch.tensor(0.0, device=memory_bank.device)
        else:
            # Get logits (before softmax)
            logits = self.mlm_temporal_linear_2(output)

            # Soft-label KL divergence loss
            loss = soft_label_kl_loss(
                logits, tgt, self.vocab_size_temporal, sigma=self.temporal_sigma
            )

            # Compute accuracy metrics
            pred_idx = logits.argmax(dim=-1)

            # Exact match accuracy (for reference/comparison)
            correct_exact = torch.sum((pred_idx == tgt).float())

            # Range accuracy: prediction within ±sigma of target
            correct_range = torch.sum(((pred_idx - tgt).abs() <= self.temporal_sigma).float())

        return loss, correct_exact, correct_range, denominator

    def forward(self, memory_bank, tgt_mlm_size, tgt_mlm_temporal):
        """
        Args:
            memory_bank: [batch, seq_len, hidden]
            tgt_mlm_size: [batch, seq_len] - size MLM targets
            tgt_mlm_temporal: [batch, seq_len] - temporal MLM targets

        Returns:
            Tuple of:
            - loss_size: Size MLM loss (CrossEntropy)
            - correct_size: Number of correct size predictions
            - denominator_size: Number of masked size positions
            - loss_temporal: Temporal MLM loss (soft-label KL divergence)
            - correct_temporal_exact: Number of exact temporal matches
            - correct_temporal_range: Number of predictions within ±sigma range
            - denominator_temporal: Number of masked temporal positions
        """
        # Size MLM (CrossEntropy)
        loss_size, correct_size, denominator_size = self.mlm_size(memory_bank, tgt_mlm_size)

        # Temporal MLM (soft-label KL divergence)
        loss_temporal, correct_temporal_exact, correct_temporal_range, denominator_temporal = \
            self.mlm_temporal(memory_bank, tgt_mlm_temporal)

        return (loss_size, correct_size, denominator_size,
                loss_temporal, correct_temporal_exact, correct_temporal_range, denominator_temporal)

#!/usr/bin/env python3
"""
Analytic parameter counter for the MM-TrafficBERT Stage-2 (multimodal) model.

Counts are derived directly from the module construction in:
  - uer/layers/embeddings.py        (RawPacketEmbedding, PacketSizeEmbedding)
  - uer/layers/transformer.py       (TransformerLayer: 4H^2 + 2HF + 9H + F)
  - uer/layers/multimodal_fusion.py (BidirectionalFusionLayer + ITGCrossAttentionGate)
  - uer/targets/multimodal_target.py(ITC/ITM/recon heads)
  - uer/models/multimodal_model.py  (momentum copies + queues)

Assumptions matching the default configs (all *_config.json with the store_true
flags left at default):
  - emb_size == hidden_size (E = H), factorized_embedding_parameterization = False
    -> no encoder emb->hidden projection; recon heads use hidden_size.
  - remove_embedding_layernorm = False (embedding LayerNorm present).
  - remove_transformer_bias = False (all Linear have bias).
  - heads_num divides hidden_size -> attention inner_hidden == H.
No torch needed: this is pure arithmetic.
"""

# ---- fixed vocab / sequence dimensions (from models/bert/vocab_*.txt) ----
VOCAB_RAW = 261        # content (raw packet) byte vocab
VOCAB_SIZE = 3006      # behavior (packet-size) token vocab
VOCAB_TEMPORAL = 1005  # IAT temporal vocab
MAX_SEQ = 512          # position-embedding rows (args.max_seq_length)


def encoder_layer(H, F):
    """One TransformerLayer: self-attn(4H^2+4H) + FFN(2HF+F+H) + 2*LN(2H)."""
    return 4 * H * H + 2 * H * F + 9 * H + F


def raw_embedding(H):
    """RawPacketEmbedding: token(261)+pos(512)+packet(9)+dir(3) emb + LN(2)."""
    return (VOCAB_RAW + MAX_SEQ + 9 + 3) * H + 2 * H          # 787H


def size_embedding(H):
    """PacketSizeEmbedding: token(3006)+temporal(1005)+pos(512) emb + LN(2)."""
    return (VOCAB_SIZE + VOCAB_TEMPORAL + MAX_SEQ) * H + 2 * H  # 4525H


def fusion_layer_body(H, F):
    """BidirectionalFusionLayer minus gates: 2 branches x (selfattn+crossattn+FFN+3LN)."""
    return 16 * H * H + 4 * H * F + 30 * H + 2 * F


def fusion_layer_gates(H):
    """ITGCA per fusion layer: gate_raw(1.5H^2+2) + gate_size(1.5H^2+5) + local(2)."""
    return 3 * H * H + 9


def lightweight_mlp_gate(H, bottleneck=64):
    """One stack-shared MLP: Linear(2H,B) + Linear(B,2)."""
    B = min(bottleneck, H)
    return B * (2 * H + 3) + 2


def target_heads(H):
    """ITC(2 x (H^2+H)) + ITM(2H^2+3H+2) + recon_size(H^2+3009H+3006) + recon_temporal(H^2+1008H+1005)."""
    itc = 2 * (H * H + H)
    itm = 2 * H * H + 3 * H + 2
    recon_size = H * H + H + 2 * H + VOCAB_SIZE * H + VOCAB_SIZE
    recon_temporal = H * H + H + 2 * H + VOCAB_TEMPORAL * H + VOCAB_TEMPORAL
    return itc + itm + recon_size + recon_temporal


def itc_proj(H):
    """One itc_proj Linear(H->H) (momentum copies 2 of these)."""
    return H * H + H


def count(name, H, F, L_content, L_behavior, L_fusion, use_itgca=True):
    raw_emb = raw_embedding(H)
    size_emb = size_embedding(H)
    content_enc = L_content * encoder_layer(H, F)
    behavior_enc = L_behavior * encoder_layer(H, F)
    fusion_body = L_fusion * fusion_layer_body(H, F)
    fusion_gates = L_fusion * fusion_layer_gates(H) if use_itgca else 0
    heads = target_heads(H)

    # (A) Inference / deployment model: what forward_inference() actually uses.
    inference = raw_emb + size_emb + content_enc + behavior_enc + fusion_body + fusion_gates
    # (B) Online trainable model: inference + training-only target heads.
    trainable = inference + heads
    # (C) Full nn.Module param count: + frozen momentum copies (EMA, requires_grad=False).
    momentum = raw_emb + size_emb + content_enc + behavior_enc + 2 * itc_proj(H)
    full = trainable + momentum
    # Buffers (NOT parameters): feature queues 2 x [4096, H].
    queues = 2 * 4096 * H

    return {
        "name": name, "H": H, "F": F,
        "L": (L_content, L_behavior, L_fusion),
        "raw_emb": raw_emb, "size_emb": size_emb,
        "content_enc": content_enc, "behavior_enc": behavior_enc,
        "fusion_body": fusion_body, "fusion_gates": fusion_gates,
        "heads": heads, "momentum": momentum, "queues": queues,
        "inference": inference, "trainable": trainable, "full": full,
    }


def fmt(n):
    return f"{n/1e6:7.2f}M"


CONFIGS = [
    # name,            H,    F,   content, behavior, fusion
    ("base (current)", 768, 3072, 12, 6, 6),
    ("medium",         512, 2048,  8, 4, 4),
    ("small",          512, 2048,  4, 2, 2),
    ("base-8L",        768, 3072,  8, 4, 4),
]

if __name__ == "__main__":
    rows = [count(*c) for c in CONFIGS]

    print("=" * 92)
    print("MM-TrafficBERT Stage-2 parameter counts (ITGCA enabled)")
    print("  behavior & fusion depth = half of content depth")
    print("=" * 92)
    hdr = f"{'config':<16}{'H':>5}{'layers(c/b/f)':>15}{'(A) infer':>12}{'(B) train':>12}{'(C) +mom':>12}"
    print(hdr)
    print("-" * 92)
    for r in rows:
        lc, lb, lf = r["L"]
        print(f"{r['name']:<16}{r['H']:>5}{f'{lc}/{lb}/{lf}':>15}"
              f"{fmt(r['inference']):>12}{fmt(r['trainable']):>12}{fmt(r['full']):>12}")
    print("-" * 92)
    print("(A) inference  = embeddings + 2 encoders + fusion(+ITGCA gates)  <- matches paper '~255M'")
    print("(B) trainable  = (A) + ITC/ITM/recon target heads (training only)")
    print("(C) +momentum  = (B) + frozen EMA copies of encoders+embeddings+itc_proj")
    print()

    # Per-component breakdown
    print("=" * 92)
    print("Per-component breakdown (millions of params)")
    print("=" * 92)
    comp_hdr = (f"{'config':<16}{'raw_emb':>9}{'size_emb':>9}{'content':>9}"
                f"{'behav':>9}{'fus_body':>9}{'gates':>8}{'heads':>8}")
    print(comp_hdr)
    print("-" * 92)
    for r in rows:
        print(f"{r['name']:<16}"
              f"{r['raw_emb']/1e6:>9.2f}{r['size_emb']/1e6:>9.2f}{r['content_enc']/1e6:>9.2f}"
              f"{r['behavior_enc']/1e6:>9.2f}{r['fusion_body']/1e6:>9.2f}"
              f"{r['fusion_gates']/1e6:>8.2f}{r['heads']/1e6:>8.2f}")
    print("-" * 92)
    print("Note: feature queues (2 x [4096,H]) are buffers, not params:")
    for r in rows:
        print(f"   {r['name']:<16} queue buffer = {r['queues']/1e6:.2f}M (excluded from all counts above)")

    print()
    print("Gate-only comparison (base H=768, 6 fusion layers):")
    print(f"   Full ITGCA       = {6 * fusion_layer_gates(768)/1e6:.2f}M parameters")
    print(f"   Shared MLP gate  = {lightweight_mlp_gate(768)/1e6:.5f}M parameters")

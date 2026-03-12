"""
Multi-Modal Fine-tuning Data Utilities

PCAP processing and dataset creation for multi-modal traffic classification.
Extracts both Raw Packet (bigram) and Packet Size features from pcap files.

Input format: datasets/label_name/flow_file.pcap
  - Each pcap file represents one bidirectional flow
  - Filename format: {protocol}_{src_ip}_{src_port}_{dst_ip}_{dst_port}.pcap

Output: train/val/test datasets with both modalities
"""

import os
import sys
sys.path.append(os.getcwd())

import random
import numpy as np
from scapy.all import rdpcap, PcapReader
from scapy.layers.inet import IP, TCP, UDP
from tqdm import tqdm
from collections import defaultdict
import pickle
import argparse
import multiprocessing as mp

from uer.utils.vocab import Vocab
from uer.utils.tokenizers import BertTokenizer
from uer.utils.constants import CLS_TOKEN, SEP_TOKEN, PAD_ID

# Pre-computed hex lookup tables (module-level constants)
_BIGRAM_TABLE = [f"{i:02x}{j:02x}" for i in range(256) for j in range(256)]
_BYTE_TABLE = [f"{i:02x}" for i in range(256)]


class MultiModalFlowExtractor:
    """Extract multi-modal features from a single PCAP file"""

    def __init__(self, bytes_per_packet=64, max_raw_packets=8, max_size_packets=256, raw_token_type='bigram'):
        self.bytes_per_packet = bytes_per_packet
        self.max_raw_packets = max_raw_packets
        self.max_size_packets = max_size_packets
        self.raw_token_type = raw_token_type
        self.epsilon = 1e-6

    def extract_pcap(self, pcap_path):
        """
        Extract features from a PCAP file (streaming mode for efficiency)

        Returns:
            dict with raw_bigrams, raw_directions, raw_packet_ids,
                 packet_sizes, size_directions, iat_tokens
            or None if extraction fails
        """
        tuple_info = self._parse_filename_5tuple(pcap_path)
        if tuple_info is None:
            return None

        protocol, src_ip, src_port, dst_ip, dst_port = tuple_info
        flow_src = (src_ip, src_port)

        raw_bigrams = []
        raw_directions = []
        raw_packet_ids = []
        packet_sizes = []
        size_directions = []
        timestamps = []

        raw_packet_count = 0

        try:
            with PcapReader(pcap_path) as pcap_reader:
                for packet in pcap_reader:
                    if len(packet_sizes) >= self.max_size_packets:
                        break

                    payload = self._extract_payload(packet, protocol)
                    if payload is None or len(payload) == 0:
                        continue

                    payload_len = min(len(payload), 1500)
                    direction = self._get_direction(packet, protocol, flow_src)
                    timestamp = float(packet.time)

                    if raw_packet_count < self.max_raw_packets:
                        bigram_hex_list = self._bytes_to_hex_tokens(payload[:self.bytes_per_packet])
                        raw_bigrams.append(bigram_hex_list)
                        raw_directions.append(direction)
                        raw_packet_ids.append(raw_packet_count)
                        raw_packet_count += 1

                    packet_sizes.append(payload_len)
                    size_directions.append(direction)
                    timestamps.append(timestamp)

        except Exception as e:
            return None

        if len(packet_sizes) == 0:
            return None

        iat_tokens = self._compute_iat_tokens(timestamps)

        return {
            'protocol': protocol,
            'raw_bigrams': raw_bigrams,
            'raw_directions': raw_directions,
            'raw_packet_ids': raw_packet_ids,
            'packet_sizes': packet_sizes,
            'size_directions': size_directions,
            'iat_tokens': iat_tokens
        }

    def _parse_filename_5tuple(self, pcap_path):
        """Parse 5-tuple from filename: {protocol}_{src_ip}_{src_port}_{dst_ip}_{dst_port}.pcap"""
        try:
            filename = os.path.basename(pcap_path)
            filename = filename.replace('.pcap', '').replace('.pcapng', '')
            parts = filename.split('_')
            if len(parts) < 5:
                return None

            protocol_str = parts[0].lower()
            src_ip = parts[1]
            src_port = int(parts[2])
            dst_ip = parts[3]
            dst_port = int(parts[4])

            protocol_map = {'tcp': 6, 'udp': 17}
            protocol = protocol_map.get(protocol_str)
            if protocol is None:
                return None

            return (protocol, src_ip, src_port, dst_ip, dst_port)
        except Exception:
            return None

    def _get_direction(self, packet, protocol, flow_src):
        """Determine packet direction: 1=uplink (client->server), -1=downlink"""
        if IP not in packet:
            return 1

        ip_layer = packet[IP]
        if protocol == 6 and TCP in packet:
            transport_layer = packet[TCP]
        elif protocol == 17 and UDP in packet:
            transport_layer = packet[UDP]
        else:
            return 1

        src_ip = ip_layer.src
        src_port = transport_layer.sport

        if (src_ip, src_port) == flow_src:
            return 1
        else:
            return -1

    def _extract_payload(self, packet, protocol):
        """Extract transport layer payload"""
        try:
            if IP not in packet:
                return None

            if protocol == 6:
                if TCP not in packet:
                    return None
                payload = bytes(packet[TCP].payload)
            elif protocol == 17:
                if UDP not in packet:
                    return None
                payload = bytes(packet[UDP].payload)
            else:
                return None

            if len(payload) == 0:
                return None

            return payload
        except Exception:
            return None

    def _bytes_to_hex_tokens(self, payload_bytes):
        """Convert bytes to hex token list using pre-computed lookup table."""
        if self.raw_token_type == 'byte':
            return [_BYTE_TABLE[b] for b in payload_bytes]
        else:
            return [_BIGRAM_TABLE[payload_bytes[i] * 256 + payload_bytes[i + 1]]
                    for i in range(len(payload_bytes) - 1)]

    def _compute_iat_tokens(self, timestamps):
        """Compute IAT tokens using numpy vectorization."""
        if len(timestamps) == 0:
            return []

        ts = np.array(timestamps)
        iats = np.empty(len(ts))
        iats[0] = self.epsilon
        iats[1:] = np.maximum(np.diff(ts), self.epsilon)

        log_iats = np.log10(iats)
        normalized = 1.0 / (1.0 + np.exp(-log_iats))
        tokens = np.clip((normalized * 1000).astype(int), 0, 999)

        return tokens.tolist()


def create_tokenizer(vocab_path):
    """Create a BertTokenizer compatible with pretraining"""
    class Args:
        def __init__(self, vocab_path):
            self.vocab_path = vocab_path
            self.spm_model_path = None

    args = Args(vocab_path)
    tokenizer = BertTokenizer(args, is_src=True, do_lower_case=True)
    return tokenizer


# ============================================================
# Token cache: pre-compute tokenizer results for all possible
# token values, eliminating per-call tokenizer overhead.
# ============================================================

def _build_token_cache(tokenizer, token_strings):
    """Pre-compute tokenizer results for a list of token strings.

    Returns dict: token_string -> list of token IDs.
    """
    cache = {}
    for token_str in token_strings:
        tokens = tokenizer.tokenize(token_str)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        cache[token_str] = ids
    return cache


def build_all_caches(tokenizer_raw, tokenizer_size, tokenizer_temporal, raw_token_type='bigram'):
    """Build lookup caches for all three tokenizers (one-time cost)."""
    print("Building token caches...")

    if raw_token_type == 'bigram':
        raw_tokens = _BIGRAM_TABLE
    else:
        raw_tokens = _BYTE_TABLE
    raw_cache = _build_token_cache(tokenizer_raw, raw_tokens)
    print(f"  Raw cache: {len(raw_cache)} entries")

    size_cache = _build_token_cache(tokenizer_size, [str(v) for v in range(3001)])
    print(f"  Size cache: {len(size_cache)} entries")

    iat_cache = _build_token_cache(tokenizer_temporal, [str(v) for v in range(1000)])
    print(f"  IAT cache: {len(iat_cache)} entries")

    return raw_cache, size_cache, iat_cache


# ============================================================
# Cached tokenization functions
# ============================================================

def _tokenize_raw_cached(flow_data, raw_cache, seq_length_raw,
                          cls_id, sep_id, vocab_size):
    """Tokenize raw packet data using pre-built cache."""
    tokens = []
    token_packet_ids = []
    token_directions = []

    for pkt_idx, (bigrams, direction) in enumerate(
            zip(flow_data['raw_bigrams'], flow_data['raw_directions'])):
        if pkt_idx >= 8:
            break

        dir_idx = 0 if direction == -1 else 2
        for bigram in bigrams:
            ids = raw_cache.get(bigram)
            if ids is None:
                continue
            for token_id in ids:
                tokens.append(token_id)
                token_packet_ids.append(pkt_idx)
                token_directions.append(dir_idx)

    max_tokens = seq_length_raw - 2
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        token_packet_ids = token_packet_ids[:max_tokens]
        token_directions = token_directions[:max_tokens]

    src = [cls_id] + tokens + [sep_id]
    packet_ids = [8] + token_packet_ids + [8]
    directions = [1] + token_directions + [1]

    pad_len = seq_length_raw - len(src)
    if pad_len > 0:
        src.extend([PAD_ID] * pad_len)
        packet_ids.extend([8] * pad_len)
        directions.extend([1] * pad_len)

    for idx in range(len(src)):
        if src[idx] < 0 or src[idx] >= vocab_size:
            src[idx] = min(max(src[idx], 0), vocab_size - 1)

    return src, packet_ids, directions


def _tokenize_size_iat_cached(flow_data, size_cache, iat_cache, seq_length_size,
                               cls_id_size, sep_id_size,
                               cls_id_iat, sep_id_iat,
                               vocab_len_size, vocab_len_temporal):
    """Tokenize size+IAT data using pre-built caches."""
    size_token_ids = []
    for size, direction in zip(flow_data['packet_sizes'], flow_data['size_directions']):
        size_token = size * direction + 1500
        ids = size_cache.get(str(size_token))
        if ids is not None:
            size_token_ids.extend(ids)

    iat_token_ids = []
    for token in flow_data['iat_tokens']:
        ids = iat_cache.get(str(token))
        if ids is not None:
            iat_token_ids.extend(ids)

    min_len = min(len(size_token_ids), len(iat_token_ids))
    size_token_ids = size_token_ids[:min_len]
    iat_token_ids = iat_token_ids[:min_len]

    max_tokens = seq_length_size - 2
    if len(size_token_ids) > max_tokens:
        size_token_ids = size_token_ids[:max_tokens]
        iat_token_ids = iat_token_ids[:max_tokens]

    size_src = [cls_id_size] + size_token_ids + [sep_id_size]
    iat_src = [cls_id_iat] + iat_token_ids + [sep_id_iat]

    pad_len_size = seq_length_size - len(size_src)
    if pad_len_size > 0:
        size_src.extend([PAD_ID] * pad_len_size)
    pad_len_iat = seq_length_size - len(iat_src)
    if pad_len_iat > 0:
        iat_src.extend([PAD_ID] * pad_len_iat)

    for idx in range(len(size_src)):
        if size_src[idx] < 0 or size_src[idx] >= vocab_len_size:
            size_src[idx] = min(max(size_src[idx], 0), vocab_len_size - 1)
    for idx in range(len(iat_src)):
        if iat_src[idx] < 0 or iat_src[idx] >= vocab_len_temporal:
            iat_src[idx] = min(max(iat_src[idx], 0), vocab_len_temporal - 1)

    return size_src, iat_src


# ============================================================
# Multiprocessing worker
# ============================================================

_worker_state = {}


def _init_worker(extractor_params, raw_cache, size_cache, iat_cache,
                 vocab_raw, vocab_size, vocab_temporal,
                 seq_length_raw, seq_length_size):
    """Initialize per-worker state (called once per worker process)."""
    _worker_state['extractor'] = MultiModalFlowExtractor(**extractor_params)
    _worker_state['raw_cache'] = raw_cache
    _worker_state['size_cache'] = size_cache
    _worker_state['iat_cache'] = iat_cache
    _worker_state['seq_length_raw'] = seq_length_raw
    _worker_state['seq_length_size'] = seq_length_size
    _worker_state['cls_id_raw'] = vocab_raw.get(CLS_TOKEN, 0)
    _worker_state['sep_id_raw'] = vocab_raw.get(SEP_TOKEN, 0)
    _worker_state['cls_id_size'] = vocab_size.get(CLS_TOKEN, 0)
    _worker_state['sep_id_size'] = vocab_size.get(SEP_TOKEN, 0)
    _worker_state['cls_id_iat'] = vocab_temporal.get(CLS_TOKEN, 0)
    _worker_state['sep_id_iat'] = vocab_temporal.get(SEP_TOKEN, 0)
    _worker_state['vocab_size_raw'] = len(vocab_raw)
    _worker_state['vocab_size_size'] = len(vocab_size)
    _worker_state['vocab_size_temporal'] = len(vocab_temporal)


def _process_pcap_worker(item):
    """Worker function: extract + tokenize a single pcap file."""
    pcap_path, label_name = item
    st = _worker_state

    flow_data = st['extractor'].extract_pcap(pcap_path)
    if flow_data is None:
        return None

    raw_src, packet_ids, directions = _tokenize_raw_cached(
        flow_data, st['raw_cache'], st['seq_length_raw'],
        st['cls_id_raw'], st['sep_id_raw'], st['vocab_size_raw']
    )

    size_src, iat_src = _tokenize_size_iat_cached(
        flow_data, st['size_cache'], st['iat_cache'], st['seq_length_size'],
        st['cls_id_size'], st['sep_id_size'],
        st['cls_id_iat'], st['sep_id_iat'],
        st['vocab_size_size'], st['vocab_size_temporal']
    )

    return {
        'raw_src': raw_src,
        'packet_ids': packet_ids,
        'directions': directions,
        'size_src': size_src,
        'iat_src': iat_src,
        'label_name': label_name
    }


def _process_items(work_items, num_workers, init_args, desc="Processing"):
    """Process work items in parallel or sequentially."""
    results = []
    if num_workers > 1 and len(work_items) > 0:
        with mp.Pool(num_workers, initializer=_init_worker, initargs=init_args) as pool:
            for result in tqdm(pool.imap_unordered(_process_pcap_worker, work_items, chunksize=16),
                              total=len(work_items), desc=desc):
                if result is not None:
                    results.append(result)
    else:
        _init_worker(*init_args)
        for item in tqdm(work_items, desc=desc):
            result = _process_pcap_worker(item)
            if result is not None:
                results.append(result)
    return results


# ============================================================
# Backward-compatible tokenization functions
# (kept for external use; main pipeline uses cached versions)
# ============================================================

def tokenize_raw_flow(flow_data, tokenizer_raw, seq_length_raw):
    """
    Tokenize raw packet data for a single flow

    Format matches pretraining (data.py RawPacketDataset):
        - [CLS] + all_packet_tokens + [SEP] + [PAD]...
        - NO SEP tokens between packets (different from BERT sentence-pair!)
        - packet_ids: 0-7 for packets, 8 for special tokens/padding
        - directions: 0=downlink, 1=neutral(special/pad), 2=uplink
    """
    vocab = tokenizer_raw.vocab

    tokens = []
    token_packet_ids = []
    token_directions = []

    for pkt_idx, (bigrams, direction) in enumerate(zip(flow_data['raw_bigrams'], flow_data['raw_directions'])):
        if pkt_idx >= 8:
            break

        bigram_str = ' '.join(bigrams)
        pkt_tokens = tokenizer_raw.tokenize(bigram_str)
        pkt_token_ids = tokenizer_raw.convert_tokens_to_ids(pkt_tokens)

        dir_idx = 0 if direction == -1 else 2
        for token_id in pkt_token_ids:
            tokens.append(token_id)
            token_packet_ids.append(pkt_idx)
            token_directions.append(dir_idx)

    max_tokens = seq_length_raw - 2
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        token_packet_ids = token_packet_ids[:max_tokens]
        token_directions = token_directions[:max_tokens]

    cls_id = vocab.get(CLS_TOKEN, 0)
    sep_id = vocab.get(SEP_TOKEN, 0)

    src = [cls_id] + tokens + [sep_id]
    packet_ids = [8] + token_packet_ids + [8]
    directions = [1] + token_directions + [1]

    pad_len = seq_length_raw - len(src)
    if pad_len > 0:
        src.extend([PAD_ID] * pad_len)
        packet_ids.extend([8] * pad_len)
        directions.extend([1] * pad_len)

    vocab_size = len(vocab)
    for idx, v in enumerate(src):
        if v < 0 or v >= vocab_size:
            print(f"WARNING: raw token ID {v} at pos {idx} out of range [0, {vocab_size}), clamping")
            src[idx] = min(max(v, 0), vocab_size - 1)

    return src, packet_ids, directions


def tokenize_size_iat_flow(flow_data, tokenizer_size, tokenizer_temporal, seq_length_size):
    """
    Tokenize Packet Size modality (Size + IAT) for a single flow

    Format matches pretraining (data.py PacketSizeDataset):
        - Size: [CLS] + size_tokens + [SEP] + [PAD]...
        - IAT:  [CLS] + iat_tokens + [SEP] + [PAD]...
        - Size token = size * direction + 1500 (direction already encoded)
        - IAT tokens are already discretized to 0-999
    """
    vocab_size = tokenizer_size.vocab
    vocab_temporal = tokenizer_temporal.vocab

    size_tokens_str = []
    for size, direction in zip(flow_data['packet_sizes'], flow_data['size_directions']):
        size_token = size * direction + 1500
        size_tokens_str.append(str(size_token))

    size_str = ' '.join(size_tokens_str)
    size_tokens = tokenizer_size.tokenize(size_str)
    size_token_ids = tokenizer_size.convert_tokens_to_ids(size_tokens)

    iat_str = ' '.join([str(token) for token in flow_data['iat_tokens']])
    iat_tokens = tokenizer_temporal.tokenize(iat_str)
    iat_token_ids = tokenizer_temporal.convert_tokens_to_ids(iat_tokens)

    min_len = min(len(size_token_ids), len(iat_token_ids))
    size_token_ids = size_token_ids[:min_len]
    iat_token_ids = iat_token_ids[:min_len]

    max_tokens = seq_length_size - 2
    if len(size_token_ids) > max_tokens:
        size_token_ids = size_token_ids[:max_tokens]
        iat_token_ids = iat_token_ids[:max_tokens]

    cls_id_size = vocab_size.get(CLS_TOKEN, 0)
    sep_id_size = vocab_size.get(SEP_TOKEN, 0)
    size_src = [cls_id_size] + size_token_ids + [sep_id_size]

    cls_id_iat = vocab_temporal.get(CLS_TOKEN, 0)
    sep_id_iat = vocab_temporal.get(SEP_TOKEN, 0)
    iat_src = [cls_id_iat] + iat_token_ids + [sep_id_iat]

    pad_len_size = seq_length_size - len(size_src)
    if pad_len_size > 0:
        size_src.extend([PAD_ID] * pad_len_size)
    pad_len_iat = seq_length_size - len(iat_src)
    if pad_len_iat > 0:
        iat_src.extend([PAD_ID] * pad_len_iat)

    size_vocab_len = len(vocab_size)
    for idx, v in enumerate(size_src):
        if v < 0 or v >= size_vocab_len:
            print(f"WARNING: size token ID {v} at pos {idx} out of range [0, {size_vocab_len}), clamping")
            size_src[idx] = min(max(v, 0), size_vocab_len - 1)
    temporal_vocab_len = len(vocab_temporal)
    for idx, v in enumerate(iat_src):
        if v < 0 or v >= temporal_vocab_len:
            print(f"WARNING: IAT token ID {v} at pos {idx} out of range [0, {temporal_vocab_len}), clamping")
            iat_src[idx] = min(max(v, 0), temporal_vocab_len - 1)

    return size_src, iat_src


# ============================================================
# Main processing pipeline
# ============================================================

def process_dataset(pcap_dir, tokenizer_raw, tokenizer_size, tokenizer_temporal,
                    seq_length_raw=512, seq_length_size=256,
                    bytes_per_packet=64, max_raw_packets=8, max_size_packets=256,
                    min_samples_per_class=10, max_samples_per_class=500,
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                    seed=42, test_dir=None, raw_token_type='bigram',
                    num_workers=None):
    """
    Process all PCAP files in dataset directory with parallel processing.

    Args:
        pcap_dir: Root directory with structure: pcap_dir/label_name/*.pcap
        tokenizer_raw: BertTokenizer for raw modality
        tokenizer_size: BertTokenizer for size modality
        tokenizer_temporal: BertTokenizer for temporal (IAT) modality
        seq_length_raw: Sequence length for raw modality
        seq_length_size: Sequence length for size/IAT modality
        bytes_per_packet: Bytes to extract per packet for raw modality
        max_raw_packets: Max packets for raw modality
        max_size_packets: Max packets for size modality
        min_samples_per_class: Minimum samples required per class
        max_samples_per_class: Maximum samples per class (for balancing)
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data (ignored when test_dir is specified)
        seed: Random seed
        test_dir: Separate test directory
        raw_token_type: 'bigram' or 'byte'
        num_workers: Number of parallel workers (default: cpu_count - 1)

    Returns:
        train_data, val_data, test_data: Lists of samples
        label2id: Label name to ID mapping
    """
    random.seed(seed)
    np.random.seed(seed)

    # Build token caches (one-time cost, eliminates tokenizer overhead)
    raw_cache, size_cache, iat_cache = build_all_caches(
        tokenizer_raw, tokenizer_size, tokenizer_temporal, raw_token_type
    )

    extractor_params = {
        'bytes_per_packet': bytes_per_packet,
        'max_raw_packets': max_raw_packets,
        'max_size_packets': max_size_packets,
        'raw_token_type': raw_token_type
    }

    vocab_raw = tokenizer_raw.vocab
    vocab_size_dict = tokenizer_size.vocab
    vocab_temporal = tokenizer_temporal.vocab

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    init_args = (extractor_params, raw_cache, size_cache, iat_cache,
                 vocab_raw, vocab_size_dict, vocab_temporal,
                 seq_length_raw, seq_length_size)

    # Collect work items for training directory
    work_items = []
    print(f"Scanning {pcap_dir}...")
    for label_name in sorted(os.listdir(pcap_dir)):
        label_path = os.path.join(pcap_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        pcap_files = [f for f in os.listdir(label_path)
                      if f.endswith('.pcap') or f.endswith('.pcapng')]
        print(f"  Found label '{label_name}': {len(pcap_files)} files")
        for pcap_file in pcap_files:
            work_items.append((os.path.join(label_path, pcap_file), label_name))

    print(f"Total: {len(work_items)} pcap files, using {num_workers} workers")

    # Process all pcap files in parallel
    results = _process_items(work_items, num_workers, init_args, desc="Processing train/val")

    # Group by label
    label_samples = defaultdict(list)
    for result in results:
        label_samples[result['label_name']].append(result)

    # Filter labels with too few samples
    valid_labels = []
    for label_name, samples in label_samples.items():
        if len(samples) >= min_samples_per_class:
            valid_labels.append(label_name)
        else:
            print(f"Skipping label '{label_name}': only {len(samples)} samples (< {min_samples_per_class})")

    valid_labels = sorted(valid_labels)
    label2id = {name: idx for idx, name in enumerate(valid_labels)}

    print(f"\nValid labels: {len(valid_labels)}")
    for name in valid_labels:
        print(f"  {name}: {len(label_samples[name])} samples")

    # Sample and split data
    train_data, val_data, test_data = [], [], []

    if test_dir is None:
        # Original mode: split from single directory
        for label_name in valid_labels:
            samples = label_samples[label_name]

            if len(samples) > max_samples_per_class:
                samples = random.sample(samples, max_samples_per_class)

            random.shuffle(samples)

            label_id = label2id[label_name]
            for s in samples:
                s['label'] = label_id

            n = len(samples)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_data.extend(samples[:n_train])
            val_data.extend(samples[n_train:n_train + n_val])
            test_data.extend(samples[n_train + n_val:])
    else:
        # Separate test directory mode
        total_ratio = train_ratio + val_ratio
        norm_train_ratio = train_ratio / total_ratio

        print(f"\nSeparate test mode: train/val split ratio = {norm_train_ratio:.2f}/{1-norm_train_ratio:.2f}")

        for label_name in valid_labels:
            samples = label_samples[label_name]

            if len(samples) > max_samples_per_class:
                samples = random.sample(samples, max_samples_per_class)

            random.shuffle(samples)

            label_id = label2id[label_name]
            for s in samples:
                s['label'] = label_id

            n = len(samples)
            n_train = int(n * norm_train_ratio)

            train_data.extend(samples[:n_train])
            val_data.extend(samples[n_train:])

        # Process test directory in parallel
        print(f"\nProcessing test directory: {test_dir}")
        test_items = []
        skipped_labels = set()

        for label_name in sorted(os.listdir(test_dir)):
            label_path = os.path.join(test_dir, label_name)
            if not os.path.isdir(label_path):
                continue

            if label_name not in label2id:
                skipped_labels.add(label_name)
                continue

            pcap_files = [f for f in os.listdir(label_path)
                          if f.endswith('.pcap') or f.endswith('.pcapng')]
            for pcap_file in pcap_files:
                test_items.append((os.path.join(label_path, pcap_file), label_name))

        test_results = _process_items(test_items, num_workers, init_args, desc="Processing test")

        test_label_counts = defaultdict(int)
        for result in test_results:
            label_name = result['label_name']
            result['label'] = label2id[label_name]
            test_data.append(result)
            test_label_counts[label_name] += 1

        if skipped_labels:
            print(f"\nSkipped labels in test_dir (not in train): {sorted(skipped_labels)}")

        print(f"\nTest set label distribution:")
        for name in sorted(test_label_counts.keys()):
            print(f"  {name}: {test_label_counts[name]} samples")

    # Shuffle splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")

    return train_data, val_data, test_data, label2id


def save_dataset(data, output_path):
    """Save dataset to pickle file"""
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved dataset to {output_path}")


def load_dataset(input_path):
    """Load dataset from pickle file"""
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description='Process PCAP files for multi-modal fine-tuning')

    parser.add_argument('--pcap_dir', type=str, required=True,
                        help='Root directory with structure: pcap_dir/label_name/*.pcap (used as train dir when --test_dir is specified)')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Separate test directory. When specified, pcap_dir is used for train/val only, '
                             'and all samples in test_dir are used for testing. Labels are determined by pcap_dir.')
    parser.add_argument('--vocab_path_raw', type=str, required=True,
                        help='Path to raw modality vocabulary file')
    parser.add_argument('--vocab_path_size', type=str, required=True,
                        help='Path to size modality vocabulary file')
    parser.add_argument('--vocab_path_temporal', type=str, required=True,
                        help='Path to temporal (IAT) modality vocabulary file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed datasets')

    # Sequence lengths
    parser.add_argument('--seq_length_raw', type=int, default=512)
    parser.add_argument('--seq_length_size', type=int, default=256)

    # Feature extraction params
    parser.add_argument('--bytes_per_packet', type=int, default=64)
    parser.add_argument('--max_raw_packets', type=int, default=8)
    parser.add_argument('--max_size_packets', type=int, default=256)
    parser.add_argument('--raw_token_type', type=str, choices=['bigram', 'byte'], default='bigram',
                        help='Raw tokenization: bigram (65K vocab, default) or byte (256 vocab)')

    # Dataset filtering
    parser.add_argument('--min_samples_per_class', type=int, default=10)
    parser.add_argument('--max_samples_per_class', type=int, default=500)

    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 1)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create tokenizers (same as pretraining)
    print("Creating tokenizers...")
    tokenizer_raw = create_tokenizer(args.vocab_path_raw)
    tokenizer_size = create_tokenizer(args.vocab_path_size)
    tokenizer_temporal = create_tokenizer(args.vocab_path_temporal)
    print(f"  Raw vocab size: {len(tokenizer_raw.vocab)}")
    print(f"  Size vocab size: {len(tokenizer_size.vocab)}")
    print(f"  Temporal vocab size: {len(tokenizer_temporal.vocab)}")

    # Process dataset
    train_data, val_data, test_data, label2id = process_dataset(
        pcap_dir=args.pcap_dir,
        tokenizer_raw=tokenizer_raw,
        tokenizer_size=tokenizer_size,
        tokenizer_temporal=tokenizer_temporal,
        seq_length_raw=args.seq_length_raw,
        seq_length_size=args.seq_length_size,
        bytes_per_packet=args.bytes_per_packet,
        max_raw_packets=args.max_raw_packets,
        max_size_packets=args.max_size_packets,
        min_samples_per_class=args.min_samples_per_class,
        max_samples_per_class=args.max_samples_per_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        test_dir=args.test_dir,
        raw_token_type=args.raw_token_type,
        num_workers=args.num_workers
    )

    # Save datasets
    save_dataset(train_data, os.path.join(args.output_dir, 'train.pkl'))
    save_dataset(val_data, os.path.join(args.output_dir, 'val.pkl'))
    save_dataset(test_data, os.path.join(args.output_dir, 'test.pkl'))
    save_dataset(label2id, os.path.join(args.output_dir, 'label2id.pkl'))

    print("\nDone!")


if __name__ == '__main__':
    main()

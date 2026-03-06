"""
Flow Prediction Data Utilities

PCAP processing and dataset creation for flow prediction task.
Predicts flow_bytes (total bytes) and flow_duration from first 8 packets.

Input format: datasets/label_name/flow_file.pcap
  - Each pcap file represents one bidirectional flow
  - Filename format: {protocol}_{src_ip}_{src_port}_{dst_ip}_{dst_port}.pcap

Output: train/val/test datasets with features from first 8 packets and
        labels computed from entire flow (flow_bytes, flow_duration)
"""

import os
import sys
sys.path.append(os.getcwd())

import random
import math
import numpy as np
from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from tqdm import tqdm
from collections import defaultdict
import pickle
import argparse

from uer.utils.vocab import Vocab
from uer.utils.tokenizers import BertTokenizer
from uer.utils.constants import CLS_TOKEN, SEP_TOKEN, PAD_ID


class FlowFeatureExtractor:
    """
    Extract features for flow prediction from a single PCAP file.

    Features: First 8 packets (Raw + Size + IAT modalities)
    Labels: Entire flow statistics (flow_bytes, flow_duration)
    """

    def __init__(self, bytes_per_packet=64, max_packets=8, raw_token_type='bigram'):
        self.bytes_per_packet = bytes_per_packet
        self.max_packets = max_packets
        self.raw_token_type = raw_token_type
        self.epsilon = 1e-6  # For IAT computation

    def extract_pcap(self, pcap_path):
        """
        Extract features from first 8 packets and labels from entire flow.

        Uses streaming PcapReader to read entire PCAP for label computation
        but only uses first 8 packets for features.

        Returns:
            dict with keys:
                - raw_bigrams: List[List[str]] - bigram hex strings per packet
                - raw_directions: List[int] - direction per packet (1=up, -1=down)
                - raw_packet_ids: List[int] - packet index per packet
                - packet_sizes: List[int] - payload sizes (first 8 packets)
                - size_directions: List[int] - direction per size packet
                - iat_tokens: List[int] - IAT temporal tokens (0-999)
                - flow_bytes: int - total bytes in entire flow
                - flow_bytes_log: float - log10(flow_bytes + 1)
                - flow_duration: float - total duration in seconds
                - flow_duration_log: float - log10(flow_duration + epsilon)
            or None if extraction fails
        """
        # Parse 5-tuple from filename
        tuple_info = self._parse_filename_5tuple(pcap_path)
        if tuple_info is None:
            return None

        protocol, src_ip, src_port, dst_ip, dst_port = tuple_info
        flow_src = (src_ip, src_port)

        # Feature arrays (first 8 packets)
        raw_bigrams = []
        raw_directions = []
        raw_packet_ids = []
        packet_sizes = []
        size_directions = []
        timestamps = []

        # Label computation (entire flow)
        total_bytes = 0
        first_timestamp = None
        last_timestamp = None

        raw_packet_count = 0

        # Streaming PCAP reading
        try:
            with PcapReader(pcap_path) as pcap_reader:
                for packet in pcap_reader:
                    payload = self._extract_payload(packet, protocol)
                    if payload is None or len(payload) == 0:
                        continue

                    payload_len = min(len(payload), 1500)
                    direction = self._get_direction(packet, protocol, flow_src)
                    timestamp = float(packet.time)

                    # Update label statistics (entire flow)
                    total_bytes += payload_len
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    last_timestamp = timestamp

                    # Feature extraction (first 8 packets only)
                    if raw_packet_count < self.max_packets:
                        # Raw modality
                        bigram_hex_list = self._bytes_to_hex_tokens(payload[:self.bytes_per_packet])
                        raw_bigrams.append(bigram_hex_list)
                        raw_directions.append(direction)
                        raw_packet_ids.append(raw_packet_count)

                        # Size modality
                        packet_sizes.append(payload_len)
                        size_directions.append(direction)
                        timestamps.append(timestamp)

                        raw_packet_count += 1

        except Exception as e:
            return None

        if len(packet_sizes) == 0:
            return None

        # Compute IAT tokens for feature packets
        iat_tokens = self._compute_iat_tokens(timestamps)

        # Compute flow labels
        flow_bytes = total_bytes
        flow_bytes_log = math.log10(flow_bytes + 1)

        flow_duration = (last_timestamp - first_timestamp) if first_timestamp is not None else 0.0
        flow_duration = max(flow_duration, 0.0)
        flow_duration_log = math.log10(flow_duration + self.epsilon)

        return {
            'protocol': protocol,
            # Features (first 8 packets)
            'raw_bigrams': raw_bigrams,
            'raw_directions': raw_directions,
            'raw_packet_ids': raw_packet_ids,
            'packet_sizes': packet_sizes,
            'size_directions': size_directions,
            'iat_tokens': iat_tokens,
            # Labels (entire flow)
            'flow_bytes': flow_bytes,
            'flow_bytes_log': flow_bytes_log,
            'flow_duration': flow_duration,
            'flow_duration_log': flow_duration_log
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
            return 1  # uplink
        else:
            return -1  # downlink

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
        """Convert bytes to hex token list.

        bigram mode: ["4500", "0006", ...] (sliding window, N-1 tokens)
        byte mode:   ["45", "00", "06", ...] (individual bytes, N tokens)
        """
        if self.raw_token_type == 'byte':
            return [f"{b:02x}" for b in payload_bytes]
        else:
            bigrams = []
            for i in range(len(payload_bytes) - 1):
                byte1 = payload_bytes[i]
                byte2 = payload_bytes[i + 1]
                bigram_hex = f"{byte1:02x}{byte2:02x}"
                bigrams.append(bigram_hex)
            return bigrams

    def _compute_iat_tokens(self, timestamps):
        """
        Compute IAT (Inter-Arrival Time) tokens.

        Method from PTU paper (same as pretraining):
        1. Calculate time difference between consecutive packets (seconds)
        2. Normalize: sigmoid(log10(IAT + epsilon))
        3. Discretize to 1000 bins (0-999)

        Args:
            timestamps: List[float] - packet timestamps (seconds)

        Returns:
            List[int]: IAT tokens (0-999)
        """
        if len(timestamps) == 0:
            return []

        iat_tokens = []

        for i in range(len(timestamps)):
            if i == 0:
                iat = self.epsilon
            else:
                iat = timestamps[i] - timestamps[i-1]
                iat = max(iat, self.epsilon)

            log_iat = math.log10(iat)
            normalized = 1.0 / (1.0 + math.exp(-log_iat))

            token = int(normalized * 1000)
            token = min(max(token, 0), 999)

            iat_tokens.append(token)

        return iat_tokens


def create_tokenizer(vocab_path):
    """
    Create a BertTokenizer compatible with pretraining.
    """
    class Args:
        def __init__(self, vocab_path):
            self.vocab_path = vocab_path
            self.spm_model_path = None

    args = Args(vocab_path)
    tokenizer = BertTokenizer(args, is_src=True, do_lower_case=True)
    return tokenizer


def tokenize_raw_flow(flow_data, tokenizer_raw, seq_length_raw):
    """
    Tokenize raw packet data for a single flow.

    Format matches pretraining:
        - [CLS] + all_packet_tokens + [SEP] + [PAD]...
        - NO SEP tokens between packets
        - packet_ids: 0-7 for packets, 8 for special tokens/padding
        - directions: 0=downlink, 1=neutral(special/pad), 2=uplink

    Args:
        flow_data: dict with 'raw_bigrams' and 'raw_directions'
        tokenizer_raw: BertTokenizer for raw modality
        seq_length_raw: target sequence length

    Returns:
        src: List[int] - token IDs
        packet_ids: List[int] - packet indices
        directions: List[int] - direction indices (0=down, 1=neutral, 2=up)
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

        for token_id in pkt_token_ids:
            tokens.append(token_id)
            token_packet_ids.append(pkt_idx)
            dir_idx = 0 if direction == -1 else 2
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

    while len(src) < seq_length_raw:
        src.append(PAD_ID)
        packet_ids.append(8)
        directions.append(1)

    return src, packet_ids, directions


def tokenize_size_iat_flow(flow_data, tokenizer_size, tokenizer_temporal, seq_length_size):
    """
    Tokenize Packet Size modality (Size + IAT) for a single flow.

    For flow prediction, we only use first 8 packets, so seq_length_size
    should be small (e.g., 10 = 8 packets + CLS + SEP).

    Format matches pretraining:
        - Size: [CLS] + size_tokens + [SEP] + [PAD]...
        - IAT:  [CLS] + iat_tokens + [SEP] + [PAD]...
        - Size token = size * direction + 1500

    Args:
        flow_data: dict with 'packet_sizes', 'size_directions', 'iat_tokens'
        tokenizer_size: BertTokenizer for size tokens
        tokenizer_temporal: BertTokenizer for temporal (IAT) tokens
        seq_length_size: target sequence length

    Returns:
        size_src: List[int] - size token IDs
        iat_src: List[int] - IAT token IDs
    """
    vocab_size = tokenizer_size.vocab
    vocab_temporal = tokenizer_temporal.vocab

    # Process Size tokens
    size_tokens_str = []
    for size, direction in zip(flow_data['packet_sizes'], flow_data['size_directions']):
        size_token = size * direction + 1500
        size_tokens_str.append(str(size_token))

    size_str = ' '.join(size_tokens_str)
    size_tokens = tokenizer_size.tokenize(size_str)
    size_token_ids = tokenizer_size.convert_tokens_to_ids(size_tokens)

    # Process IAT tokens
    iat_str = ' '.join([str(token) for token in flow_data['iat_tokens']])
    iat_tokens = tokenizer_temporal.tokenize(iat_str)
    iat_token_ids = tokenizer_temporal.convert_tokens_to_ids(iat_tokens)

    # Ensure Size and IAT have same length
    min_len = min(len(size_token_ids), len(iat_token_ids))
    size_token_ids = size_token_ids[:min_len]
    iat_token_ids = iat_token_ids[:min_len]

    # Truncate to max_tokens
    max_tokens = seq_length_size - 2
    if len(size_token_ids) > max_tokens:
        size_token_ids = size_token_ids[:max_tokens]
        iat_token_ids = iat_token_ids[:max_tokens]

    # Build sequences with special tokens
    cls_id_size = vocab_size.get(CLS_TOKEN, 0)
    sep_id_size = vocab_size.get(SEP_TOKEN, 0)
    size_src = [cls_id_size] + size_token_ids + [sep_id_size]

    cls_id_iat = vocab_temporal.get(CLS_TOKEN, 0)
    sep_id_iat = vocab_temporal.get(SEP_TOKEN, 0)
    iat_src = [cls_id_iat] + iat_token_ids + [sep_id_iat]

    # Padding
    while len(size_src) < seq_length_size:
        size_src.append(PAD_ID)
    while len(iat_src) < seq_length_size:
        iat_src.append(PAD_ID)

    return size_src, iat_src


def process_pcap_dir(pcap_dir, tokenizer_raw, tokenizer_size, tokenizer_temporal,
                     seq_length_raw=512, seq_length_size=10,
                     bytes_per_packet=64, max_packets=8,
                     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                     seed=42, min_flow_bytes=100, max_samples=None,
                     raw_token_type='bigram'):
    """
    Process all PCAP files in directory for flow prediction.

    Args:
        pcap_dir: Root directory with structure: pcap_dir/label_name/*.pcap
                  (label_name is ignored for regression task, but used for organization)
        tokenizer_raw: BertTokenizer for raw modality
        tokenizer_size: BertTokenizer for size modality
        tokenizer_temporal: BertTokenizer for temporal (IAT) modality
        seq_length_raw: Sequence length for raw modality (default 512)
        seq_length_size: Sequence length for size/IAT modality (default 10 for 8 packets)
        bytes_per_packet: Bytes to extract per packet for raw modality
        max_packets: Max packets for feature extraction (default 8)
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed
        min_flow_bytes: Minimum flow bytes to include (filter tiny flows)
        max_samples: Maximum total samples to collect (None for unlimited)

    Returns:
        train_data, val_data, test_data: Lists of samples
        stats: dict with dataset statistics
    """
    random.seed(seed)
    np.random.seed(seed)

    extractor = FlowFeatureExtractor(
        bytes_per_packet=bytes_per_packet,
        max_packets=max_packets,
        raw_token_type=raw_token_type
    )

    all_samples = []

    # Walk through directory structure
    print(f"Scanning {pcap_dir}...")
    for label_name in os.listdir(pcap_dir):
        label_path = os.path.join(pcap_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        pcap_files = [f for f in os.listdir(label_path)
                      if f.endswith('.pcap') or f.endswith('.pcapng')]

        print(f"Processing '{label_name}': {len(pcap_files)} files")

        for pcap_file in tqdm(pcap_files, desc=label_name):
            if max_samples is not None and len(all_samples) >= max_samples:
                break

            pcap_path = os.path.join(label_path, pcap_file)
            flow_data = extractor.extract_pcap(pcap_path)

            if flow_data is None:
                continue

            # Filter tiny flows
            if flow_data['flow_bytes'] < min_flow_bytes:
                continue

            # Tokenize raw modality
            raw_src, packet_ids, directions = tokenize_raw_flow(
                flow_data, tokenizer_raw, seq_length_raw
            )

            # Tokenize size modality
            size_src, iat_src = tokenize_size_iat_flow(
                flow_data, tokenizer_size, tokenizer_temporal, seq_length_size
            )

            sample = {
                'raw_src': raw_src,
                'packet_ids': packet_ids,
                'directions': directions,
                'size_src': size_src,
                'iat_src': iat_src,
                'flow_bytes': flow_data['flow_bytes'],
                'flow_bytes_log': flow_data['flow_bytes_log'],
                'flow_duration': flow_data['flow_duration'],
                'flow_duration_log': flow_data['flow_duration_log'],
                'label_name': label_name  # Keep for reference
            }
            all_samples.append(sample)

        if max_samples is not None and len(all_samples) >= max_samples:
            print(f"Reached max_samples limit ({max_samples})")
            break

    print(f"\nTotal samples collected: {len(all_samples)}")

    # Compute statistics
    flow_bytes_values = [s['flow_bytes'] for s in all_samples]
    flow_bytes_log_values = [s['flow_bytes_log'] for s in all_samples]
    flow_duration_values = [s['flow_duration'] for s in all_samples]
    flow_duration_log_values = [s['flow_duration_log'] for s in all_samples]

    stats = {
        'flow_bytes': {
            'mean': np.mean(flow_bytes_values),
            'std': np.std(flow_bytes_values),
            'min': np.min(flow_bytes_values),
            'max': np.max(flow_bytes_values),
            'median': np.median(flow_bytes_values)
        },
        'flow_bytes_log': {
            'mean': np.mean(flow_bytes_log_values),
            'std': np.std(flow_bytes_log_values),
            'min': np.min(flow_bytes_log_values),
            'max': np.max(flow_bytes_log_values),
            'median': np.median(flow_bytes_log_values)
        },
        'flow_duration': {
            'mean': np.mean(flow_duration_values),
            'std': np.std(flow_duration_values),
            'min': np.min(flow_duration_values),
            'max': np.max(flow_duration_values),
            'median': np.median(flow_duration_values)
        },
        'flow_duration_log': {
            'mean': np.mean(flow_duration_log_values),
            'std': np.std(flow_duration_log_values),
            'min': np.min(flow_duration_log_values),
            'max': np.max(flow_duration_log_values),
            'median': np.median(flow_duration_log_values)
        }
    }

    print("\nFlow Statistics:")
    print(f"  flow_bytes: mean={stats['flow_bytes']['mean']:.2f}, "
          f"std={stats['flow_bytes']['std']:.2f}, "
          f"min={stats['flow_bytes']['min']:.2f}, max={stats['flow_bytes']['max']:.2f}")
    print(f"  flow_bytes_log: mean={stats['flow_bytes_log']['mean']:.4f}, "
          f"std={stats['flow_bytes_log']['std']:.4f}")
    print(f"  flow_duration: mean={stats['flow_duration']['mean']:.4f}s, "
          f"std={stats['flow_duration']['std']:.4f}s")
    print(f"  flow_duration_log: mean={stats['flow_duration_log']['mean']:.4f}, "
          f"std={stats['flow_duration_log']['std']:.4f}")

    # Shuffle and split
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = all_samples[:n_train]
    val_data = all_samples[n_train:n_train + n_val]
    test_data = all_samples[n_train + n_val:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")

    return train_data, val_data, test_data, stats


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
    parser = argparse.ArgumentParser(description='Process PCAP files for flow prediction')

    parser.add_argument('--pcap_dir', type=str, required=True,
                        help='Root directory with structure: pcap_dir/label_name/*.pcap')
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
    parser.add_argument('--seq_length_size', type=int, default=10,
                        help='Sequence length for size modality (8 packets + CLS + SEP)')

    # Feature extraction params
    parser.add_argument('--bytes_per_packet', type=int, default=64)
    parser.add_argument('--max_packets', type=int, default=8)
    parser.add_argument('--raw_token_type', type=str, choices=['bigram', 'byte'], default='bigram',
                        help='Raw tokenization: bigram (65K vocab, default) or byte (256 vocab)')

    # Dataset filtering
    parser.add_argument('--min_flow_bytes', type=int, default=100,
                        help='Minimum flow bytes to include')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum total samples to collect')

    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create tokenizers
    print("Creating tokenizers...")
    tokenizer_raw = create_tokenizer(args.vocab_path_raw)
    tokenizer_size = create_tokenizer(args.vocab_path_size)
    tokenizer_temporal = create_tokenizer(args.vocab_path_temporal)
    print(f"  Raw vocab size: {len(tokenizer_raw.vocab)}")
    print(f"  Size vocab size: {len(tokenizer_size.vocab)}")
    print(f"  Temporal vocab size: {len(tokenizer_temporal.vocab)}")

    # Process dataset
    train_data, val_data, test_data, stats = process_pcap_dir(
        pcap_dir=args.pcap_dir,
        tokenizer_raw=tokenizer_raw,
        tokenizer_size=tokenizer_size,
        tokenizer_temporal=tokenizer_temporal,
        seq_length_raw=args.seq_length_raw,
        seq_length_size=args.seq_length_size,
        bytes_per_packet=args.bytes_per_packet,
        max_packets=args.max_packets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_flow_bytes=args.min_flow_bytes,
        max_samples=args.max_samples,
        raw_token_type=args.raw_token_type
    )

    # Save datasets
    save_dataset(train_data, os.path.join(args.output_dir, 'train.pkl'))
    save_dataset(val_data, os.path.join(args.output_dir, 'val.pkl'))
    save_dataset(test_data, os.path.join(args.output_dir, 'test.pkl'))
    save_dataset(stats, os.path.join(args.output_dir, 'stats.pkl'))

    print("\nDone!")


if __name__ == '__main__':
    main()

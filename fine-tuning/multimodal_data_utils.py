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
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
from tqdm import tqdm
from collections import defaultdict
import pickle
import argparse

from uer.utils.vocab import Vocab
from uer.utils.tokenizers import BertTokenizer
from uer.utils.constants import CLS_TOKEN, SEP_TOKEN, PAD_ID


class MultiModalFlowExtractor:
    """Extract multi-modal features from a single PCAP file"""

    def __init__(self, bytes_per_packet=64, max_raw_packets=8, max_size_packets=256):
        self.bytes_per_packet = bytes_per_packet
        self.max_raw_packets = max_raw_packets
        self.max_size_packets = max_size_packets

    def extract_pcap(self, pcap_path):
        """
        Extract features from a PCAP file

        Returns:
            dict with keys:
                - raw_bigrams: List[List[str]] - bigram hex strings per packet
                - raw_directions: List[int] - direction per packet (1=up, -1=down)
                - raw_packet_ids: List[int] - packet index per packet
                - packet_sizes: List[int] - payload sizes
                - size_directions: List[int] - direction per size packet
            or None if extraction fails
        """
        try:
            packets = rdpcap(pcap_path)
        except Exception as e:
            return None

        if len(packets) == 0:
            return None

        # Parse 5-tuple from filename
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

        raw_packet_count = 0

        for packet in packets:
            payload = self._extract_payload(packet, protocol)
            if payload is None or len(payload) == 0:
                continue

            payload_len = min(len(payload), 1500)
            direction = self._get_direction(packet, protocol, flow_src)

            # Raw modality
            if raw_packet_count < self.max_raw_packets:
                bigram_hex_list = self._bytes_to_bigram_hex(payload[:self.bytes_per_packet])
                raw_bigrams.append(bigram_hex_list)
                raw_directions.append(direction)
                raw_packet_ids.append(raw_packet_count)
                raw_packet_count += 1

            # Size modality
            if len(packet_sizes) < self.max_size_packets:
                packet_sizes.append(payload_len)
                size_directions.append(direction)

        if len(packet_sizes) == 0:
            return None

        return {
            'protocol': protocol,
            'raw_bigrams': raw_bigrams,
            'raw_directions': raw_directions,
            'raw_packet_ids': raw_packet_ids,
            'packet_sizes': packet_sizes,
            'size_directions': size_directions
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

    def _bytes_to_bigram_hex(self, payload_bytes):
        """Convert bytes to bigram hex string list: ["4500", "0006", ...]"""
        bigrams = []
        for i in range(len(payload_bytes) - 1):
            byte1 = payload_bytes[i]
            byte2 = payload_bytes[i + 1]
            bigram_hex = f"{byte1:02x}{byte2:02x}"
            bigrams.append(bigram_hex)
        return bigrams


def create_tokenizer(vocab_path):
    """
    Create a BertTokenizer compatible with pretraining

    Uses the same tokenizer as pretraining (data.py)
    """
    # Create a minimal args object for BertTokenizer
    class Args:
        def __init__(self, vocab_path):
            self.vocab_path = vocab_path
            self.spm_model_path = None

    args = Args(vocab_path)
    tokenizer = BertTokenizer(args, is_src=True, do_lower_case=True)
    return tokenizer


def tokenize_raw_flow(flow_data, tokenizer_raw, seq_length_raw):
    """
    Tokenize raw packet data for a single flow

    Format matches pretraining (data.py RawPacketDataset):
        - [CLS] + all_packet_tokens + [SEP] + [PAD]...
        - NO SEP tokens between packets (different from BERT sentence-pair!)
        - packet_ids: 0-7 for packets, 8 for special tokens/padding
        - directions: 0=downlink, 1=neutral(special/pad), 2=uplink

    Args:
        flow_data: dict with 'raw_bigrams' (List[List[str]]) and 'raw_directions' (List[int])
        tokenizer_raw: BertTokenizer for raw modality
        seq_length_raw: target sequence length

    Returns:
        src: List[int] - token IDs
        packet_ids: List[int] - packet indices
        directions: List[int] - direction indices (0=down, 1=neutral, 2=up)
    """
    vocab = tokenizer_raw.vocab

    # Collect all tokens from all packets (no SEP between packets!)
    tokens = []
    token_packet_ids = []
    token_directions = []

    for pkt_idx, (bigrams, direction) in enumerate(zip(flow_data['raw_bigrams'], flow_data['raw_directions'])):
        # Limit to 8 packets (indices 0-7)
        if pkt_idx >= 8:
            break

        # Convert bigram list to space-separated string, then tokenize
        # This matches pretraining: tokenizer.tokenize(bigram_str)
        bigram_str = ' '.join(bigrams)
        pkt_tokens = tokenizer_raw.tokenize(bigram_str)
        pkt_token_ids = tokenizer_raw.convert_tokens_to_ids(pkt_tokens)

        # Add tokens for this packet
        for token_id in pkt_token_ids:
            tokens.append(token_id)
            token_packet_ids.append(pkt_idx)
            # Convert direction: -1 -> 0 (downlink), 1 -> 2 (uplink)
            dir_idx = 0 if direction == -1 else 2
            token_directions.append(dir_idx)

    # Reserve space for [CLS] and [SEP]
    max_tokens = seq_length_raw - 2

    # Truncate if needed
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        token_packet_ids = token_packet_ids[:max_tokens]
        token_directions = token_directions[:max_tokens]

    # Build sequence: [CLS] + tokens + [SEP]
    cls_id = vocab.get(CLS_TOKEN, 0)
    sep_id = vocab.get(SEP_TOKEN, 0)

    src = [cls_id] + tokens + [sep_id]
    packet_ids = [8] + token_packet_ids + [8]  # 8 for special tokens
    directions = [1] + token_directions + [1]  # 1 for neutral

    # Pad to seq_length_raw
    while len(src) < seq_length_raw:
        src.append(PAD_ID)
        packet_ids.append(8)
        directions.append(1)

    return src, packet_ids, directions


def tokenize_size_flow(flow_data, tokenizer_size, seq_length_size):
    """
    Tokenize packet size data for a single flow

    Format matches pretraining (data.py PacketSizeDataset):
        - [CLS] + size_tokens + [SEP] + [PAD]...
        - Size token = size * direction + 1500 (direction already encoded)

    Args:
        flow_data: dict with 'packet_sizes' (List[int]) and 'size_directions' (List[int])
        tokenizer_size: BertTokenizer for size modality
        seq_length_size: target sequence length

    Returns:
        src: List[int] - token IDs
    """
    vocab = tokenizer_size.vocab

    # Build size token string: "1672 2185 953 ..."
    size_tokens_str = []
    for size, direction in zip(flow_data['packet_sizes'], flow_data['size_directions']):
        # Size token = size * direction + 1500
        size_token = size * direction + 1500
        size_tokens_str.append(str(size_token))

    # Tokenize the size string
    # This matches pretraining: tokenizer.tokenize(tokens_line)
    size_str = ' '.join(size_tokens_str)
    tokens = tokenizer_size.tokenize(size_str)
    token_ids = tokenizer_size.convert_tokens_to_ids(tokens)

    # Reserve space for [CLS] and [SEP]
    max_tokens = seq_length_size - 2

    # Truncate if needed
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]

    # Build sequence: [CLS] + tokens + [SEP]
    cls_id = vocab.get(CLS_TOKEN, 0)
    sep_id = vocab.get(SEP_TOKEN, 0)

    src = [cls_id] + token_ids + [sep_id]

    # Pad to seq_length_size
    while len(src) < seq_length_size:
        src.append(PAD_ID)

    return src


def process_dataset(pcap_dir, tokenizer_raw, tokenizer_size,
                    seq_length_raw=512, seq_length_size=256,
                    bytes_per_packet=64, max_raw_packets=8, max_size_packets=256,
                    min_samples_per_class=10, max_samples_per_class=500,
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                    seed=42):
    """
    Process all PCAP files in dataset directory

    Args:
        pcap_dir: Root directory with structure: pcap_dir/label_name/*.pcap
        tokenizer_raw: BertTokenizer for raw modality
        tokenizer_size: BertTokenizer for size modality
        ...

    Returns:
        train_data, val_data, test_data: Lists of samples
        label2id: Label name to ID mapping
    """
    random.seed(seed)
    np.random.seed(seed)

    extractor = MultiModalFlowExtractor(
        bytes_per_packet=bytes_per_packet,
        max_raw_packets=max_raw_packets,
        max_size_packets=max_size_packets
    )

    # Collect all samples per label
    label_samples = defaultdict(list)

    # Walk through directory structure
    print(f"Scanning {pcap_dir}...")
    for label_name in os.listdir(pcap_dir):
        label_path = os.path.join(pcap_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        pcap_files = [f for f in os.listdir(label_path)
                      if f.endswith('.pcap') or f.endswith('.pcapng')]

        print(f"Processing label '{label_name}': {len(pcap_files)} files")

        for pcap_file in tqdm(pcap_files, desc=label_name):
            pcap_path = os.path.join(label_path, pcap_file)
            flow_data = extractor.extract_pcap(pcap_path)

            if flow_data is None:
                continue

            # Tokenize raw modality (using tokenizer, same as pretraining)
            raw_src, packet_ids, directions = tokenize_raw_flow(
                flow_data, tokenizer_raw, seq_length_raw
            )

            # Tokenize size modality (using tokenizer, same as pretraining)
            size_src = tokenize_size_flow(
                flow_data, tokenizer_size, seq_length_size
            )

            sample = {
                'raw_src': raw_src,
                'packet_ids': packet_ids,
                'directions': directions,
                'size_src': size_src,
                'label_name': label_name
            }
            label_samples[label_name].append(sample)

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

    for label_name in valid_labels:
        samples = label_samples[label_name]

        # Limit max samples
        if len(samples) > max_samples_per_class:
            samples = random.sample(samples, max_samples_per_class)

        # Shuffle
        random.shuffle(samples)

        # Add label ID
        label_id = label2id[label_name]
        for s in samples:
            s['label'] = label_id

        # Split
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_data.extend(samples[:n_train])
        val_data.extend(samples[n_train:n_train + n_val])
        test_data.extend(samples[n_train + n_val:])

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
                        help='Root directory with structure: pcap_dir/label_name/*.pcap')
    parser.add_argument('--vocab_path_raw', type=str, required=True,
                        help='Path to raw modality vocabulary file')
    parser.add_argument('--vocab_path_size', type=str, required=True,
                        help='Path to size modality vocabulary file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed datasets')

    # Sequence lengths
    parser.add_argument('--seq_length_raw', type=int, default=512)
    parser.add_argument('--seq_length_size', type=int, default=256)

    # Feature extraction params
    parser.add_argument('--bytes_per_packet', type=int, default=64)
    parser.add_argument('--max_raw_packets', type=int, default=8)
    parser.add_argument('--max_size_packets', type=int, default=256)

    # Dataset filtering
    parser.add_argument('--min_samples_per_class', type=int, default=10)
    parser.add_argument('--max_samples_per_class', type=int, default=500)

    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create tokenizers (same as pretraining)
    print("Creating tokenizers...")
    tokenizer_raw = create_tokenizer(args.vocab_path_raw)
    tokenizer_size = create_tokenizer(args.vocab_path_size)
    print(f"  Raw vocab size: {len(tokenizer_raw.vocab)}")
    print(f"  Size vocab size: {len(tokenizer_size.vocab)}")

    # Process dataset
    train_data, val_data, test_data, label2id = process_dataset(
        pcap_dir=args.pcap_dir,
        tokenizer_raw=tokenizer_raw,
        tokenizer_size=tokenizer_size,
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
        seed=args.seed
    )

    # Save datasets
    save_dataset(train_data, os.path.join(args.output_dir, 'train.pkl'))
    save_dataset(val_data, os.path.join(args.output_dir, 'val.pkl'))
    save_dataset(test_data, os.path.join(args.output_dir, 'test.pkl'))
    save_dataset(label2id, os.path.join(args.output_dir, 'label2id.pkl'))

    print("\nDone!")


if __name__ == '__main__':
    main()

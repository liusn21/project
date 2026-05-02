"""
Multi-Modal Traffic Data Extractor (UER-compatible) with IAT temporal information.

Output text format compatible with uer/preprocess.py:
- Raw packet: byte-level hex tokens (one byte per token, space-separated)
- Packet size: integer tokens (space-separated)
- Direction: encoded into the size token, or written separately for raw
- IAT (Inter-Arrival Time): temporal tokens normalized and discretized to [0, 999]

Files produced:
- corpus_raw.txt:  raw packet bytes
- corpus_size.txt: packet sizes + IAT temporal tokens

Streaming processing:
- Uses scapy.PcapReader to stream packets from each PCAP (no full-file load)
- Each flow is written to disk immediately (no in-memory accumulation)
- Suitable for very large pcap directories without OOM
"""

import os
import math
from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse


class MultiModalExtractor:

    def __init__(self, bytes_per_packet=64, max_raw_packets=8):
        self.bytes_per_packet = bytes_per_packet
        self.max_raw_packets = max_raw_packets

    def extract_pcap(self, pcap_path):
        """Extract features from a PCAP, streaming with PcapReader to avoid OOM."""
        # Parse the 5-tuple from the filename.
        tuple_info = self.parse_filename_5tuple(pcap_path)
        if tuple_info is None:
            return None

        protocol, src_ip, src_port, dst_ip, dst_port = tuple_info
        flow_src = (src_ip, src_port)

        raw_bytes_per_pkt = []  # List[List[str]] - byte-level hex tokens per packet
        raw_directions = []
        packet_sizes = []  # List[int]
        size_directions = []
        timestamps = []  # List[float] - packet timestamps

        try:
            with PcapReader(pcap_path) as pcap_reader:
                for packet in pcap_reader:
                    payload = self._extract_payload(packet, protocol)
                    if payload is None or len(payload) == 0:
                        continue

                    payload_len = min(len(payload), 1500)
                    direction = self._get_direction(packet, protocol, flow_src)
                    timestamp = float(packet.time)

                    # Raw modality: byte-level hex tokens (first max_raw_packets only).
                    if len(raw_bytes_per_pkt) < self.max_raw_packets:
                        byte_hex_list = self._bytes_to_hex_tokens(payload[:self.bytes_per_packet])
                        raw_bytes_per_pkt.append(byte_hex_list)
                        raw_directions.append(direction)

                    # Size modality.
                    packet_sizes.append(payload_len)
                    size_directions.append(direction)
                    timestamps.append(timestamp)
        except Exception:
            return None

        if len(packet_sizes) == 0:
            return None

        # Compute IAT tokens (normalized and discretized).
        iat_tokens = self._compute_iat_tokens(timestamps)

        return {
            'protocol': protocol,
            'protocol_name': 'TCP' if protocol == 6 else 'UDP',
            'raw_bytes_per_pkt': raw_bytes_per_pkt,  # List[List[str]]
            'raw_directions': raw_directions,        # [1, -1, 1, -1, ...]
            'packet_sizes': packet_sizes,
            'size_directions': size_directions,      # [1, -1, 1, -1, ...]
            'iat_tokens': iat_tokens                 # List[int] - IAT tokens (0-999)
        }

    def _bytes_to_hex_tokens(self, payload_bytes):
        """Byte-level tokenization: N bytes -> N tokens, e.g. ["45", "00", "06", ...]"""
        return [f"{b:02x}" for b in payload_bytes]

    def _compute_iat_tokens(self, timestamps):
        """
        Compute IAT (Inter-Arrival Time) tokens.

        Steps:
        1. Compute the time delta between consecutive packets (seconds).
        2. Normalize: sigmoid(log10(IAT + epsilon)).
        3. Discretize into 1000 bins (0..999).

        Args:
            timestamps: List[float] - packet timestamps (seconds).

        Returns:
            List[int]: IAT tokens (0..999).
        """
        if len(timestamps) == 0:
            return []

        iat_tokens = []
        epsilon = 1e-6

        for i in range(len(timestamps)):
            if i == 0:
                iat = epsilon
            else:
                iat = timestamps[i] - timestamps[i - 1]
                iat = max(iat, epsilon)  # Ensure non-negative and non-zero.

            # Normalization: sigmoid(log10(IAT)).
            log_iat = math.log10(iat)
            normalized = 1.0 / (1.0 + math.exp(-log_iat))

            # Discretize into [0, 999].
            token = int(normalized * 1000)
            token = min(max(token, 0), 999)
            iat_tokens.append(token)

        return iat_tokens

    def parse_filename_5tuple(self, pcap_path):
        """Parse the 5-tuple from the filename."""
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
        """Determine packet direction (+1 upstream, -1 downstream)."""
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
            return 1   # upstream
        else:
            return -1  # downstream

    def _extract_payload(self, packet, protocol):
        """Extract the transport-layer payload."""
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


def write_flow(flow, f_raw, f_size):
    """
    Stream a single flow to the output files (no in-memory accumulation).
    """
    protocol = flow['protocol']

    # Raw packet corpus.
    f_raw.write("||\n")
    f_raw.write(str(protocol))
    f_raw.write("\n")
    for i in range(len(flow['raw_bytes_per_pkt'])):
        f_raw.write(f"{flow['raw_directions'][i]} ")
        f_raw.write(" ".join(flow['raw_bytes_per_pkt'][i]))
        f_raw.write("\n")

    # Size corpus (with IAT temporal information).
    f_size.write("||\n")
    f_size.write(str(protocol))
    f_size.write("\n")
    # Line 1: size tokens (direction encoded: size_token = size * direction + 1500).
    size_tokens = []
    for size, direction in zip(flow['packet_sizes'], flow['size_directions']):
        size_token = size * direction + 1500
        size_tokens.append(str(size_token))
    f_size.write(" ".join(size_tokens))
    f_size.write("\n")
    # Line 2: IAT temporal tokens (0..999).
    iat_token_strs = [str(token) for token in flow['iat_tokens']]
    f_size.write(" ".join(iat_token_strs))
    f_size.write("\n")


def process_single_pcap(args_tuple):
    """Worker function for processing a single PCAP (multiprocessing entry point)."""
    pcap_path, bytes_per_packet, max_raw_packets = args_tuple
    extractor = MultiModalExtractor(
        bytes_per_packet=bytes_per_packet,
        max_raw_packets=max_raw_packets,
    )
    return extractor.extract_pcap(pcap_path)


def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Data Extractor (UER-compatible)')

    parser.add_argument('--pcap_dir', type=str, required=True)
    parser.add_argument('--output_raw', type=str, required=True,
                        help='Output raw packet corpus (text)')
    parser.add_argument('--output_size', type=str, required=True,
                        help='Output packet size corpus (text)')
    parser.add_argument('--bytes_per_packet', type=int, default=64)
    parser.add_argument('--max_raw_packets', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')

    args = parser.parse_args()

    print(f"Extracting from: {args.pcap_dir}")

    # Collect PCAP file paths (paths only; do not open them yet).
    pcap_files = []
    for root, dirs, files in os.walk(args.pcap_dir):
        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                pcap_files.append(os.path.join(root, file))

    print(f"Found {len(pcap_files)} PCAP files")

    # Worker count.
    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"Using {num_workers} worker processes")

    # Multiprocessing arguments.
    task_args = [(pcap_file, args.bytes_per_packet, args.max_raw_packets)
                 for pcap_file in pcap_files]

    # Ensure output directories exist.
    for path in [args.output_raw, args.output_size]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Streaming: imap_unordered yields one result at a time and we flush to disk
    # immediately, so memory usage is bounded by the worker pool buffer.
    total_flows = 0
    failed = 0

    with open(args.output_raw, 'w') as f_raw, \
         open(args.output_size, 'w') as f_size, \
         Pool(num_workers) as pool:

        for flow in tqdm(
            pool.imap_unordered(process_single_pcap, task_args),
            total=len(pcap_files),
            desc="Processing"
        ):
            if flow is None:
                failed += 1
                continue

            write_flow(flow, f_raw, f_size)
            total_flows += 1

    print(f"\nExtracted {total_flows} flows ({failed} failed)")


if __name__ == '__main__':
    main()

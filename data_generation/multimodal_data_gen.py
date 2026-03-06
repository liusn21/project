"""
Multi-Modal Traffic Data Extractor V3 (uer框架兼容版 + IAT Temporal Information)

输出文本格式，兼容uer的preprocess.py：
- Raw packet: bigram文本（hex，空格分隔）
- Packet size: 整数文本（空格分隔）
- Direction: 编码在size中，或单独输出
- IAT (Inter-Arrival Time): 时序信息（归一化后离散化到0-999）

输出格式：
- corpus_raw.txt: raw packet bigrams
- corpus_size.txt: packet sizes + IAT temporal tokens

支持流式处理：
- 使用 PcapReader 流式读取单个 PCAP（不一次性加载整个文件）
- 处理结果立即写入磁盘，不在内存中累积
- 适用于超大 pcap_dir，不会 OOM
"""

import os
import math
from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse


class MultiModalExtractorV3:

    def __init__(self, bytes_per_packet=64, max_raw_packets=8, raw_token_type='bigram'):
        self.bytes_per_packet = bytes_per_packet
        self.max_raw_packets = max_raw_packets
        self.raw_token_type = raw_token_type

    def extract_pcap(self, pcap_path):
        """从PCAP提取特征，使用 PcapReader 流式读取避免 OOM"""
        # 从文件名解析5元组
        tuple_info = self.parse_filename_5tuple(pcap_path)
        if tuple_info is None:
            return None

        protocol, src_ip, src_port, dst_ip, dst_port = tuple_info
        flow_src = (src_ip, src_port)

        raw_bigrams = []  # List[List[str]] - hex bigrams per packet
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

                    # Raw modality: hex tokens (first max_raw_packets)
                    if len(raw_bigrams) < self.max_raw_packets:
                        bigram_hex_list = self._bytes_to_hex_tokens(payload[:self.bytes_per_packet])
                        raw_bigrams.append(bigram_hex_list)
                        raw_directions.append(direction)

                    # Size modality
                    packet_sizes.append(payload_len)
                    size_directions.append(direction)
                    timestamps.append(timestamp)
        except Exception:
            return None

        if len(packet_sizes) == 0:
            return None

        # Compute IAT tokens (normalized and discretized)
        iat_tokens = self._compute_iat_tokens(timestamps)

        return {
            'protocol': protocol,
            'protocol_name': 'TCP' if protocol == 6 else 'UDP',
            'raw_bigrams': raw_bigrams,  # List[List[str]]
            'raw_directions': raw_directions, # [1,-1,1,-1...]
            'packet_sizes': packet_sizes,
            'size_directions': size_directions, #[1,-1,1,-1...]
            'iat_tokens': iat_tokens  # List[int] - IAT temporal tokens (0-999)
        }

    def _bytes_to_hex_tokens(self, payload_bytes):
        """
        将bytes转换为hex token列表

        bigram mode: ["4500", "0006", ...] (滑窗bigram, N bytes → N-1 tokens)
        byte mode:   ["45", "00", "06", ...] (单字节, N bytes → N tokens)
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
        计算IAT（Inter-Arrival Time）tokens

        参考PTU论文的方法：
        1. 计算相邻包时间差（秒）
        2. 归一化：sigmoid(log10(IAT + epsilon))
        3. 离散化到1000个bins（0-999）

        Args:
            timestamps: List[float] - 数据包时间戳（秒）

        Returns:
            List[int]: IAT tokens (0-999)
        """
        if len(timestamps) == 0:
            return []

        iat_tokens = []
        epsilon = 1e-6

        for i in range(len(timestamps)):
            if i == 0:
                iat = epsilon
            else:
                # 计算与前一个包的时间差
                iat = timestamps[i] - timestamps[i-1]
                iat = max(iat, epsilon)  # 确保非负且非零

            # normalization: sigmoid(log10(IAT))
            # sigmoid(x) = 1 / (1 + exp(-x))
            log_iat = math.log10(iat)
            normalized = 1.0 / (1.0 + math.exp(-log_iat))

            # 离散化到 [0, 999]
            token = int(normalized * 1000)
            token = min(max(token, 0), 999)  # Clip to [0, 999]

            iat_tokens.append(token)

        return iat_tokens

    def parse_filename_5tuple(self, pcap_path):
        """从文件名解析5元组"""
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
        """判断包方向"""
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
            return 1  # 上行
        else:
            return -1  # 下行

    def _extract_payload(self, packet, protocol):
        """提取transport layer payload"""
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
    将单个 flow 写入输出文件（流式写入，不累积内存）

    格式与原 save_to_text_format 完全一致。
    """
    protocol = flow['protocol']

    # Raw packet corpus
    f_raw.write("||\n")
    f_raw.write(str(protocol))
    f_raw.write("\n")
    for i in range(len(flow['raw_bigrams'])):
        f_raw.write(f"{flow['raw_directions'][i]} ")
        f_raw.write(" ".join(flow['raw_bigrams'][i]))
        f_raw.write("\n")

    # Size corpus (with IAT temporal information)
    f_size.write("||\n")
    f_size.write(str(protocol))
    f_size.write("\n")
    # Line 1: size tokens (direction encoded: size_token = size * direction + 1500)
    size_tokens = []
    for size, direction in zip(flow['packet_sizes'], flow['size_directions']):
        size_token = size * direction + 1500
        size_tokens.append(str(size_token))
    f_size.write(" ".join(size_tokens))
    f_size.write("\n")
    # Line 2: IAT temporal tokens (0-999)
    iat_token_strs = [str(token) for token in flow['iat_tokens']]
    f_size.write(" ".join(iat_token_strs))
    f_size.write("\n")


def process_single_pcap(args_tuple):
    """单个PCAP文件处理函数，用于多进程"""
    pcap_path, bytes_per_packet, max_raw_packets, raw_token_type = args_tuple
    extractor = MultiModalExtractorV3(
        bytes_per_packet=bytes_per_packet,
        max_raw_packets=max_raw_packets,
        raw_token_type=raw_token_type
    )
    return extractor.extract_pcap(pcap_path)


def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Data Extractor V3 (uer compatible)')

    parser.add_argument('--pcap_dir', type=str, required=True)
    parser.add_argument('--output_raw', type=str, required=True,
                       help='Output raw packet corpus (text)')
    parser.add_argument('--output_size', type=str, required=True,
                       help='Output packet size corpus (text)')
    parser.add_argument('--bytes_per_packet', type=int, default=64)
    parser.add_argument('--max_raw_packets', type=int, default=8)
    parser.add_argument('--raw_token_type', type=str, choices=['bigram', 'byte'], default='bigram',
                       help='Raw tokenization: bigram (65K vocab, default) or byte (256 vocab)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count)')

    args = parser.parse_args()

    print(f"Extracting from: {args.pcap_dir}")

    # 收集PCAP文件路径（只存路径字符串，不占内存）
    pcap_files = []
    for root, dirs, files in os.walk(args.pcap_dir):
        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                pcap_files.append(os.path.join(root, file))

    print(f"Found {len(pcap_files)} PCAP files")

    # 设置进程数
    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"Using {num_workers} worker processes")

    # 准备多进程参数
    task_args = [(pcap_file, args.bytes_per_packet, args.max_raw_packets, args.raw_token_type)
                 for pcap_file in pcap_files]

    # 确保输出目录存在
    for path in [args.output_raw, args.output_size]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # 流式处理：imap_unordered 逐个返回结果，立即写入磁盘
    # 内存中同时只有 ~num_workers 个 flow 结果（进程池缓冲区）
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

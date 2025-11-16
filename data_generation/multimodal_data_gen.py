"""
Multi-Modal Traffic Data Extractor for Pre-training

从 PCAP 文件提取多模态特征：
1. Raw packet tokens (hex bigram)
2. Temporal tokens (Inter-Arrival Time)
3. Size+Direction tokens

输出格式：JSON
Author: Based on TrafficFormer, extended for multi-modal learning
"""

import os
import json
import numpy as np
import scapy.all as scapy
from flowcontainer.extractor import extract
from tqdm import tqdm
import binascii


class MultiModalExtractor:
    """多模态流量特征提取器"""

    def __init__(self,
                 bytes_per_packet=64,
                 start_index=28,
                 min_packets=5,
                 temporal_precision=3):
        """
        Args:
            bytes_per_packet: 每个包提取的字节数（从IP层开始）
            start_index: 起始索引（28=跳过Ethernet 14B * 2）
            min_packets: 流的最小包数（<5的流丢弃）
            temporal_precision: 时间戳精度（Lt，默认3位小数）
        """
        self.bytes_per_packet = bytes_per_packet
        self.start_index = start_index
        self.min_packets = min_packets
        self.temporal_precision = temporal_precision
        self.max_temporal_tokens = 10 

    def extract_pcap(self, pcap_path):
        """
        从单个 PCAP 文件提取所有流的特征

        Returns:
            flows: List[Dict] - 每个流的特征字典
        """
        try:
            packets = scapy.rdpcap(pcap_path)
        except Exception as e:
            print(f"Failed to read {pcap_path}: {e}")
            return []

        if len(packets) == 0:
            return []

        # 检查是否有以太网头
        no_ether = not hasattr(packets[0], 'type')

        # 检查是否是 IPv6（暂不处理）
        if not no_ether and packets[0].type == 0x86dd:
            return []
        if no_ether and packets[0].version == 6:
            return []

        # 提取流方向信息
        packet_directions = self._extract_directions(pcap_path, len(packets))
        if not packet_directions:
            return []

        # 按流聚合数据包
        flows = self._aggregate_by_flow(packets, packet_directions, no_ether)

        # 过滤包数过少的流
        flows = [f for f in flows if len(f['packets']) >= self.min_packets]

        # 提取特征
        flow_features = []
        for flow in flows:
            features = self._extract_flow_features(flow)
            if features:
                flow_features.append(features)

        return flow_features

    def _extract_directions(self, pcap_path, expected_len):
        """提取包的方向信息（1=正向，-1=反向）"""
        try:
            feature_result = extract(pcap_path)
            for key in feature_result.keys():
                value = feature_result[key]
                directions = [x // abs(x) if x != 0 else 1
                             for x in value.ip_lengths]
                if len(directions) == expected_len:
                    return directions
        except Exception as e:
            print(f"Direction extraction failed: {e}")
        return None

    def _aggregate_by_flow(self, packets, directions, no_ether):
        """按流ID聚合数据包"""
        flows = {}

        for i, packet in enumerate(packets):
            direction = directions[i]

            # 提取流标识符（简化版：使用包索引模拟）
            # 实际应该根据 5-tuple 聚合
            flow_id = 0  # 简化：所有包属于同一流

            if flow_id not in flows:
                flows[flow_id] = {
                    'packets': [],
                    'directions': [],
                    'timestamps': [],
                    'protocol': self._get_protocol(packet, no_ether)
                }

            flows[flow_id]['packets'].append(packet)
            flows[flow_id]['directions'].append(direction)

            # 提取时间戳
            ts = packet.time if hasattr(packet, 'time') else 0
            flows[flow_id]['timestamps'].append(ts)

        return list(flows.values())

    def _get_protocol(self, packet, no_ether):
        """获取协议类型（TCP=6, UDP=17）"""
        if no_ether:
            if packet.version == 4:
                return packet.proto
        else:
            if packet.type == 0x0800:  # IPv4
                return packet.payload.proto
        return 0

    def _extract_flow_features(self, flow):
        """提取单个流的所有特征"""
        packets = flow['packets']
        directions = flow['directions']
        timestamps = flow['timestamps']

        if len(packets) < self.min_packets:
            return None

        raw_tokens_list = []
        temporal_tokens_list = []
        size_tokens_list = []

        for i, packet in enumerate(packets):
            # 1. Raw packet tokens (bigram)
            raw_hex = self._packet_to_hex(packet)
            if not raw_hex:
                continue
            raw_bigrams = self._hex_to_bigram(raw_hex)
            raw_tokens_list.append(raw_bigrams)

            # 2. Temporal tokens (IAT)
            temporal_tokens = self._compute_temporal_tokens(timestamps, i)
            temporal_tokens_list.append(temporal_tokens)

            # 3. Size + Direction token
            size = len(packet)
            direction = directions[i]
            size_token = self._encode_size_direction(size, direction)
            size_tokens_list.append(size_token)

        return {
            'flow_id': f"flow_{id(flow)}",
            'protocol': flow['protocol'],
            'num_packets': len(packets),
            'raw_tokens': raw_tokens_list,
            'temporal_tokens': temporal_tokens_list,
            'size_tokens': size_tokens_list,
            'directions': directions
        }

    def _packet_to_hex(self, packet):
        """将数据包转为hex字符串（从IP层开始）"""
        try:
            data = binascii.hexlify(bytes(packet)).decode()
            # 从 start_index 开始，提取 bytes_per_packet * 2 个字符
            extracted = data[self.start_index:self.start_index + self.bytes_per_packet * 2]
            return extracted
        except:
            return None

    def _hex_to_bigram(self, hex_string):
        """
        将hex字符串转为bigram tokens
        例如: "4500003c" → ["4500", "5000", "0000", "003c"]
        """
        bigrams = []
        for i in range(0, len(hex_string) - 1, 2):
            if i + 3 < len(hex_string):
                bigram = hex_string[i:i+4]
                bigrams.append(bigram)
        return bigrams

    def _compute_temporal_tokens(self, timestamps, current_idx):
        """
        计算当前包的 temporal tokens（基于PTU方案）

        对于包n，取前 min(n, 10) 个包的 IAT：
        IAT_i = timestamps[i+1] - timestamps[i]
        temporal_value = sigmoid(log10(IAT))
        temporal_token = int(temporal_value * 10^Lt)

        Returns:
            List[int]: temporal token IDs
        """
        temporal_tokens = []

        # 取前 min(current_idx, max_temporal_tokens) 个包的IAT
        start_idx = max(0, current_idx - self.max_temporal_tokens)

        for i in range(start_idx, current_idx):
            iat = timestamps[i+1] - timestamps[i]
            if iat <= 0:
                iat = 1e-6  # 避免log(0)

            # PTU公式: sigmoid(log10(IAT))
            temporal_value = self._sigmoid(np.log10(iat))

            # 转为离散token: [0, 999]
            token_id = int(temporal_value * (10 ** self.temporal_precision))
            token_id = min(token_id, 10 ** self.temporal_precision - 1)

            temporal_tokens.append(token_id)

        # 如果是第一个包，添加一个特殊的temporal token (0)
        if current_idx == 0:
            temporal_tokens.append(0)

        return temporal_tokens

    def _sigmoid(self, x):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-x))

    def _encode_size_direction(self, size, direction):
        """
        编码 Size + Direction 为单个token

        size: [0, 1500] (MTU)
        direction: {1, -1}

        token = size * direction + 1500
        范围: [0, 3000]
        """
        size = min(size, 1500)  # 截断到MTU
        token_id = size * direction + 1500
        return token_id

    def process_directory(self, pcap_dir, output_json):
        """
        处理整个目录的PCAP文件

        Args:
            pcap_dir: PCAP文件目录
            output_json: 输出的JSON文件路径
        """
        all_flows = []

        # 遍历目录下所有PCAP文件
        pcap_files = []
        for root, dirs, files in os.walk(pcap_dir):
            for file in files:
                if file.endswith('.pcap') or file.endswith('.pcapng'):
                    pcap_files.append(os.path.join(root, file))

        print(f"Found {len(pcap_files)} PCAP files")

        for pcap_path in tqdm(pcap_files, desc="Processing PCAPs"):
            flows = self.extract_pcap(pcap_path)
            all_flows.extend(flows)

        # 保存为JSON
        output_data = {
            'num_flows': len(all_flows),
            'flows': all_flows
        }

        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Extracted {len(all_flows)} flows")
        print(f"Saved to {output_json}")

        return all_flows


def main():
    """示例用法"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcap_dir', type=str, required=True,
                       help='Directory containing PCAP files')
    parser.add_argument('--output_json', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--bytes_per_packet', type=int, default=64,
                       help='Bytes to extract per packet')
    parser.add_argument('--min_packets', type=int, default=5,
                       help='Minimum packets per flow')

    args = parser.parse_args()

    extractor = MultiModalExtractor(
        bytes_per_packet=args.bytes_per_packet,
        min_packets=args.min_packets
    )

    extractor.process_directory(args.pcap_dir, args.output_json)


if __name__ == '__main__':
    main()

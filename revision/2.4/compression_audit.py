#!/usr/bin/env python3
"""Audit confirmed wire-visible compression in a fine-tuning PCAP dataset.

The input layout is the same as ``fine-tuning/multimodal_data_utils.py``::

    DATASET/
      label_a/*.pcap
      label_b/*.pcap

Each PCAP is treated as one bidirectional flow/sample.  The audit deliberately
reports a conservative lower bound: a flow is "confirmed compressed" only when
a detector admitted by the selected ``--compression-level`` produces exact
wire-byte regions.  Encryption and unsupported evidence are never silently
treated as uncompressed.

Compression levels are cumulative:

* level 1 preserves the original strict HTTP/gzip/ZIP/PNG/JPEG/WebP scope;
* level 2 adds fully validated zlib, BZIP2, XZ, GIF, WOFF1, and CWS objects;
* level 3 adds exactly bounded HTTP br/zstd/compress declarations, declared ZIP
  methods, and structurally complete Zstandard/LZ4 compressed blocks.

The default is level 1 so existing commands and strict results remain stable.

The content window exactly follows the default fine-tuning path: the first eight
payload-bearing TCP/UDP packets, at most 64 payload bytes per packet, concatenated
in capture order and truncated to 510 payload bytes (512 positions minus CLS/SEP).

Outputs:

* ``dataset_summary.csv``: one aggregate row for the input dataset;
* ``exposure_bins.csv``: counts and mean entropy for the four e_i bins;
* ``flow_details.csv``: auditable per-PCAP results;
* ``audit_metadata.json``: detector scope and exact metric definitions.

Example::

    python revision/compression_audit.py /data/Browser \
        --output-dir /results/Browser_compression_audit \
        --compression-level 2 \
        --workers 8

This script is an offline audit utility.  It does not modify the input dataset.
"""

from __future__ import annotations

import argparse
import bisect
import binascii
import bz2
import csv
import io
import json
import lzma
import math
import os
import re
import statistics
import struct
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP


SCRIPT_VERSION = "1.3.0"

BYTES_PER_PACKET = 64
MAX_RAW_PACKETS = 8
RAW_SEQUENCE_LENGTH = 512
MAX_WINDOW_BYTES = RAW_SEQUENCE_LENGTH - 2

# models/bert/vocab_raw.txt is [PAD], [UNK], [CLS], [SEP], [MASK], 00..ff.
RAW_VOCAB_SIZE = 261
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
BYTE_TOKEN_OFFSET = 5

DEFAULT_MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_HTTP_HEADER_BYTES = 64 * 1024
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
DEFAULT_COMPRESSION_LEVEL = 1
COMPRESSION_LEVELS = (1, 2, 3)
PENDING_TASKS_PER_WORKER = 2
DECODE_PROBE_CHUNK_BYTES = 4 * 1024
DECODE_INPUT_CHUNK_BYTES = 64 * 1024

TCP_SEQUENCE_MODULUS = 1 << 32
TCP_SEQUENCE_MASK = TCP_SEQUENCE_MODULUS - 1

SUPPORTED_HTTP_CODINGS = {"gzip", "x-gzip", "deflate", "x-deflate"}
DECLARED_HTTP_COMPRESSION_CODINGS = {
    *SUPPORTED_HTTP_CODINGS,
    "br",
    "zstd",
    "compress",
    "x-compress",
}
NON_COMPRESSING_HTTP_CODINGS = {"identity", "chunked"}

ZIP_SUPPORTED_METHODS = {
    zipfile.ZIP_DEFLATED,
    zipfile.ZIP_BZIP2,
    zipfile.ZIP_LZMA,
}

# APPNOTE compression method identifiers.  Method 99 (AES) and encrypted
# entries are deliberately excluded: their wire representation is ciphertext,
# not observable compressed bytes.
ZIP_DECLARED_COMPRESSION_METHODS = {
    1,              # Shrunk
    2, 3, 4, 5,     # Reduced
    6,              # Imploded
    8,              # Deflate
    9,              # Deflate64
    10,             # Old IBM TERSE
    12,             # BZIP2
    14,             # LZMA
    16,             # IBM z/OS CMPSC
    18,             # IBM TERSE
    19,             # IBM LZ77
    20, 93,         # Zstandard (deprecated/current identifiers)
    94,             # MP3
    95,             # XZ
    96,             # JPEG
    97,             # WavPack
    98,             # PPMd
}

ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"
LZ4_FRAME_MAGIC = b"\x04\x22\x4d\x18"
XZ_STREAM_MAGIC = b"\xfd7zXZ\x00"
ZLIB_NO_DICTIONARY_HEADERS = tuple(
    bytes((cmf, flags))
    for cmf in range(0x08, 0x79, 0x10)
    for flags in range(256)
    if ((cmf << 8) | flags) % 31 == 0 and not (flags & 0x20)
)
ZLIB_HEADER_PATTERN = re.compile(
    b"(?=(?:" + b"|".join(
        re.escape(header) for header in ZLIB_NO_DICTIONARY_HEADERS
    ) + b"))"
)

PCAP_SUFFIXES = {".pcap", ".pcapng"}


class DetectionError(Exception):
    """A candidate could not be strictly validated."""


class DecompressionLimitError(DetectionError):
    """Decoded output exceeded the configured safety limit."""


@dataclass(frozen=True)
class WindowByteRef:
    """One payload byte that reaches the model's raw-content window."""

    carrier: str                 # "tcp" or "udp"
    direction: int               # +1 filename direction, -1 reverse
    position: int                # absolute TCP seq or UDP payload offset
    packet_index: Optional[int]
    value: int
    canonical: bool = True       # False for conflicting TCP overlap bytes


@dataclass(frozen=True)
class ByteRegion:
    """Half-open byte interval validated as compressed wire representation."""

    carrier: str
    direction: int
    start: int
    end: int
    packet_index: Optional[int] = None

    def contains(self, ref: WindowByteRef) -> bool:
        if not ref.canonical or self.carrier != ref.carrier:
            return False
        if self.direction != ref.direction:
            return False
        if self.carrier == "udp" and self.packet_index != ref.packet_index:
            return False
        return self.start <= ref.position < self.end

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "carrier": self.carrier,
            "direction": self.direction,
            "start": self.start,
            "end": self.end,
        }
        if self.packet_index is not None:
            result["packet_index"] = self.packet_index
        return result


@dataclass
class CompressionEvidence:
    category: str                # "protocol" or "intrinsic"
    kind: str                    # HTTP coding or intrinsic format
    source: str
    regions: List[ByteRegion]
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "kind": self.kind,
            "source": self.source,
            "regions": [region.to_dict() for region in self.regions],
            "detail": self.detail,
        }


@dataclass(frozen=True)
class StreamPiece:
    start: int
    data: bytes

    @property
    def end(self) -> int:
        return self.start + len(self.data)


@dataclass(frozen=True)
class SourceBuffer:
    """A contiguous byte buffer with coordinates in the original capture."""

    carrier: str
    direction: int
    start: int
    data: bytes
    source_id: str
    packet_index: Optional[int] = None
    closed_at_end: bool = False

    def region(self, local_start: int, local_end: int) -> ByteRegion:
        if not (0 <= local_start < local_end <= len(self.data)):
            raise ValueError(
                f"invalid region [{local_start}, {local_end}) for {self.source_id} "
                f"with length {len(self.data)}"
            )
        return ByteRegion(
            carrier=self.carrier,
            direction=self.direction,
            start=self.start + local_start,
            end=self.start + local_end,
            packet_index=self.packet_index,
        )


class TcpDirectionState:
    """First-seen TCP reassembly with stable sequence coordinates.

    Pieces remain non-overlapping.  Keeping them separate during ingestion avoids
    repeatedly copying an ever-growing stream for ordinary sequential traffic.
    """

    def __init__(self, direction: int) -> None:
        self.direction = direction
        self.pieces: List[StreamPiece] = []
        self.starts: List[int] = []
        self.unwrap_reference: Optional[int] = None
        self.fin_positions: Set[int] = set()

    def _unwrap(self, sequence_32: int) -> int:
        sequence_32 &= TCP_SEQUENCE_MASK
        if self.unwrap_reference is None:
            return sequence_32

        base = self.unwrap_reference & ~TCP_SEQUENCE_MASK
        candidates = (
            base + sequence_32,
            base + sequence_32 - TCP_SEQUENCE_MODULUS,
            base + sequence_32 + TCP_SEQUENCE_MODULUS,
        )
        return min(candidates, key=lambda value: abs(value - self.unwrap_reference))

    def observe(
        self,
        sequence_32: int,
        payload: bytes,
        syn: bool,
        fin: bool,
    ) -> int:
        header_sequence = self._unwrap(sequence_32)
        data_start = header_sequence + (1 if syn else 0)
        data_end = data_start + len(payload)

        if payload:
            self._insert_first_seen(data_start, payload)
        if fin:
            self.fin_positions.add(data_end)

        next_reference = data_end + (1 if fin else 0)
        if self.unwrap_reference is None:
            self.unwrap_reference = next_reference
        else:
            self.unwrap_reference = max(self.unwrap_reference, next_reference)
        return data_start

    def _insert_first_seen(self, start: int, data: bytes) -> None:
        end = start + len(data)
        if not data:
            return

        # Find the first piece that could overlap [start, end).
        index = bisect.bisect_right(self.starts, start) - 1
        if index < 0 or self.pieces[index].end <= start:
            index += 1

        cursor = start
        new_pieces: List[StreamPiece] = []
        scan = index
        while scan < len(self.pieces) and self.pieces[scan].start < end:
            existing = self.pieces[scan]
            if existing.start > cursor:
                gap_end = min(existing.start, end)
                local_start = cursor - start
                local_end = gap_end - start
                new_pieces.append(StreamPiece(cursor, data[local_start:local_end]))
            cursor = max(cursor, existing.end)
            if cursor >= end:
                break
            scan += 1

        if cursor < end:
            local_start = cursor - start
            new_pieces.append(StreamPiece(cursor, data[local_start:]))

        for piece in new_pieces:
            insert_at = bisect.bisect_left(self.starts, piece.start)
            self.starts.insert(insert_at, piece.start)
            self.pieces.insert(insert_at, piece)

    def byte_at(self, position: int) -> Optional[int]:
        index = bisect.bisect_right(self.starts, position) - 1
        if index < 0:
            return None
        piece = self.pieces[index]
        if position >= piece.end:
            return None
        return piece.data[position - piece.start]

    def contiguous_buffers(self) -> List[SourceBuffer]:
        if not self.pieces:
            return []

        buffers: List[SourceBuffer] = []
        chunk_start = self.pieces[0].start
        chunk_end = self.pieces[0].end
        chunk = bytearray(self.pieces[0].data)

        for piece in self.pieces[1:]:
            if piece.start == chunk_end:
                chunk.extend(piece.data)
                chunk_end = piece.end
                continue
            buffers.append(
                SourceBuffer(
                    carrier="tcp",
                    direction=self.direction,
                    start=chunk_start,
                    data=bytes(chunk),
                    source_id=f"tcp:{self.direction}:{chunk_start}",
                    closed_at_end=chunk_end in self.fin_positions,
                )
            )
            chunk_start = piece.start
            chunk_end = piece.end
            chunk = bytearray(piece.data)

        buffers.append(
            SourceBuffer(
                carrier="tcp",
                direction=self.direction,
                start=chunk_start,
                data=bytes(chunk),
                source_id=f"tcp:{self.direction}:{chunk_start}",
                closed_at_end=chunk_end in self.fin_positions,
            )
        )
        return buffers


@dataclass
class ParsedPcap:
    protocol_name: str
    protocol_number: int
    window_bytes: bytes
    window_refs: List[WindowByteRef]
    payload_packet_count: int
    tcp_states: Dict[int, TcpDirectionState]
    tcp_buffers: List[SourceBuffer]
    udp_buffers: List[SourceBuffer]
    encrypted_transport_evidence: List[str]


def parse_flow_filename(path: Path) -> Tuple[str, int, str, int, str, int]:
    """Parse the same five-tuple convention used by fine-tuning preprocessing."""

    name = path.name
    lower_name = name.lower()
    if lower_name.endswith(".pcapng"):
        stem = name[:-7]
    elif lower_name.endswith(".pcap"):
        stem = name[:-5]
    else:
        raise ValueError("file extension is not .pcap or .pcapng")

    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(
            "filename must be {TCP|UDP}_{src_ip}_{src_port}_{dst_ip}_{dst_port}.pcap"
        )

    protocol_name = parts[0].lower()
    protocol_map = {"tcp": 6, "udp": 17}
    if protocol_name not in protocol_map:
        raise ValueError(f"unsupported filename protocol: {parts[0]}")

    try:
        src_port = int(parts[2])
        dst_port = int(parts[4])
    except ValueError as exc:
        raise ValueError("filename ports must be decimal integers") from exc

    return (
        protocol_name,
        protocol_map[protocol_name],
        parts[1],
        src_port,
        parts[3],
        dst_port,
    )


def _direction_for_packet(packet: Any, protocol_number: int, flow_src: Tuple[str, int]) -> int:
    """Mirror MultiModalFlowExtractor: +1 for filename direction, else -1."""

    ip_layer = packet[IP]
    transport = packet[TCP] if protocol_number == 6 else packet[UDP]
    return 1 if (ip_layer.src, int(transport.sport)) == flow_src else -1


def _looks_like_tls(buffer: bytes) -> bool:
    """Conservative TLS-record evidence; it is not used as a compression label."""

    max_scan = min(len(buffer), 64 * 1024)
    for offset in range(max(0, max_scan - 4)):
        content_type = buffer[offset]
        major = buffer[offset + 1]
        minor = buffer[offset + 2]
        length = int.from_bytes(buffer[offset + 3:offset + 5], "big")
        if content_type not in {20, 21, 22, 23}:
            continue
        if major != 3 or minor > 4 or not (0 < length <= 18432):
            continue
        record_end = offset + 5 + length
        if record_end > len(buffer):
            continue

        # A syntactically plausible handshake record is strong enough by itself.
        if content_type == 22 and length >= 4:
            handshake_type = buffer[offset + 5]
            handshake_length = int.from_bytes(buffer[offset + 6:offset + 9], "big")
            if handshake_type in {1, 2, 4, 8, 11, 12, 13, 14, 15, 16, 20}:
                if handshake_length <= (1 << 24) - 1:
                    return True

        # Application records are less distinctive: require a second valid record.
        if record_end + 5 <= len(buffer):
            next_type = buffer[record_end]
            next_major = buffer[record_end + 1]
            next_minor = buffer[record_end + 2]
            next_length = int.from_bytes(buffer[record_end + 3:record_end + 5], "big")
            if (
                next_type in {20, 21, 22, 23}
                and next_major == 3
                and next_minor <= 4
                and 0 < next_length <= 18432
                and record_end + 5 + next_length <= len(buffer)
            ):
                return True
    return False


def _looks_like_quic_long_header(datagram: bytes) -> bool:
    """Validate the stable portion of a QUIC long header."""

    if len(datagram) < 7 or (datagram[0] & 0xC0) != 0xC0:
        return False
    version = int.from_bytes(datagram[1:5], "big")
    if version == 0:  # Version Negotiation is not encrypted application data.
        return False
    dcid_length = datagram[5]
    if dcid_length > 20 or 6 + dcid_length >= len(datagram):
        return False
    scid_length_offset = 6 + dcid_length
    scid_length = datagram[scid_length_offset]
    if scid_length > 20:
        return False
    return scid_length_offset + 1 + scid_length <= len(datagram)


def read_pcap_for_audit(path: Path) -> ParsedPcap:
    (
        protocol_name,
        protocol_number,
        src_ip,
        src_port,
        _dst_ip,
        _dst_port,
    ) = parse_flow_filename(path)
    flow_src = (src_ip, src_port)

    tcp_states = {
        1: TcpDirectionState(1),
        -1: TcpDirectionState(-1),
    }
    udp_buffers: List[SourceBuffer] = []
    window_values = bytearray()
    window_refs: List[WindowByteRef] = []
    selected_raw_packets = 0
    payload_packet_count = 0
    encrypted_evidence: Set[str] = set()

    try:
        with PcapReader(str(path)) as reader:
            for packet_index, packet in enumerate(reader):
                if IP not in packet:
                    continue

                if protocol_number == 6:
                    if TCP not in packet:
                        continue
                    transport = packet[TCP]
                    direction = _direction_for_packet(packet, protocol_number, flow_src)
                    payload = bytes(transport.payload)
                    flags = int(transport.flags)
                    data_start = tcp_states[direction].observe(
                        int(transport.seq),
                        payload,
                        syn=bool(flags & 0x02),
                        fin=bool(flags & 0x01),
                    )
                    if not payload:
                        continue
                    if _looks_like_tls(payload):
                        encrypted_evidence.add("tls")
                else:
                    if UDP not in packet:
                        continue
                    transport = packet[UDP]
                    direction = _direction_for_packet(packet, protocol_number, flow_src)
                    payload = bytes(transport.payload)
                    data_start = 0
                    if not payload:
                        continue
                    udp_buffers.append(
                        SourceBuffer(
                            carrier="udp",
                            direction=direction,
                            start=0,
                            data=payload,
                            source_id=f"udp:{packet_index}",
                            packet_index=packet_index,
                        )
                    )
                    if _looks_like_quic_long_header(payload):
                        encrypted_evidence.add("quic-long-header")

                payload_packet_count += 1

                if selected_raw_packets < MAX_RAW_PACKETS:
                    selected = payload[:BYTES_PER_PACKET]
                    remaining = MAX_WINDOW_BYTES - len(window_values)
                    selected_for_window = selected[:max(0, remaining)]
                    for payload_offset, value in enumerate(selected_for_window):
                        window_values.append(value)
                        window_refs.append(
                            WindowByteRef(
                                carrier=protocol_name,
                                direction=direction,
                                position=data_start + payload_offset,
                                packet_index=(
                                    packet_index if protocol_number == 17 else None
                                ),
                                value=value,
                            )
                        )
                    selected_raw_packets += 1
    except Exception as exc:
        raise ValueError(f"PCAP read failed: {type(exc).__name__}: {exc}") from exc

    if payload_packet_count == 0:
        raise ValueError("no non-empty IPv4 TCP/UDP payload found")

    tcp_buffers: List[SourceBuffer] = []
    if protocol_number == 6:
        for direction in (1, -1):
            tcp_buffers.extend(tcp_states[direction].contiguous_buffers())
        for source in tcp_buffers:
            if _looks_like_tls(source.data):
                encrypted_evidence.add("tls")

        # A retransmission with conflicting bytes cannot be mapped conservatively
        # onto the first-seen stream used by the protocol/file parsers.
        mapped_refs: List[WindowByteRef] = []
        for ref in window_refs:
            canonical_value = tcp_states[ref.direction].byte_at(ref.position)
            mapped_refs.append(
                WindowByteRef(
                    carrier=ref.carrier,
                    direction=ref.direction,
                    position=ref.position,
                    packet_index=ref.packet_index,
                    value=ref.value,
                    canonical=canonical_value == ref.value,
                )
            )
        window_refs = mapped_refs

    return ParsedPcap(
        protocol_name=protocol_name,
        protocol_number=protocol_number,
        window_bytes=bytes(window_values),
        window_refs=window_refs,
        payload_packet_count=payload_packet_count,
        tcp_states=tcp_states,
        tcp_buffers=tcp_buffers,
        udp_buffers=udp_buffers,
        encrypted_transport_evidence=sorted(encrypted_evidence),
    )


@dataclass(frozen=True)
class DecodeResult:
    output: bytes
    consumed: int
    flavor: str


def _bounded_zlib_decode(
    data: bytes,
    wbits: int,
    max_output: int,
    flavor: str,
) -> DecodeResult:
    """Decode exactly one zlib/deflate/gzip member with an output cap."""

    decoder = zlib.decompressobj(wbits)
    output = bytearray()
    pending = data

    while True:
        remaining = max_output - len(output)
        if remaining < 0:
            raise DecompressionLimitError(
                f"decoded output exceeds {max_output} bytes"
            )
        try:
            part = decoder.decompress(pending, remaining + 1)
        except zlib.error as exc:
            raise DetectionError(f"{flavor} decode failed: {exc}") from exc
        output.extend(part)
        if len(output) > max_output:
            raise DecompressionLimitError(
                f"decoded output exceeds {max_output} bytes"
            )

        if decoder.eof:
            try:
                tail = decoder.flush(max(1, max_output - len(output) + 1))
            except zlib.error as exc:
                raise DetectionError(f"{flavor} flush failed: {exc}") from exc
            output.extend(tail)
            if len(output) > max_output:
                raise DecompressionLimitError(
                    f"decoded output exceeds {max_output} bytes"
                )
            consumed = len(data) - len(decoder.unused_data)
            if consumed <= 0:
                raise DetectionError(f"{flavor} decoder consumed no input")
            return DecodeResult(bytes(output), consumed, flavor)

        if decoder.unconsumed_tail:
            if decoder.unconsumed_tail == pending and not part:
                raise DetectionError(f"{flavor} decoder made no progress")
            pending = decoder.unconsumed_tail
            continue

        raise DetectionError(f"incomplete {flavor} stream")


def _bounded_zlib_decode_range(
    data: bytes,
    wbits: int,
    max_output: int,
    flavor: str,
    *,
    start: int,
    end: Optional[int] = None,
) -> DecodeResult:
    """Decode a Level 2/3 candidate without allocating ``data[start:]``.

    ``consumed`` is relative to ``start``.  Fixed-size input chunks prevent the
    common two-byte zlib candidates in high-entropy traffic from repeatedly
    copying the entire remaining flow.
    """

    if end is None:
        end = len(data)
    if not (0 <= start <= end <= len(data)):
        raise ValueError(
            f"invalid {flavor} input bounds [{start}, {end}) for {len(data)} bytes"
        )

    decoder = zlib.decompressobj(wbits)
    output = bytearray()
    cursor = start
    pending = b""

    while True:
        if not pending:
            if cursor >= end:
                raise DetectionError(f"incomplete {flavor} stream")
            chunk_size = (
                DECODE_PROBE_CHUNK_BYTES
                if cursor == start
                else DECODE_INPUT_CHUNK_BYTES
            )
            chunk_end = min(end, cursor + chunk_size)
            pending = data[cursor:chunk_end]
            cursor = chunk_end

        remaining = max_output - len(output)
        if remaining < 0:
            raise DecompressionLimitError(
                f"decoded output exceeds {max_output} bytes"
            )
        try:
            part = decoder.decompress(pending, remaining + 1)
        except zlib.error as exc:
            raise DetectionError(f"{flavor} decode failed: {exc}") from exc
        output.extend(part)
        if len(output) > max_output:
            raise DecompressionLimitError(
                f"decoded output exceeds {max_output} bytes"
            )

        if decoder.eof:
            try:
                tail = decoder.flush(max(1, max_output - len(output) + 1))
            except zlib.error as exc:
                raise DetectionError(f"{flavor} flush failed: {exc}") from exc
            output.extend(tail)
            if len(output) > max_output:
                raise DecompressionLimitError(
                    f"decoded output exceeds {max_output} bytes"
                )
            consumed = cursor - start - len(decoder.unused_data)
            if consumed <= 0:
                raise DetectionError(f"{flavor} decoder consumed no input")
            return DecodeResult(bytes(output), consumed, flavor)

        if decoder.unconsumed_tail:
            if len(decoder.unconsumed_tail) == len(pending) and not part:
                raise DetectionError(f"{flavor} decoder made no progress")
            pending = decoder.unconsumed_tail
            continue
        pending = b""


def _decode_gzip_members(data: bytes, max_output: int) -> DecodeResult:
    offset = 0
    output = bytearray()
    members = 0

    while offset < len(data) and data[offset:offset + 3] == b"\x1f\x8b\x08":
        remaining_limit = max_output - len(output)
        result = _bounded_zlib_decode(
            data[offset:],
            16 + zlib.MAX_WBITS,
            remaining_limit,
            "gzip",
        )
        output.extend(result.output)
        offset += result.consumed
        members += 1
        if len(output) > max_output:
            raise DecompressionLimitError(
                f"decoded output exceeds {max_output} bytes"
            )

    if members == 0:
        raise DetectionError("gzip magic/header not found")
    return DecodeResult(bytes(output), offset, f"gzip:{members}-member")


def _decode_deflate(data: bytes, max_output: int) -> DecodeResult:
    errors: List[str] = []
    for wbits, flavor in (
        (zlib.MAX_WBITS, "deflate-zlib-wrapper"),
        (-zlib.MAX_WBITS, "deflate-raw"),
    ):
        try:
            return _bounded_zlib_decode(data, wbits, max_output, flavor)
        except DecompressionLimitError:
            raise
        except DetectionError as exc:
            errors.append(str(exc))
    raise DetectionError("; ".join(errors))


def _bounded_bz2_decode(
    data: bytes,
    max_output: int,
    *,
    start: int = 0,
    end: Optional[int] = None,
) -> DecodeResult:
    """Decode one BZIP2 stream in fixed-size input chunks."""

    if end is None:
        end = len(data)
    if not (0 <= start <= end <= len(data)):
        raise ValueError(
            f"invalid bzip2 input bounds [{start}, {end}) for {len(data)} bytes"
        )

    decoder = bz2.BZ2Decompressor()
    output = bytearray()
    cursor = start

    while True:
        if decoder.needs_input:
            if cursor >= end:
                raise DetectionError("incomplete bzip2 stream")
            chunk_end = min(end, cursor + DECODE_INPUT_CHUNK_BYTES)
            pending = data[cursor:chunk_end]
            cursor = chunk_end
        else:
            pending = b""

        remaining = max_output - len(output)
        if remaining < 0:
            raise DecompressionLimitError(
                f"decoded output exceeds {max_output} bytes"
            )
        try:
            part = decoder.decompress(pending, remaining + 1)
        except (OSError, EOFError) as exc:
            raise DetectionError(f"bzip2 decode failed: {exc}") from exc
        output.extend(part)
        if len(output) > max_output:
            raise DecompressionLimitError(
                f"decoded output exceeds {max_output} bytes"
            )
        if decoder.eof:
            consumed = cursor - start - len(decoder.unused_data)
            if consumed <= 0:
                raise DetectionError("bzip2 decoder consumed no input")
            return DecodeResult(bytes(output), consumed, "bzip2")
        if not part and not pending:
            raise DetectionError("bzip2 decoder made no progress")


def _bounded_xz_decode(
    data: bytes,
    max_output: int,
    *,
    start: int = 0,
    end: Optional[int] = None,
) -> DecodeResult:
    """Decode one XZ stream in fixed-size input chunks."""

    if end is None:
        end = len(data)
    if not (0 <= start <= end <= len(data)):
        raise ValueError(
            f"invalid xz input bounds [{start}, {end}) for {len(data)} bytes"
        )

    decoder = lzma.LZMADecompressor(
        format=lzma.FORMAT_XZ,
        memlimit=max(1024 * 1024, max_output),
    )
    output = bytearray()
    cursor = start

    while True:
        if decoder.needs_input:
            if cursor >= end:
                raise DetectionError("incomplete xz stream")
            chunk_end = min(end, cursor + DECODE_INPUT_CHUNK_BYTES)
            pending = data[cursor:chunk_end]
            cursor = chunk_end
        else:
            pending = b""

        remaining = max_output - len(output)
        if remaining < 0:
            raise DecompressionLimitError(
                f"decoded output exceeds {max_output} bytes"
            )
        try:
            part = decoder.decompress(pending, remaining + 1)
        except lzma.LZMAError as exc:
            raise DetectionError(f"xz decode failed: {exc}") from exc
        output.extend(part)
        if len(output) > max_output:
            raise DecompressionLimitError(
                f"decoded output exceeds {max_output} bytes"
            )
        if decoder.eof:
            consumed = cursor - start - len(decoder.unused_data)
            if consumed <= 0:
                raise DetectionError("xz decoder consumed no input")
            return DecodeResult(bytes(output), consumed, "xz")
        if not part and not pending:
            raise DetectionError("xz decoder made no progress")


def _canonical_http_coding(coding: str) -> str:
    if coding in {"gzip", "x-gzip"}:
        return "gzip"
    if coding in {"deflate", "x-deflate"}:
        return "deflate"
    if coding in {"compress", "x-compress"}:
        return "compress"
    return coding


def validate_http_encoding_chain(
    encoded: bytes,
    codings_in_application_order: Sequence[str],
    max_output: int,
    allow_outer_trailing: bool,
) -> Tuple[int, int, List[str]]:
    """Decode HTTP codings in reverse and return wire bytes consumed.

    ``outer_consumed`` can be shorter than ``encoded`` only for a close-delimited
    candidate where the reconstructed TCP buffer also contains later traffic.
    """

    current = encoded
    outer_consumed: Optional[int] = None
    flavors: List[str] = []

    decode_order = [
        _canonical_http_coding(value)
        for value in reversed(codings_in_application_order)
        if value != "identity"
    ]
    if not decode_order:
        raise DetectionError("no compressing HTTP coding")

    for index, coding in enumerate(decode_order):
        if coding == "gzip":
            result = _decode_gzip_members(current, max_output)
        elif coding == "deflate":
            result = _decode_deflate(current, max_output)
        else:
            raise DetectionError(f"unsupported HTTP coding: {coding}")

        if index == 0:
            outer_consumed = result.consumed
            if not allow_outer_trailing and result.consumed != len(current):
                raise DetectionError(
                    f"trailing bytes after outer {coding} stream"
                )
        elif result.consumed != len(current):
            raise DetectionError(f"trailing bytes after inner {coding} stream")

        current = result.output
        flavors.append(result.flavor)

    if outer_consumed is None:
        raise DetectionError("decoder did not run")
    return outer_consumed, len(current), flavors


# ---------------------------------------------------------------------------
# HTTP/1.x explicit compression
# ---------------------------------------------------------------------------


_HTTP_START_LINE = re.compile(
    rb"(?m)^(?:"
    rb"HTTP/1\.[01][ \t]+[0-9]{3}(?:[ \t][^\r\n]*)?"
    rb"|"
    rb"[A-Z][A-Z0-9!#$%&'*+.^_`|~-]{1,19}[ \t]+[^\r\n \t]+"
    rb"[ \t]+HTTP/1\.[01]"
    rb")\r?$"
)
_HTTP_HEADER_NAME = re.compile(rb"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


@dataclass(frozen=True)
class ParsedHttpHeader:
    start: int
    body_start: int
    start_line: str
    headers: Dict[str, List[str]]


@dataclass(frozen=True)
class ExtractedHttpBody:
    encoded: bytes
    spans: List[Tuple[int, int]]
    mode: str
    framed_complete: bool


def _find_http_header_terminator(
    data: bytes,
    start: int,
    max_header_bytes: int,
) -> Optional[Tuple[int, int]]:
    search_end = min(len(data), start + max_header_bytes)
    crlf_end = data.find(b"\r\n\r\n", start, search_end)
    lf_end = data.find(b"\n\n", start, search_end)

    candidates: List[Tuple[int, int]] = []
    if crlf_end >= 0:
        candidates.append((crlf_end, 4))
    if lf_end >= 0:
        candidates.append((lf_end, 2))
    return min(candidates, key=lambda value: value[0]) if candidates else None


def _parse_http_header_at(
    data: bytes,
    start: int,
    max_header_bytes: int,
) -> Optional[ParsedHttpHeader]:
    terminator = _find_http_header_terminator(data, start, max_header_bytes)
    if terminator is None:
        return None
    header_end, delimiter_length = terminator
    raw_block = data[start:header_end]
    if b"\x00" in raw_block:
        return None

    lines = re.split(rb"\r?\n", raw_block)
    if not lines:
        return None
    try:
        start_line = lines[0].decode("iso-8859-1")
    except UnicodeDecodeError:
        return None

    headers: Dict[str, List[str]] = defaultdict(list)
    current_name: Optional[str] = None
    for raw_line in lines[1:]:
        if raw_line.startswith((b" ", b"\t")):
            if current_name is None or not headers[current_name]:
                return None
            continuation = raw_line.strip().decode("iso-8859-1")
            headers[current_name][-1] += " " + continuation
            continue
        if b":" not in raw_line:
            return None
        raw_name, raw_value = raw_line.split(b":", 1)
        if not _HTTP_HEADER_NAME.fullmatch(raw_name):
            return None
        current_name = raw_name.decode("ascii").lower()
        headers[current_name].append(raw_value.strip().decode("iso-8859-1"))

    return ParsedHttpHeader(
        start=start,
        body_start=header_end + delimiter_length,
        start_line=start_line,
        headers=dict(headers),
    )


def _split_http_codings(values: Iterable[str]) -> List[str]:
    codings: List[str] = []
    for value in values:
        for item in value.split(","):
            coding = item.strip().split(";", 1)[0].strip().lower()
            if coding:
                codings.append(coding)
    return codings


def _parse_http_content_length(values: Sequence[str]) -> Optional[int]:
    parsed: List[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item or not item.isdigit():
                raise DetectionError("invalid HTTP Content-Length")
            parsed.append(int(item))
    if not parsed:
        return None
    if len(set(parsed)) != 1:
        raise DetectionError("conflicting HTTP Content-Length values")
    return parsed[0]


def _parse_chunked_body(data: bytes, body_start: int) -> ExtractedHttpBody:
    cursor = body_start
    decoded_chunks: List[bytes] = []
    spans: List[Tuple[int, int]] = []

    while True:
        line_end = data.find(b"\r\n", cursor, min(len(data), cursor + 8192))
        if line_end < 0:
            raise DetectionError("incomplete HTTP chunk-size line")
        size_token = data[cursor:line_end].split(b";", 1)[0].strip()
        if not size_token or not re.fullmatch(rb"[0-9A-Fa-f]+", size_token):
            raise DetectionError("invalid HTTP chunk size")
        size = int(size_token, 16)
        cursor = line_end + 2

        if size == 0:
            # Trailer section ends at the first empty line.  With no trailers,
            # the CRLF immediately following the zero-size line is that empty line.
            if cursor + 2 <= len(data) and data[cursor:cursor + 2] == b"\r\n":
                return ExtractedHttpBody(
                    encoded=b"".join(decoded_chunks),
                    spans=spans,
                    mode="chunked",
                    framed_complete=True,
                )
            trailer_end = data.find(b"\r\n\r\n", cursor)
            if trailer_end < 0:
                raise DetectionError("incomplete HTTP chunked trailers")
            return ExtractedHttpBody(
                encoded=b"".join(decoded_chunks),
                spans=spans,
                mode="chunked",
                framed_complete=True,
            )

        chunk_end = cursor + size
        if chunk_end + 2 > len(data):
            raise DetectionError("incomplete HTTP chunk data")
        if data[chunk_end:chunk_end + 2] != b"\r\n":
            raise DetectionError("missing CRLF after HTTP chunk data")
        decoded_chunks.append(data[cursor:chunk_end])
        spans.append((cursor, chunk_end))
        cursor = chunk_end + 2


def _extract_http_body(
    data: bytes,
    header: ParsedHttpHeader,
    transfer_codings: Sequence[str],
) -> ExtractedHttpBody:
    if "chunked" in transfer_codings:
        if transfer_codings[-1] != "chunked":
            raise DetectionError("HTTP chunked transfer coding is not final")
        return _parse_chunked_body(data, header.body_start)

    # RFC 7230/9112: Transfer-Encoding overrides Content-Length.  A non-chunked
    # transfer coding is delimited by connection close (or by its own coding EOF).
    if transfer_codings:
        body = data[header.body_start:]
        spans = [] if not body else [(header.body_start, len(data))]
        return ExtractedHttpBody(
            encoded=body,
            spans=spans,
            mode="transfer-coding-eof",
            framed_complete=False,
        )

    content_length = _parse_http_content_length(
        header.headers.get("content-length", [])
    )
    if content_length is not None:
        body_end = header.body_start + content_length
        if body_end > len(data):
            raise DetectionError("incomplete Content-Length body")
        spans = [] if content_length == 0 else [(header.body_start, body_end)]
        return ExtractedHttpBody(
            encoded=data[header.body_start:body_end],
            spans=spans,
            mode="content-length",
            framed_complete=True,
        )

    # A close-delimited entity can still be validated by the coding's own EOF
    # marker.  The decoder is allowed to leave a later HTTP message unused.
    body = data[header.body_start:]
    spans = [] if not body else [(header.body_start, len(data))]
    return ExtractedHttpBody(
        encoded=body,
        spans=spans,
        mode="coding-eof",
        framed_complete=False,
    )


def _trim_logical_spans(
    spans: Sequence[Tuple[int, int]],
    logical_length: int,
) -> List[Tuple[int, int]]:
    """Map a prefix of de-chunked bytes back to original stream intervals."""

    remaining = logical_length
    result: List[Tuple[int, int]] = []
    for start, end in spans:
        if remaining <= 0:
            break
        take = min(end - start, remaining)
        if take > 0:
            result.append((start, start + take))
        remaining -= take
    if remaining != 0:
        raise DetectionError("HTTP body span mapping is incomplete")
    return result


def detect_http_compression(
    source: SourceBuffer,
    max_output: int,
    max_header_bytes: int,
    compression_level: int,
    evidence: List[CompressionEvidence],
    unsupported: List[Dict[str, Any]],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    if source.carrier != "tcp":
        return

    seen_evidence: Set[Tuple[Tuple[int, int], ...]] = set()
    for start_match in _HTTP_START_LINE.finditer(source.data):
        header = _parse_http_header_at(
            source.data,
            start_match.start(),
            max_header_bytes,
        )
        if header is None:
            continue

        content_codings = _split_http_codings(
            header.headers.get("content-encoding", [])
        )
        transfer_codings = _split_http_codings(
            header.headers.get("transfer-encoding", [])
        )
        applied_codings = [
            coding for coding in content_codings + transfer_codings
            if coding != "chunked"
        ]
        compressing_codings = [
            coding for coding in applied_codings if coding != "identity"
        ]
        if not compressing_codings:
            continue

        unsupported_codings = sorted({
            coding for coding in compressing_codings
            if coding not in SUPPORTED_HTTP_CODINGS
        })
        if unsupported_codings:
            declared_supported = (
                compression_level >= 3
                and all(
                    coding in DECLARED_HTTP_COMPRESSION_CODINGS
                    for coding in compressing_codings
                )
            )
            if declared_supported:
                try:
                    body = _extract_http_body(
                        source.data,
                        header,
                        transfer_codings,
                    )
                    if not body.encoded or not body.spans:
                        raise DetectionError(
                            "declared-compressed HTTP entity has an empty body"
                        )
                    if not body.framed_complete and not source.closed_at_end:
                        raise DetectionError(
                            "declared compression lacks an exact HTTP body boundary"
                        )
                    span_key = tuple(body.spans)
                    if span_key in seen_evidence:
                        continue
                    seen_evidence.add(span_key)
                    evidence.append(
                        CompressionEvidence(
                            category="protocol",
                            kind="http:" + "+".join(
                                _canonical_http_coding(value)
                                for value in compressing_codings
                            ),
                            source=source.source_id,
                            regions=[
                                source.region(start, end)
                                for start, end in body.spans
                            ],
                            detail={
                                "start_line": header.start_line,
                                "header_stream_offset": (
                                    source.start + header.start
                                ),
                                "body_mode": body.mode,
                                "wire_encoded_bytes": len(body.encoded),
                                "minimum_level": 3,
                                "validation": (
                                    "explicit recognized HTTP compression coding "
                                    "plus exact message framing; payload not decoded"
                                ),
                            },
                        )
                    )
                except (DetectionError, ValueError) as exc:
                    candidate_failures.append({
                        "kind": "http-declared:" + "+".join(
                            compressing_codings
                        ),
                        "source": source.source_id,
                        "stream_offset": source.start + header.start,
                        "reason": str(exc),
                    })
                continue

            unsupported.append({
                "kind": "http-content-coding",
                "source": source.source_id,
                "stream_offset": source.start + header.start,
                "codings": unsupported_codings,
                "reason": "explicit coding is outside the confirmed detector scope",
            })
            continue

        try:
            body = _extract_http_body(source.data, header, transfer_codings)
            if not body.encoded or not body.spans:
                raise DetectionError("compressed HTTP entity has an empty body")
            outer_consumed, decoded_length, flavors = validate_http_encoding_chain(
                body.encoded,
                applied_codings,
                max_output,
                allow_outer_trailing=not body.framed_complete,
            )
            mapped_spans = _trim_logical_spans(body.spans, outer_consumed)
            span_key = tuple(mapped_spans)
            if span_key in seen_evidence:
                continue
            seen_evidence.add(span_key)
            regions = [source.region(start, end) for start, end in mapped_spans]
            evidence.append(
                CompressionEvidence(
                    category="protocol",
                    kind="http:" + "+".join(
                        _canonical_http_coding(value)
                        for value in compressing_codings
                    ),
                    source=source.source_id,
                    regions=regions,
                    detail={
                        "start_line": header.start_line,
                        "header_stream_offset": source.start + header.start,
                        "body_mode": body.mode,
                        "wire_encoded_bytes": outer_consumed,
                        "decoded_bytes": decoded_length,
                        "decoder_flavors": flavors,
                    },
                )
            )
        except (DetectionError, ValueError) as exc:
            candidate_failures.append({
                "kind": "http:" + "+".join(compressing_codings),
                "source": source.source_id,
                "stream_offset": source.start + header.start,
                "reason": str(exc),
            })


# ---------------------------------------------------------------------------
# Intrinsic compressed formats
# ---------------------------------------------------------------------------


def _gzip_header_end(data: bytes, start: int) -> int:
    if data[start:start + 3] != b"\x1f\x8b\x08" or start + 10 > len(data):
        raise DetectionError("invalid gzip fixed header")
    flags = data[start + 3]
    if flags & 0xE0:
        raise DetectionError("gzip reserved flags are set")
    cursor = start + 10

    if flags & 0x04:  # FEXTRA
        if cursor + 2 > len(data):
            raise DetectionError("incomplete gzip FEXTRA length")
        extra_length = int.from_bytes(data[cursor:cursor + 2], "little")
        cursor += 2 + extra_length
        if cursor > len(data):
            raise DetectionError("incomplete gzip FEXTRA")

    for flag, name in ((0x08, "FNAME"), (0x10, "FCOMMENT")):
        if flags & flag:
            terminator = data.find(b"\x00", cursor)
            if terminator < 0:
                raise DetectionError(f"incomplete gzip {name}")
            cursor = terminator + 1

    if flags & 0x02:  # FHCRC
        cursor += 2
        if cursor > len(data):
            raise DetectionError("incomplete gzip FHCRC")
    return cursor


def detect_gzip(
    source: SourceBuffer,
    max_output: int,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    cursor = 0
    while True:
        start = source.data.find(b"\x1f\x8b\x08", cursor)
        if start < 0:
            return
        try:
            header_end = _gzip_header_end(source.data, start)
            decoded = _bounded_zlib_decode(
                source.data[start:],
                16 + zlib.MAX_WBITS,
                max_output,
                "gzip",
            )
            object_end = start + decoded.consumed
            deflate_end = object_end - 8  # CRC32 + ISIZE trailer
            if header_end >= deflate_end:
                raise DetectionError("gzip member has no deflate payload")
            evidence.append(
                CompressionEvidence(
                    category="intrinsic",
                    kind="gzip",
                    source=source.source_id,
                    regions=[source.region(header_end, deflate_end)],
                    detail={
                        "object_start": source.start + start,
                        "object_end": source.start + object_end,
                        "decoded_bytes": len(decoded.output),
                        "validation": "gzip decode plus CRC/ISIZE",
                    },
                )
            )
            cursor = object_end
        except DecompressionLimitError as exc:
            candidate_failures.append({
                "kind": "gzip",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + 3
        except (DetectionError, ValueError) as exc:
            # A three-byte magic match alone is not evidence.  Keep the failure
            # only for auditability; it never contributes to a compressed count.
            candidate_failures.append({
                "kind": "gzip",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + 3


def _iter_signature_offsets(data: bytes, signature: bytes) -> Iterable[int]:
    cursor = 0
    while True:
        offset = data.find(signature, cursor)
        if offset < 0:
            return
        yield offset
        cursor = offset + 1


def _validate_zip_entry(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    max_output: int,
) -> None:
    if info.flag_bits & 0x1:
        raise DetectionError(f"encrypted ZIP entry: {info.filename}")
    if info.file_size > max_output:
        raise DecompressionLimitError(
            f"ZIP entry {info.filename!r} exceeds {max_output} decoded bytes"
        )

    total = 0
    try:
        with archive.open(info, "r") as entry:
            while True:
                block = entry.read(min(1024 * 1024, max_output - total + 1))
                if not block:
                    break
                total += len(block)
                if total > max_output:
                    raise DecompressionLimitError(
                        f"ZIP entry {info.filename!r} exceeds {max_output} decoded bytes"
                    )
    except (
        RuntimeError,
        EOFError,
        OSError,
        ValueError,
        zlib.error,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        NotImplementedError,
    ) as exc:
        raise DetectionError(f"ZIP entry validation failed: {exc}") from exc
    if total != info.file_size:
        raise DetectionError(
            f"ZIP entry size mismatch for {info.filename!r}: {total} != {info.file_size}"
        )


def _zip_local_data_region(blob: bytes, info: zipfile.ZipInfo) -> Tuple[int, int]:
    offset = info.header_offset
    if offset < 0 or offset + 30 > len(blob):
        raise DetectionError("ZIP local header is out of bounds")
    fields = struct.unpack_from("<4s5H3L2H", blob, offset)
    signature = fields[0]
    method = fields[3]
    name_length = fields[9]
    extra_length = fields[10]
    if signature != b"PK\x03\x04":
        raise DetectionError("ZIP local header signature mismatch")
    if method != info.compress_type:
        raise DetectionError("ZIP local/central compression method mismatch")
    data_start = offset + 30 + name_length + extra_length
    data_end = data_start + info.compress_size
    if data_start < 0 or data_end > len(blob):
        raise DetectionError("ZIP compressed entry data is out of bounds")
    return data_start, data_end


def detect_zip(
    source: SourceBuffer,
    max_output: int,
    compression_level: int,
    evidence: List[CompressionEvidence],
    unsupported: List[Dict[str, Any]],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    accepted_archives: Set[Tuple[int, int]] = set()
    for eocd in _iter_signature_offsets(source.data, b"PK\x05\x06"):
        if eocd + 22 > len(source.data):
            continue
        try:
            (
                signature,
                disk_number,
                central_disk,
                entries_on_disk,
                entries_total,
                central_size,
                central_offset,
                comment_length,
            ) = struct.unpack_from("<4s4H2LH", source.data, eocd)
            if signature != b"PK\x05\x06":
                continue
            archive_end = eocd + 22 + comment_length
            if archive_end > len(source.data):
                raise DetectionError("incomplete ZIP end record/comment")
            if disk_number != 0 or central_disk != 0 or entries_on_disk != entries_total:
                raise DetectionError("multi-disk ZIP is outside detector scope")
            if (
                entries_total == 0xFFFF
                or central_size == 0xFFFFFFFF
                or central_offset == 0xFFFFFFFF
            ):
                raise DetectionError("ZIP64 is outside detector scope")

            archive_start = eocd - central_size - central_offset
            if archive_start < 0 or archive_start >= archive_end:
                raise DetectionError("cannot infer embedded ZIP start")
            archive_key = (archive_start, archive_end)
            if archive_key in accepted_archives:
                continue

            blob = source.data[archive_start:archive_end]
            regions: List[ByteRegion] = []
            declared_regions: List[ByteRegion] = []
            methods: Set[int] = set()
            declared_methods: Set[int] = set()
            unsupported_entries: List[Dict[str, Any]] = []

            with zipfile.ZipFile(io.BytesIO(blob), "r") as archive:
                infos = archive.infolist()
                if len(infos) != entries_total:
                    raise DetectionError("ZIP central-directory entry count mismatch")
                archive_decoded_total = 0
                for info in infos:
                    if info.is_dir() or info.compress_size == 0:
                        continue
                    if info.compress_type == zipfile.ZIP_STORED:
                        continue
                    if info.flag_bits & 0x1:
                        unsupported_entries.append({
                            "filename": info.filename,
                            "method": info.compress_type,
                            "encrypted": True,
                        })
                        continue
                    if info.compress_type not in ZIP_SUPPORTED_METHODS:
                        if info.flag_bits & 0x40:
                            unsupported_entries.append({
                                "filename": info.filename,
                                "method": info.compress_type,
                                "encrypted": True,
                            })
                            continue
                        if (
                            compression_level >= 3
                            and info.compress_type
                            in ZIP_DECLARED_COMPRESSION_METHODS
                        ):
                            local_start, local_end = _zip_local_data_region(
                                blob,
                                info,
                            )
                            declared_regions.append(
                                source.region(
                                    archive_start + local_start,
                                    archive_start + local_end,
                                )
                            )
                            declared_methods.add(info.compress_type)
                            continue
                        unsupported_entries.append({
                            "filename": info.filename,
                            "method": info.compress_type,
                        })
                        continue
                    archive_decoded_total += info.file_size
                    if archive_decoded_total > max_output:
                        raise DecompressionLimitError(
                            f"ZIP decoded entries exceed {max_output} bytes in total"
                        )
                    _validate_zip_entry(archive, info, max_output)
                    local_start, local_end = _zip_local_data_region(blob, info)
                    regions.append(
                        source.region(
                            archive_start + local_start,
                            archive_start + local_end,
                        )
                    )
                    methods.add(info.compress_type)

            if unsupported_entries:
                unsupported.append({
                    "kind": "zip-compression-method",
                    "source": source.source_id,
                    "offset": source.start + archive_start,
                    "entries": unsupported_entries,
                    "reason": "ZIP entry method is outside confirmed scope",
                })
            if regions:
                accepted_archives.add(archive_key)
                evidence.append(
                    CompressionEvidence(
                        category="intrinsic",
                        kind="zip",
                        source=source.source_id,
                        regions=regions,
                        detail={
                            "object_start": source.start + archive_start,
                            "object_end": source.start + archive_end,
                            "compressed_entries": len(regions),
                            "methods": sorted(methods),
                            "validation": "central/local headers plus per-entry decode/CRC",
                        },
                    )
                )
            if declared_regions:
                accepted_archives.add(archive_key)
                evidence.append(
                    CompressionEvidence(
                        category="intrinsic",
                        kind="zip-declared",
                        source=source.source_id,
                        regions=declared_regions,
                        detail={
                            "object_start": source.start + archive_start,
                            "object_end": source.start + archive_end,
                            "compressed_entries": len(declared_regions),
                            "methods": sorted(declared_methods),
                            "minimum_level": 3,
                            "validation": (
                                "complete ZIP central/local structure plus "
                                "recognized non-stored, non-encrypted method; "
                                "entry payload not decoded"
                            ),
                        },
                    )
                )
        except DecompressionLimitError as exc:
            candidate_failures.append({
                "kind": "zip",
                "source": source.source_id,
                "offset": source.start + eocd,
                "reason": str(exc),
            })
        except (
            DetectionError,
            ValueError,
            struct.error,
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
        ) as exc:
            candidate_failures.append({
                "kind": "zip",
                "source": source.source_id,
                "offset": source.start + eocd,
                "reason": str(exc),
            })


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _validate_png_ihdr(payload: bytes) -> None:
    if len(payload) != 13:
        raise DetectionError("PNG IHDR length is not 13")
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", payload)
    )
    if width == 0 or height == 0:
        raise DetectionError("PNG dimensions must be non-zero")
    allowed_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    if color_type not in allowed_depths or bit_depth not in allowed_depths[color_type]:
        raise DetectionError("invalid PNG color type/bit depth")
    if compression != 0 or filtering != 0 or interlace not in {0, 1}:
        raise DetectionError("unsupported or invalid PNG IHDR methods")


def detect_png(
    source: SourceBuffer,
    max_output: int,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    cursor = 0
    while True:
        start = source.data.find(PNG_SIGNATURE, cursor)
        if start < 0:
            return
        try:
            position = start + len(PNG_SIGNATURE)
            chunk_index = 0
            idat_payloads: List[bytes] = []
            idat_spans: List[Tuple[int, int]] = []
            saw_iend = False

            while position + 12 <= len(source.data):
                length = int.from_bytes(source.data[position:position + 4], "big")
                chunk_type = source.data[position + 4:position + 8]
                payload_start = position + 8
                payload_end = payload_start + length
                crc_end = payload_end + 4
                if crc_end > len(source.data):
                    raise DetectionError("incomplete PNG chunk")
                if not re.fullmatch(rb"[A-Za-z]{4}", chunk_type):
                    raise DetectionError("invalid PNG chunk type")
                expected_crc = int.from_bytes(source.data[payload_end:crc_end], "big")
                actual_crc = binascii.crc32(
                    chunk_type + source.data[payload_start:payload_end]
                ) & 0xFFFFFFFF
                if expected_crc != actual_crc:
                    raise DetectionError("PNG chunk CRC mismatch")

                if chunk_index == 0:
                    if chunk_type != b"IHDR":
                        raise DetectionError("PNG first chunk is not IHDR")
                    _validate_png_ihdr(source.data[payload_start:payload_end])
                elif chunk_type == b"IDAT":
                    idat_payloads.append(source.data[payload_start:payload_end])
                    if length:
                        idat_spans.append((payload_start, payload_end))
                elif chunk_type == b"IEND":
                    if length != 0:
                        raise DetectionError("PNG IEND is non-empty")
                    saw_iend = True
                    object_end = crc_end
                    break

                chunk_index += 1
                position = crc_end

            if not saw_iend:
                raise DetectionError("PNG IEND not found")
            if not idat_payloads or not idat_spans:
                raise DetectionError("PNG has no non-empty IDAT data")
            compressed = b"".join(idat_payloads)
            decoded = _bounded_zlib_decode(
                compressed,
                zlib.MAX_WBITS,
                max_output,
                "png-idat-zlib",
            )
            if decoded.consumed != len(compressed):
                raise DetectionError("trailing bytes in PNG IDAT zlib stream")

            evidence.append(
                CompressionEvidence(
                    category="intrinsic",
                    kind="png",
                    source=source.source_id,
                    regions=[source.region(a, b) for a, b in idat_spans],
                    detail={
                        "object_start": source.start + start,
                        "object_end": source.start + object_end,
                        "decoded_idat_bytes": len(decoded.output),
                        "validation": "PNG structure/chunk CRC plus IDAT zlib decode",
                    },
                )
            )
            cursor = object_end
        except DecompressionLimitError as exc:
            candidate_failures.append({
                "kind": "png",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + len(PNG_SIGNATURE)
        except (DetectionError, ValueError, struct.error) as exc:
            candidate_failures.append({
                "kind": "png",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + len(PNG_SIGNATURE)


JPEG_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3,
    0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB,
    0xCD, 0xCE, 0xCF,
}
JPEG_STANDALONE_MARKERS = {0x01, *range(0xD0, 0xD8)}


def _parse_jpeg(source_data: bytes, start: int) -> Tuple[int, List[Tuple[int, int]]]:
    if source_data[start:start + 2] != b"\xff\xd8":
        raise DetectionError("JPEG SOI not found")

    position = start + 2
    saw_sof = False
    saw_sos = False
    scan_spans: List[Tuple[int, int]] = []

    while position < len(source_data):
        if source_data[position] != 0xFF:
            raise DetectionError("JPEG marker prefix expected")
        marker_start = position
        while position < len(source_data) and source_data[position] == 0xFF:
            position += 1
        if position >= len(source_data):
            raise DetectionError("truncated JPEG marker")
        marker = source_data[position]
        position += 1

        if marker == 0x00:
            raise DetectionError("stuffed JPEG byte outside entropy scan")
        if marker == 0xD9:  # EOI
            if not saw_sof or not saw_sos or not scan_spans:
                raise DetectionError("JPEG lacks SOF/SOS entropy-coded data")
            return position, scan_spans
        if marker == 0xD8:
            raise DetectionError("nested JPEG SOI")
        if marker in JPEG_STANDALONE_MARKERS:
            # Restart markers should have been consumed inside a scan.
            if 0xD0 <= marker <= 0xD7:
                raise DetectionError("JPEG restart marker outside entropy scan")
            continue

        if position + 2 > len(source_data):
            raise DetectionError("truncated JPEG segment length")
        segment_length = int.from_bytes(source_data[position:position + 2], "big")
        if segment_length < 2:
            raise DetectionError("invalid JPEG segment length")
        segment_end = position + segment_length
        if segment_end > len(source_data):
            raise DetectionError("truncated JPEG segment")

        if marker in JPEG_SOF_MARKERS:
            if segment_length < 8:
                raise DetectionError("invalid JPEG SOF segment")
            payload = source_data[position + 2:segment_end]
            height = int.from_bytes(payload[1:3], "big")
            width = int.from_bytes(payload[3:5], "big")
            components = payload[5]
            if width == 0 or height == 0 or components == 0:
                raise DetectionError("invalid JPEG dimensions/components")
            if segment_length != 8 + 3 * components:
                raise DetectionError("JPEG SOF component table length mismatch")
            saw_sof = True

        if marker != 0xDA:  # not SOS
            position = segment_end
            continue

        if segment_length < 6:
            raise DetectionError("invalid JPEG SOS segment")
        saw_sos = True
        scan_start = segment_end
        scan_cursor = scan_start

        while True:
            ff = source_data.find(b"\xff", scan_cursor)
            if ff < 0:
                raise DetectionError("JPEG entropy scan has no terminating marker")
            after_ff = ff + 1
            while after_ff < len(source_data) and source_data[after_ff] == 0xFF:
                after_ff += 1
            if after_ff >= len(source_data):
                raise DetectionError("truncated JPEG entropy scan marker")
            following = source_data[after_ff]
            if following == 0x00 or 0xD0 <= following <= 0xD7:
                scan_cursor = after_ff + 1
                continue

            if ff > scan_start:
                scan_spans.append((scan_start, ff))
            position = ff
            break


def detect_jpeg(
    source: SourceBuffer,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    cursor = 0
    while True:
        start = source.data.find(b"\xff\xd8\xff", cursor)
        if start < 0:
            return
        try:
            object_end, scan_spans = _parse_jpeg(source.data, start)
            evidence.append(
                CompressionEvidence(
                    category="intrinsic",
                    kind="jpeg",
                    source=source.source_id,
                    regions=[source.region(a, b) for a, b in scan_spans],
                    detail={
                        "object_start": source.start + start,
                        "object_end": source.start + object_end,
                        "scan_count": len(scan_spans),
                        "validation": "complete JPEG marker structure through EOI",
                    },
                )
            )
            cursor = object_end
        except (DetectionError, ValueError) as exc:
            candidate_failures.append({
                "kind": "jpeg",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + 3


def _validate_vp8_bitstream(payload: bytes) -> None:
    if len(payload) < 10:
        raise DetectionError("VP8 bitstream is too short")
    frame_tag = int.from_bytes(payload[0:3], "little")
    if frame_tag & 0x1:
        raise DetectionError("top-level VP8 frame is not a key frame")
    if payload[3:6] != b"\x9d\x01\x2a":
        raise DetectionError("VP8 key-frame start code mismatch")
    first_partition_length = frame_tag >> 5
    if 10 + first_partition_length > len(payload):
        raise DetectionError("VP8 first partition exceeds chunk length")
    width = int.from_bytes(payload[6:8], "little") & 0x3FFF
    height = int.from_bytes(payload[8:10], "little") & 0x3FFF
    if width == 0 or height == 0:
        raise DetectionError("VP8 dimensions are zero")


def _validate_vp8l_bitstream(payload: bytes) -> None:
    if len(payload) < 5 or payload[0] != 0x2F:
        raise DetectionError("VP8L signature is invalid")
    packed = int.from_bytes(payload[1:5], "little")
    width = (packed & 0x3FFF) + 1
    height = ((packed >> 14) & 0x3FFF) + 1
    version = (packed >> 29) & 0x7
    if width <= 0 or height <= 0 or version != 0:
        raise DetectionError("VP8L dimensions/version are invalid")


def _parse_webp_chunks(
    data: bytes,
    start: int,
    end: int,
    allow_anmf: bool,
) -> List[Tuple[int, int]]:
    position = start
    regions: List[Tuple[int, int]] = []

    while position < end:
        if position + 8 > end:
            raise DetectionError("truncated WebP chunk header")
        fourcc = data[position:position + 4]
        length = int.from_bytes(data[position + 4:position + 8], "little")
        payload_start = position + 8
        payload_end = payload_start + length
        padded_end = payload_end + (length & 1)
        if payload_end > end or padded_end > end:
            raise DetectionError("WebP chunk exceeds RIFF boundary")

        payload = data[payload_start:payload_end]
        if fourcc == b"VP8 ":
            _validate_vp8_bitstream(payload)
            regions.append((payload_start, payload_end))
        elif fourcc == b"VP8L":
            _validate_vp8l_bitstream(payload)
            regions.append((payload_start, payload_end))
        elif fourcc == b"VP8X":
            if length != 10:
                raise DetectionError("VP8X chunk length is not 10")
        elif fourcc == b"ANMF" and allow_anmf:
            if length < 16:
                raise DetectionError("ANMF chunk is too short")
            regions.extend(
                _parse_webp_chunks(data, payload_start + 16, payload_end, False)
            )

        position = padded_end

    if position != end:
        raise DetectionError("WebP RIFF chunk alignment mismatch")
    return regions


def detect_webp(
    source: SourceBuffer,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    cursor = 0
    while True:
        start = source.data.find(b"RIFF", cursor)
        if start < 0:
            return
        if source.data[start + 8:start + 12] != b"WEBP":
            cursor = start + 4
            continue
        try:
            if start + 12 > len(source.data):
                raise DetectionError("truncated WebP RIFF header")
            riff_size = int.from_bytes(source.data[start + 4:start + 8], "little")
            object_end = start + 8 + riff_size
            if riff_size < 4 or object_end > len(source.data):
                raise DetectionError("WebP RIFF size exceeds available bytes")
            bitstream_spans = _parse_webp_chunks(
                source.data,
                start + 12,
                object_end,
                True,
            )
            if not bitstream_spans:
                raise DetectionError("WebP has no validated VP8/VP8L bitstream")
            evidence.append(
                CompressionEvidence(
                    category="intrinsic",
                    kind="webp",
                    source=source.source_id,
                    regions=[source.region(a, b) for a, b in bitstream_spans],
                    detail={
                        "object_start": source.start + start,
                        "object_end": source.start + object_end,
                        "bitstream_count": len(bitstream_spans),
                        "validation": "RIFF/chunk structure plus VP8/VP8L headers",
                    },
                )
            )
            cursor = object_end
        except (DetectionError, ValueError) as exc:
            candidate_failures.append({
                "kind": "webp",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + 4


def _has_supported_zlib_header(data: bytes, start: int) -> bool:
    if start < 0 or start + 2 > len(data):
        return False
    cmf = data[start]
    flags = data[start + 1]
    return (
        cmf & 0x0F == 8
        and cmf >> 4 <= 7
        and ((cmf << 8) | flags) % 31 == 0
        and not (flags & 0x20)  # preset dictionary unavailable to the audit
    )


def detect_zlib_streams(
    source: SourceBuffer,
    max_output: int,
    evidence: List[CompressionEvidence],
) -> None:
    """Find fully validated zlib streams without preset dictionaries.

    A two-byte zlib header is too common to retain failed candidates on large
    high-entropy datasets.  Only no-dictionary streams that reach EOF and pass
    Adler-32 are emitted.
    """

    accepted_until = -1
    for match in ZLIB_HEADER_PATTERN.finditer(source.data):
        start = match.start()
        if start < accepted_until:
            continue
        try:
            decoded = _bounded_zlib_decode_range(
                source.data,
                zlib.MAX_WBITS,
                max_output,
                "zlib",
                start=start,
            )
            if not decoded.output:
                raise DetectionError("zlib stream expands to an empty payload")
            object_end = start + decoded.consumed
            deflate_start = start + 2
            deflate_end = object_end - 4  # Adler-32 trailer
            if deflate_start >= deflate_end:
                raise DetectionError("zlib stream has no DEFLATE payload")
            evidence.append(
                CompressionEvidence(
                    category="intrinsic",
                    kind="zlib",
                    source=source.source_id,
                    regions=[source.region(deflate_start, deflate_end)],
                    detail={
                        "object_start": source.start + start,
                        "object_end": source.start + object_end,
                        "decoded_bytes": len(decoded.output),
                        "minimum_level": 2,
                        "validation": (
                            "valid no-dictionary zlib header plus complete decode "
                            "and Adler-32"
                        ),
                    },
                )
            )
            accepted_until = object_end
        except (DetectionError, DecompressionLimitError, ValueError):
            # Failed two-byte candidates are intentionally silent; otherwise
            # random/high-entropy traffic would create enormous audit JSON.
            continue


def detect_bzip2_streams(
    source: SourceBuffer,
    max_output: int,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    cursor = 0
    while True:
        start = source.data.find(b"BZh", cursor)
        if start < 0:
            return
        try:
            if (
                start + 4 > len(source.data)
                or source.data[start + 3] not in b"123456789"
            ):
                raise DetectionError("invalid BZIP2 block-size header")
            decoded = _bounded_bz2_decode(
                source.data,
                max_output,
                start=start,
            )
            if not decoded.output:
                raise DetectionError("BZIP2 stream expands to an empty payload")
            object_end = start + decoded.consumed
            evidence.append(
                CompressionEvidence(
                    category="intrinsic",
                    kind="bzip2",
                    source=source.source_id,
                    regions=[source.region(start, object_end)],
                    detail={
                        "object_start": source.start + start,
                        "object_end": source.start + object_end,
                        "decoded_bytes": len(decoded.output),
                        "minimum_level": 2,
                        "validation": "complete BZIP2 decode plus internal CRC",
                    },
                )
            )
            cursor = object_end
        except (DetectionError, DecompressionLimitError, ValueError) as exc:
            candidate_failures.append({
                "kind": "bzip2",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + 3


def detect_xz_streams(
    source: SourceBuffer,
    max_output: int,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    cursor = 0
    while True:
        start = source.data.find(XZ_STREAM_MAGIC, cursor)
        if start < 0:
            return
        try:
            decoded = _bounded_xz_decode(
                source.data,
                max_output,
                start=start,
            )
            if not decoded.output:
                raise DetectionError("XZ stream expands to an empty payload")
            object_end = start + decoded.consumed
            evidence.append(
                CompressionEvidence(
                    category="intrinsic",
                    kind="xz",
                    source=source.source_id,
                    regions=[source.region(start, object_end)],
                    detail={
                        "object_start": source.start + start,
                        "object_end": source.start + object_end,
                        "decoded_bytes": len(decoded.output),
                        "minimum_level": 2,
                        "validation": "complete XZ decode plus integrity check",
                    },
                )
            )
            cursor = object_end
        except (DetectionError, DecompressionLimitError, ValueError) as exc:
            candidate_failures.append({
                "kind": "xz",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + len(XZ_STREAM_MAGIC)


def detect_cws(
    source: SourceBuffer,
    max_output: int,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    """Validate zlib-compressed SWF (CWS) objects."""

    cursor = 0
    while True:
        start = source.data.find(b"CWS", cursor)
        if start < 0:
            return
        try:
            if start + 8 > len(source.data):
                raise DetectionError("truncated CWS header")
            version = source.data[start + 3]
            file_length = int.from_bytes(
                source.data[start + 4:start + 8],
                "little",
            )
            if version < 6 or file_length < 8:
                raise DetectionError("invalid CWS version or file length")
            if file_length - 8 > max_output:
                raise DecompressionLimitError(
                    f"CWS decoded body exceeds {max_output} bytes"
                )
            encoded_start = start + 8
            if not _has_supported_zlib_header(source.data, encoded_start):
                raise DetectionError("CWS body lacks a supported zlib header")
            decoded = _bounded_zlib_decode_range(
                source.data,
                zlib.MAX_WBITS,
                max_output,
                "cws-zlib",
                start=encoded_start,
            )
            if len(decoded.output) != file_length - 8:
                raise DetectionError(
                    "CWS decoded length does not match FileLength"
                )
            object_end = encoded_start + decoded.consumed
            deflate_start = encoded_start + 2
            deflate_end = object_end - 4
            if deflate_start >= deflate_end:
                raise DetectionError("CWS body has no DEFLATE payload")
            evidence.append(
                CompressionEvidence(
                    category="intrinsic",
                    kind="swf-cws",
                    source=source.source_id,
                    regions=[source.region(deflate_start, deflate_end)],
                    detail={
                        "object_start": source.start + start,
                        "object_end": source.start + object_end,
                        "swf_version": version,
                        "decoded_bytes": len(decoded.output),
                        "minimum_level": 2,
                        "validation": (
                            "CWS header plus complete zlib decode to declared "
                            "SWF FileLength"
                        ),
                    },
                )
            )
            cursor = object_end
        except (DetectionError, DecompressionLimitError, ValueError) as exc:
            candidate_failures.append({
                "kind": "swf-cws",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + 3


def _read_gif_subblocks(
    data: bytes,
    position: int,
) -> Tuple[bytes, List[Tuple[int, int]], int]:
    payloads: List[bytes] = []
    spans: List[Tuple[int, int]] = []
    while True:
        if position >= len(data):
            raise DetectionError("truncated GIF data sub-block length")
        length = data[position]
        position += 1
        if length == 0:
            return b"".join(payloads), spans, position
        block_end = position + length
        if block_end > len(data):
            raise DetectionError("truncated GIF data sub-block")
        payloads.append(data[position:block_end])
        spans.append((position, block_end))
        position = block_end


def _decode_gif_lzw(
    encoded: bytes,
    minimum_code_size: int,
    expected_pixels: int,
    color_table_size: int,
    max_output: int,
) -> int:
    if not 2 <= minimum_code_size <= 8:
        raise DetectionError("GIF LZW minimum code size is outside 2..8")
    if expected_pixels <= 0:
        raise DetectionError("GIF image has no pixels")
    if color_table_size <= 0:
        raise DetectionError("GIF image has no active color table")
    if expected_pixels > max_output:
        raise DecompressionLimitError(
            f"GIF decoded image exceeds {max_output} bytes"
        )

    clear_code = 1 << minimum_code_size
    eoi_code = clear_code + 1
    bit_position = 0
    total_bits = len(encoded) * 8
    dictionary: Dict[int, bytes] = {}
    code_size = minimum_code_size + 1
    next_code = eoi_code + 1
    previous: Optional[bytes] = None
    saw_clear = False
    saw_eoi = False
    output_count = 0

    def reset_dictionary() -> None:
        nonlocal dictionary, code_size, next_code, previous
        dictionary = {value: bytes([value]) for value in range(clear_code)}
        code_size = minimum_code_size + 1
        next_code = eoi_code + 1
        previous = None

    def read_code(width: int) -> int:
        nonlocal bit_position
        if bit_position + width > total_bits:
            raise DetectionError("GIF LZW stream ends before EOI")
        value = 0
        for shift in range(width):
            byte_index = (bit_position + shift) >> 3
            bit_index = (bit_position + shift) & 7
            value |= ((encoded[byte_index] >> bit_index) & 1) << shift
        bit_position += width
        return value

    reset_dictionary()
    while bit_position + code_size <= total_bits:
        code = read_code(code_size)
        if code == clear_code:
            reset_dictionary()
            saw_clear = True
            continue
        if code == eoi_code:
            saw_eoi = True
            break
        if not saw_clear:
            raise DetectionError("GIF LZW stream does not begin with a clear code")

        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code and previous is not None:
            entry = previous + previous[:1]
        else:
            raise DetectionError("GIF LZW code is outside the active dictionary")

        output_count += len(entry)
        if output_count > expected_pixels or output_count > max_output:
            raise DetectionError("GIF LZW output exceeds declared image dimensions")
        if any(value >= color_table_size for value in entry):
            raise DetectionError("GIF LZW output indexes outside the color table")

        if previous is not None and next_code < 4096:
            dictionary[next_code] = previous + entry[:1]
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
        previous = entry

    if not saw_eoi:
        raise DetectionError("GIF LZW EOI code not found")
    if output_count != expected_pixels:
        raise DetectionError(
            f"GIF LZW pixel count mismatch: {output_count} != {expected_pixels}"
        )
    if total_bits - bit_position > 7:
        raise DetectionError("GIF LZW data contains bytes after the EOI code")
    return output_count


def _parse_gif(
    data: bytes,
    start: int,
    max_output: int,
) -> Tuple[int, List[Tuple[int, int]], int, int]:
    if data[start:start + 6] not in {b"GIF87a", b"GIF89a"}:
        raise DetectionError("GIF signature is invalid")
    if start + 13 > len(data):
        raise DetectionError("truncated GIF logical screen descriptor")

    screen_width = int.from_bytes(data[start + 6:start + 8], "little")
    screen_height = int.from_bytes(data[start + 8:start + 10], "little")
    if screen_width == 0 or screen_height == 0:
        raise DetectionError("GIF logical screen dimensions are zero")
    packed = data[start + 10]
    position = start + 13
    global_color_count = 0
    if packed & 0x80:
        global_color_count = 1 << ((packed & 0x07) + 1)
        global_table_bytes = 3 * global_color_count
        position += global_table_bytes
        if position > len(data):
            raise DetectionError("truncated GIF global color table")

    compressed_spans: List[Tuple[int, int]] = []
    image_count = 0
    decoded_pixels = 0

    while position < len(data):
        marker = data[position]
        if marker == 0x3B:  # trailer
            if image_count == 0 or not compressed_spans:
                raise DetectionError("GIF contains no compressed image data")
            return position + 1, compressed_spans, image_count, decoded_pixels

        if marker == 0x21:  # extension
            if position + 2 > len(data):
                raise DetectionError("truncated GIF extension label")
            label = data[position + 1]
            extension, _spans, position = _read_gif_subblocks(
                data,
                position + 2,
            )
            if label == 0xF9 and len(extension) != 4:
                raise DetectionError("invalid GIF graphic-control extension")
            continue

        if marker != 0x2C:
            raise DetectionError(f"unexpected GIF block marker 0x{marker:02x}")
        if position + 10 > len(data):
            raise DetectionError("truncated GIF image descriptor")

        left = int.from_bytes(data[position + 1:position + 3], "little")
        top = int.from_bytes(data[position + 3:position + 5], "little")
        width = int.from_bytes(data[position + 5:position + 7], "little")
        height = int.from_bytes(data[position + 7:position + 9], "little")
        image_packed = data[position + 9]
        if width == 0 or height == 0:
            raise DetectionError("GIF image dimensions are zero")
        if left + width > screen_width or top + height > screen_height:
            raise DetectionError("GIF image lies outside the logical screen")
        if image_packed & 0x18:
            raise DetectionError("GIF image descriptor reserved bits are set")

        position += 10
        active_color_count = global_color_count
        if image_packed & 0x80:
            active_color_count = 1 << ((image_packed & 0x07) + 1)
            local_table_bytes = 3 * active_color_count
            position += local_table_bytes
            if position > len(data):
                raise DetectionError("truncated GIF local color table")
        if position >= len(data):
            raise DetectionError("truncated GIF LZW code-size field")

        minimum_code_size = data[position]
        encoded, spans, position = _read_gif_subblocks(data, position + 1)
        if not encoded or not spans:
            raise DetectionError("GIF image has empty LZW data")
        pixels = width * height
        decoded_pixels += _decode_gif_lzw(
            encoded,
            minimum_code_size,
            pixels,
            active_color_count,
            max_output,
        )
        if decoded_pixels > max_output:
            raise DecompressionLimitError(
                f"GIF decoded images exceed {max_output} bytes in total"
            )
        compressed_spans.extend(spans)
        image_count += 1

    raise DetectionError("GIF trailer not found")


def detect_gif(
    source: SourceBuffer,
    max_output: int,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    starts = sorted({
        *list(_iter_signature_offsets(source.data, b"GIF87a")),
        *list(_iter_signature_offsets(source.data, b"GIF89a")),
    })
    accepted_until = -1
    for start in starts:
        if start < accepted_until:
            continue
        try:
            object_end, spans, image_count, decoded_pixels = _parse_gif(
                source.data,
                start,
                max_output,
            )
            evidence.append(
                CompressionEvidence(
                    category="intrinsic",
                    kind="gif",
                    source=source.source_id,
                    regions=[source.region(a, b) for a, b in spans],
                    detail={
                        "object_start": source.start + start,
                        "object_end": source.start + object_end,
                        "image_count": image_count,
                        "decoded_pixels": decoded_pixels,
                        "minimum_level": 2,
                        "validation": (
                            "complete GIF structure plus per-image LZW decode "
                            "to EOI and declared pixel count"
                        ),
                    },
                )
            )
            accepted_until = object_end
        except (DetectionError, DecompressionLimitError, ValueError) as exc:
            candidate_failures.append({
                "kind": "gif",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })


def _validate_non_overlapping_ranges(
    ranges: Sequence[Tuple[int, int, str]],
    minimum_start: int,
    maximum_end: int,
) -> None:
    previous_end = minimum_start
    for start, end, name in sorted(ranges):
        if start < minimum_start or end <= start or end > maximum_end:
            raise DetectionError(f"{name} range is outside the WOFF object")
        if start < previous_end:
            raise DetectionError(f"{name} overlaps another WOFF region")
        previous_end = end


def _sfnt_table_checksum(tag: bytes, table: bytes) -> int:
    checksum_data = bytearray(table)
    if tag == b"head" and len(checksum_data) >= 12:
        checksum_data[8:12] = b"\x00\x00\x00\x00"
    padding = (-len(checksum_data)) % 4
    if padding:
        checksum_data.extend(b"\x00" * padding)
    return sum(
        int.from_bytes(checksum_data[offset:offset + 4], "big")
        for offset in range(0, len(checksum_data), 4)
    ) & 0xFFFFFFFF


def detect_woff(
    source: SourceBuffer,
    max_output: int,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    cursor = 0
    while True:
        start = source.data.find(b"wOFF", cursor)
        if start < 0:
            return
        try:
            if start + 44 > len(source.data):
                raise DetectionError("truncated WOFF header")
            (
                signature,
                _flavor,
                length,
                num_tables,
                reserved,
                total_sfnt_size,
                _major_version,
                _minor_version,
                meta_offset,
                meta_length,
                meta_orig_length,
                priv_offset,
                priv_length,
            ) = struct.unpack_from(">4s4sIHHIHHIIIII", source.data, start)
            if signature != b"wOFF" or reserved != 0:
                raise DetectionError("invalid WOFF signature/reserved field")
            directory_end = 44 + 20 * num_tables
            if num_tables == 0 or directory_end > length:
                raise DetectionError("invalid WOFF table count or directory")
            if total_sfnt_size == 0 or start + length > len(source.data):
                raise DetectionError("invalid or incomplete WOFF object length")

            compressed_spans: List[Tuple[int, int]] = []
            occupied: List[Tuple[int, int, str]] = []
            decoded_total = 0
            calculated_sfnt_size = 12 + 16 * num_tables
            table_position = start + 44
            previous_tag: Optional[bytes] = None
            for table_index in range(num_tables):
                (
                    tag,
                    offset,
                    comp_length,
                    orig_length,
                    orig_checksum,
                ) = struct.unpack_from(">4sIIII", source.data, table_position)
                table_position += 20
                label = f"WOFF table {tag!r}#{table_index}"
                if previous_tag is not None and tag <= previous_tag:
                    raise DetectionError("WOFF table tags are not strictly sorted")
                previous_tag = tag
                if (
                    offset % 4 != 0
                    or comp_length == 0
                    or orig_length == 0
                    or comp_length > orig_length
                    or offset < directory_end
                    or offset + comp_length > length
                ):
                    raise DetectionError(f"invalid {label} lengths/alignment")
                occupied.append((offset, offset + comp_length, label))
                decoded_total += orig_length
                calculated_sfnt_size += (orig_length + 3) & ~3
                if decoded_total > max_output:
                    raise DecompressionLimitError(
                        f"WOFF decoded tables exceed {max_output} bytes"
                    )
                if comp_length == orig_length:
                    table_data = source.data[
                        start + offset:start + offset + comp_length
                    ]
                else:
                    decoded = _bounded_zlib_decode_range(
                        source.data,
                        zlib.MAX_WBITS,
                        max_output,
                        "woff-table-zlib",
                        start=start + offset,
                        end=start + offset + comp_length,
                    )
                    if (
                        decoded.consumed != comp_length
                        or len(decoded.output) != orig_length
                    ):
                        raise DetectionError(
                            f"{label} zlib length does not match its directory entry"
                        )
                    table_data = decoded.output
                    compressed_spans.append(
                        (start + offset, start + offset + comp_length)
                    )
                if _sfnt_table_checksum(tag, table_data) != orig_checksum:
                    raise DetectionError(f"{label} original checksum mismatch")

            if calculated_sfnt_size != total_sfnt_size:
                raise DetectionError(
                    "WOFF totalSfntSize does not match the table directory"
                )

            if meta_offset == 0:
                if meta_length != 0 or meta_orig_length != 0:
                    raise DetectionError("WOFF metadata lengths lack an offset")
            else:
                if (
                    meta_length == 0
                    or meta_orig_length == 0
                    or meta_offset % 4 != 0
                    or meta_offset < directory_end
                    or meta_offset + meta_length > length
                ):
                    raise DetectionError("WOFF metadata has an empty length")
                occupied.append((
                    meta_offset,
                    meta_offset + meta_length,
                    "WOFF metadata",
                ))
                decoded_total += meta_orig_length
                if decoded_total > max_output:
                    raise DecompressionLimitError(
                        f"WOFF decoded content exceeds {max_output} bytes"
                    )
                decoded = _bounded_zlib_decode_range(
                    source.data,
                    zlib.MAX_WBITS,
                    max_output,
                    "woff-metadata-zlib",
                    start=start + meta_offset,
                    end=start + meta_offset + meta_length,
                )
                if (
                    decoded.consumed != meta_length
                    or len(decoded.output) != meta_orig_length
                ):
                    raise DetectionError(
                        "WOFF metadata zlib length does not match the header"
                    )
                compressed_spans.append((
                    start + meta_offset,
                    start + meta_offset + meta_length,
                ))

            if priv_offset == 0:
                if priv_length != 0:
                    raise DetectionError("WOFF private length lacks an offset")
            else:
                if (
                    priv_length == 0
                    or priv_offset % 4 != 0
                    or priv_offset < directory_end
                    or priv_offset + priv_length > length
                ):
                    raise DetectionError("WOFF private block has an empty length")
                occupied.append((
                    priv_offset,
                    priv_offset + priv_length,
                    "WOFF private data",
                ))

            _validate_non_overlapping_ranges(occupied, directory_end, length)
            object_end = start + length
            if compressed_spans:
                evidence.append(
                    CompressionEvidence(
                        category="intrinsic",
                        kind="woff",
                        source=source.source_id,
                        regions=[
                            source.region(a, b) for a, b in compressed_spans
                        ],
                        detail={
                            "object_start": source.start + start,
                            "object_end": source.start + object_end,
                            "compressed_regions": len(compressed_spans),
                            "decoded_bytes": decoded_total,
                            "minimum_level": 2,
                            "validation": (
                                "complete WOFF1 directory plus zlib decode of "
                                "every compressed table/metadata region"
                            ),
                        },
                    )
                )
            cursor = object_end
        except (
            DetectionError,
            DecompressionLimitError,
            ValueError,
            struct.error,
        ) as exc:
            candidate_failures.append({
                "kind": "woff",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + 4


def _parse_zstd_frame(
    data: bytes,
    start: int,
) -> Tuple[int, List[Tuple[int, int]], int, bool]:
    if data[start:start + 4] != ZSTD_FRAME_MAGIC:
        raise DetectionError("Zstandard frame magic is invalid")
    if start + 5 > len(data):
        raise DetectionError("truncated Zstandard frame header")

    descriptor = data[start + 4]
    frame_content_size_flag = descriptor >> 6
    single_segment = bool(descriptor & 0x20)
    if descriptor & 0x18:
        raise DetectionError("Zstandard frame unused/reserved bit is set")
    content_checksum = bool(descriptor & 0x04)
    dictionary_id_flag = descriptor & 0x03

    position = start + 5
    window_size: Optional[int] = None
    if not single_segment:
        if position >= len(data):
            raise DetectionError("truncated Zstandard window descriptor")
        window_descriptor = data[position]
        window_log = 10 + (window_descriptor >> 3)
        window_base = 1 << window_log
        window_size = window_base + (window_base >> 3) * (
            window_descriptor & 0x07
        )
        position += 1

    dictionary_id_size = (0, 1, 2, 4)[dictionary_id_flag]
    frame_content_size_size = (
        (1 if single_segment else 0)
        if frame_content_size_flag == 0
        else (2 if frame_content_size_flag == 1 else (
            4 if frame_content_size_flag == 2 else 8
        ))
    )
    frame_content_size_start = position + dictionary_id_size
    header_end = frame_content_size_start + frame_content_size_size
    if header_end > len(data):
        raise DetectionError("truncated Zstandard optional frame fields")
    if frame_content_size_size:
        frame_content_size = int.from_bytes(
            data[frame_content_size_start:header_end],
            "little",
        )
        if frame_content_size_size == 2:
            frame_content_size += 256
        if single_segment:
            window_size = frame_content_size
    if window_size is None:
        raise DetectionError("Zstandard frame has no derivable window size")
    maximum_block_size = min(window_size, 128 * 1024)
    position = header_end

    compressed_spans: List[Tuple[int, int]] = []
    block_count = 0
    while True:
        if position + 3 > len(data):
            raise DetectionError("truncated Zstandard block header")
        block_header = int.from_bytes(data[position:position + 3], "little")
        position += 3
        last_block = bool(block_header & 0x01)
        block_type = (block_header >> 1) & 0x03
        block_size = block_header >> 3
        if block_type == 3:
            raise DetectionError("Zstandard block uses the reserved type")
        if block_size > maximum_block_size:
            raise DetectionError(
                "Zstandard block exceeds the frame block/window limit"
            )

        if block_type == 1:  # RLE: one encoded byte expands to block_size bytes.
            if block_size == 0:
                raise DetectionError("Zstandard RLE block has zero output size")
            encoded_size = 1
        else:
            encoded_size = block_size
        block_end = position + encoded_size
        if block_end > len(data):
            raise DetectionError("truncated Zstandard block payload")
        if block_type in {1, 2} and encoded_size:
            compressed_spans.append((position, block_end))
        position = block_end
        block_count += 1
        if last_block:
            break

    if content_checksum:
        if position + 4 > len(data):
            raise DetectionError("truncated Zstandard content checksum")
        position += 4
    return position, compressed_spans, block_count, content_checksum


def detect_zstd_frames(
    source: SourceBuffer,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    cursor = 0
    while True:
        start = source.data.find(ZSTD_FRAME_MAGIC, cursor)
        if start < 0:
            return
        try:
            object_end, spans, block_count, has_checksum = _parse_zstd_frame(
                source.data,
                start,
            )
            if spans:
                evidence.append(
                    CompressionEvidence(
                        category="intrinsic",
                        kind="zstd-frame",
                        source=source.source_id,
                        regions=[source.region(a, b) for a, b in spans],
                        detail={
                            "object_start": source.start + start,
                            "object_end": source.start + object_end,
                            "block_count": block_count,
                            "compressed_or_rle_blocks": len(spans),
                            "content_checksum_present": has_checksum,
                            "minimum_level": 3,
                            "validation": (
                                "complete Zstandard frame/block structure; "
                                "compressed payload not decoded"
                            ),
                        },
                    )
                )
            cursor = object_end
        except (DetectionError, ValueError) as exc:
            candidate_failures.append({
                "kind": "zstd-frame",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + len(ZSTD_FRAME_MAGIC)


def _rotate_left_32(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (32 - bits))) & 0xFFFFFFFF


def _xxh32(data: bytes, seed: int = 0) -> int:
    """Small dependency-free XXH32 implementation for LZ4 checksums."""

    prime1 = 0x9E3779B1
    prime2 = 0x85EBCA77
    prime3 = 0xC2B2AE3D
    prime4 = 0x27D4EB2F
    prime5 = 0x165667B1
    length = len(data)
    position = 0

    def round_value(accumulator: int, lane: int) -> int:
        accumulator = (accumulator + lane * prime2) & 0xFFFFFFFF
        accumulator = _rotate_left_32(accumulator, 13)
        return (accumulator * prime1) & 0xFFFFFFFF

    if length >= 16:
        v1 = (seed + prime1 + prime2) & 0xFFFFFFFF
        v2 = (seed + prime2) & 0xFFFFFFFF
        v3 = seed & 0xFFFFFFFF
        v4 = (seed - prime1) & 0xFFFFFFFF
        limit = length - 16
        while position <= limit:
            v1 = round_value(v1, int.from_bytes(
                data[position:position + 4], "little"
            ))
            position += 4
            v2 = round_value(v2, int.from_bytes(
                data[position:position + 4], "little"
            ))
            position += 4
            v3 = round_value(v3, int.from_bytes(
                data[position:position + 4], "little"
            ))
            position += 4
            v4 = round_value(v4, int.from_bytes(
                data[position:position + 4], "little"
            ))
            position += 4
        value = (
            _rotate_left_32(v1, 1)
            + _rotate_left_32(v2, 7)
            + _rotate_left_32(v3, 12)
            + _rotate_left_32(v4, 18)
        ) & 0xFFFFFFFF
    else:
        value = (seed + prime5) & 0xFFFFFFFF

    value = (value + length) & 0xFFFFFFFF
    while position + 4 <= length:
        lane = int.from_bytes(data[position:position + 4], "little")
        value = (value + lane * prime3) & 0xFFFFFFFF
        value = (_rotate_left_32(value, 17) * prime4) & 0xFFFFFFFF
        position += 4
    while position < length:
        value = (value + data[position] * prime5) & 0xFFFFFFFF
        value = (_rotate_left_32(value, 11) * prime1) & 0xFFFFFFFF
        position += 1

    value ^= value >> 15
    value = (value * prime2) & 0xFFFFFFFF
    value ^= value >> 13
    value = (value * prime3) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def _parse_lz4_frame(
    data: bytes,
    start: int,
) -> Tuple[int, List[Tuple[int, int]], int, bool, bool]:
    if data[start:start + 4] != LZ4_FRAME_MAGIC:
        raise DetectionError("LZ4 frame magic is invalid")
    if start + 7 > len(data):
        raise DetectionError("truncated LZ4 frame header")

    descriptor_start = start + 4
    flags = data[descriptor_start]
    block_descriptor = data[descriptor_start + 1]
    if flags & 0xC0 != 0x40 or flags & 0x02:
        raise DetectionError("invalid LZ4 frame version/reserved flags")
    if block_descriptor & 0x8F:
        raise DetectionError("LZ4 block descriptor reserved bits are set")
    maximum_code = (block_descriptor >> 4) & 0x07
    maximum_sizes = {
        4: 64 * 1024,
        5: 256 * 1024,
        6: 1024 * 1024,
        7: 4 * 1024 * 1024,
    }
    if maximum_code not in maximum_sizes:
        raise DetectionError("invalid LZ4 maximum block-size code")

    block_checksum = bool(flags & 0x10)
    content_size_present = bool(flags & 0x08)
    content_checksum = bool(flags & 0x04)
    dictionary_id_present = bool(flags & 0x01)
    position = descriptor_start + 2
    if content_size_present:
        position += 8
    if dictionary_id_present:
        position += 4
    if position >= len(data):
        raise DetectionError("truncated LZ4 optional frame fields")
    expected_header_checksum = data[position]
    actual_header_checksum = (_xxh32(data[descriptor_start:position]) >> 8) & 0xFF
    if expected_header_checksum != actual_header_checksum:
        raise DetectionError("LZ4 frame header checksum mismatch")
    position += 1

    compressed_spans: List[Tuple[int, int]] = []
    block_count = 0
    while True:
        if position + 4 > len(data):
            raise DetectionError("truncated LZ4 block-size field")
        raw_size = int.from_bytes(data[position:position + 4], "little")
        position += 4
        if raw_size == 0:
            break
        uncompressed = bool(raw_size & 0x80000000)
        block_size = raw_size & 0x7FFFFFFF
        if (
            (block_size == 0 and not uncompressed)
            or block_size > maximum_sizes[maximum_code]
        ):
            raise DetectionError("invalid LZ4 block length")
        block_end = position + block_size
        if block_end > len(data):
            raise DetectionError("truncated LZ4 block payload")
        block_payload = data[position:block_end]
        if not uncompressed:
            compressed_spans.append((position, block_end))
        position = block_end
        if block_checksum:
            if position + 4 > len(data):
                raise DetectionError("truncated LZ4 block checksum")
            expected = int.from_bytes(data[position:position + 4], "little")
            if _xxh32(block_payload) != expected:
                raise DetectionError("LZ4 block checksum mismatch")
            position += 4
        block_count += 1

    if content_checksum:
        if position + 4 > len(data):
            raise DetectionError("truncated LZ4 content checksum")
        # Validation would require decompression.  Its presence and exact
        # boundary are recorded, but Level 3 deliberately does not decode it.
        position += 4
    return (
        position,
        compressed_spans,
        block_count,
        block_checksum,
        content_checksum,
    )


def detect_lz4_frames(
    source: SourceBuffer,
    evidence: List[CompressionEvidence],
    candidate_failures: List[Dict[str, Any]],
) -> None:
    cursor = 0
    while True:
        start = source.data.find(LZ4_FRAME_MAGIC, cursor)
        if start < 0:
            return
        try:
            (
                object_end,
                spans,
                block_count,
                block_checksum,
                content_checksum,
            ) = _parse_lz4_frame(source.data, start)
            if spans:
                evidence.append(
                    CompressionEvidence(
                        category="intrinsic",
                        kind="lz4-frame",
                        source=source.source_id,
                        regions=[source.region(a, b) for a, b in spans],
                        detail={
                            "object_start": source.start + start,
                            "object_end": source.start + object_end,
                            "block_count": block_count,
                            "compressed_blocks": len(spans),
                            "block_checksum_present": block_checksum,
                            "content_checksum_present": content_checksum,
                            "minimum_level": 3,
                            "validation": (
                                "complete LZ4 frame structure and header/block "
                                "checksums when present; payload not decoded"
                            ),
                        },
                    )
                )
            cursor = object_end
        except (DetectionError, ValueError) as exc:
            candidate_failures.append({
                "kind": "lz4-frame",
                "source": source.source_id,
                "offset": source.start + start,
                "reason": str(exc),
            })
            cursor = start + len(LZ4_FRAME_MAGIC)


def detect_intrinsic_compression(
    source: SourceBuffer,
    max_output: int,
    compression_level: int,
    evidence: List[CompressionEvidence],
    unsupported: List[Dict[str, Any]],
    candidate_failures: List[Dict[str, Any]],
    analysis_errors: List[str],
) -> None:
    detectors = (
        ("gzip", lambda: detect_gzip(
            source, max_output, evidence, candidate_failures
        )),
        ("zip", lambda: detect_zip(
            source,
            max_output,
            compression_level,
            evidence,
            unsupported,
            candidate_failures,
        )),
        ("png", lambda: detect_png(
            source, max_output, evidence, candidate_failures
        )),
        ("jpeg", lambda: detect_jpeg(
            source, evidence, candidate_failures
        )),
        ("webp", lambda: detect_webp(
            source, evidence, candidate_failures
        )),
    )
    for detector_name, detector in detectors:
        try:
            detector()
        except Exception as exc:  # unexpected detector bug, kept visible in output
            analysis_errors.append(
                f"{detector_name}@{source.source_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    if compression_level >= 2:
        extended_detectors = (
            ("gif", lambda: detect_gif(
                source, max_output, evidence, candidate_failures
            )),
            ("woff", lambda: detect_woff(
                source, max_output, evidence, candidate_failures
            )),
            ("swf-cws", lambda: detect_cws(
                source, max_output, evidence, candidate_failures
            )),
            ("zlib", lambda: detect_zlib_streams(
                source, max_output, evidence
            )),
            ("bzip2", lambda: detect_bzip2_streams(
                source, max_output, evidence, candidate_failures
            )),
            ("xz", lambda: detect_xz_streams(
                source, max_output, evidence, candidate_failures
            )),
        )
        for detector_name, detector in extended_detectors:
            try:
                detector()
            except Exception as exc:
                analysis_errors.append(
                    f"{detector_name}@{source.source_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

    if compression_level >= 3:
        declared_detectors = (
            ("zstd-frame", lambda: detect_zstd_frames(
                source, evidence, candidate_failures
            )),
            ("lz4-frame", lambda: detect_lz4_frames(
                source, evidence, candidate_failures
            )),
        )
        for detector_name, detector in declared_detectors:
            try:
                detector()
            except Exception as exc:
                analysis_errors.append(
                    f"{detector_name}@{source.source_id}: "
                    f"{type(exc).__name__}: {exc}"
                )


def _map_dechunked_region(
    raw_spans: Sequence[Tuple[int, int]],
    logical_start: int,
    logical_end: int,
    source: SourceBuffer,
) -> List[ByteRegion]:
    """Map one interval in a de-chunked body back to TCP stream intervals."""

    if logical_start < 0 or logical_end <= logical_start:
        raise DetectionError("invalid de-chunked intrinsic region")
    logical_cursor = 0
    mapped: List[ByteRegion] = []
    covered = 0

    for raw_start, raw_end in raw_spans:
        span_length = raw_end - raw_start
        span_logical_start = logical_cursor
        span_logical_end = logical_cursor + span_length
        overlap_start = max(logical_start, span_logical_start)
        overlap_end = min(logical_end, span_logical_end)
        if overlap_start < overlap_end:
            mapped_start = raw_start + (overlap_start - span_logical_start)
            mapped_end = raw_start + (overlap_end - span_logical_start)
            mapped.append(source.region(mapped_start, mapped_end))
            covered += overlap_end - overlap_start
        logical_cursor = span_logical_end
        if logical_cursor >= logical_end:
            break

    if covered != logical_end - logical_start:
        raise DetectionError("de-chunked intrinsic region mapping is incomplete")
    return mapped


def detect_chunked_intrinsic_compression(
    source: SourceBuffer,
    max_output: int,
    max_header_bytes: int,
    compression_level: int,
    evidence: List[CompressionEvidence],
    unsupported: List[Dict[str, Any]],
    candidate_failures: List[Dict[str, Any]],
    analysis_errors: List[str],
) -> None:
    """Validate intrinsic objects carried across plain HTTP chunk boundaries.

    Chunk-size lines and CRLF delimiters are not part of the application object.
    This pass is limited to identity content coding and identity/chunked transfer
    coding; once a content coding is present, decoded bytes cannot be mapped
    one-to-one back to wire offsets.
    """

    if source.carrier != "tcp":
        return

    for start_match in _HTTP_START_LINE.finditer(source.data):
        header = _parse_http_header_at(
            source.data,
            start_match.start(),
            max_header_bytes,
        )
        if header is None:
            continue
        content_codings = _split_http_codings(
            header.headers.get("content-encoding", [])
        )
        transfer_codings = _split_http_codings(
            header.headers.get("transfer-encoding", [])
        )
        if not transfer_codings or transfer_codings[-1] != "chunked":
            continue
        if any(value != "identity" for value in content_codings):
            continue
        if any(value not in {"identity", "chunked"} for value in transfer_codings):
            continue

        try:
            body = _parse_chunked_body(source.data, header.body_start)
            if not body.encoded or not body.spans:
                continue
        except DetectionError as exc:
            candidate_failures.append({
                "kind": "http-chunked-intrinsic",
                "source": source.source_id,
                "stream_offset": source.start + header.start,
                "reason": str(exc),
            })
            continue

        logical_source = SourceBuffer(
            carrier="logical-http-body",
            direction=source.direction,
            start=0,
            data=body.encoded,
            source_id=(
                f"{source.source_id}:dechunked-http@"
                f"{source.start + header.start}"
            ),
        )
        logical_evidence: List[CompressionEvidence] = []
        logical_unsupported: List[Dict[str, Any]] = []
        logical_failures: List[Dict[str, Any]] = []
        detect_intrinsic_compression(
            logical_source,
            max_output,
            compression_level,
            logical_evidence,
            logical_unsupported,
            logical_failures,
            analysis_errors,
        )

        for item in logical_evidence:
            mapped_regions: List[ByteRegion] = []
            try:
                for region in item.regions:
                    mapped_regions.extend(
                        _map_dechunked_region(
                            body.spans,
                            region.start,
                            region.end,
                            source,
                        )
                    )
            except DetectionError as exc:
                analysis_errors.append(
                    f"chunk-map@{logical_source.source_id}: {exc}"
                )
                continue

            detail = dict(item.detail)
            if "object_start" in detail:
                detail["logical_object_start"] = detail.pop("object_start")
            if "object_end" in detail:
                detail["logical_object_end"] = detail.pop("object_end")
            detail.update({
                "http_header_stream_offset": source.start + header.start,
                "wire_mapping": "dechunked body bytes; chunk framing excluded",
            })
            evidence.append(
                CompressionEvidence(
                    category=item.category,
                    kind=item.kind,
                    source=logical_source.source_id,
                    regions=mapped_regions,
                    detail=detail,
                )
            )

        for finding in logical_unsupported:
            enriched = dict(finding)
            enriched["coordinate_space"] = "dechunked-http-body"
            unsupported.append(enriched)
        for failure in logical_failures:
            enriched = dict(failure)
            enriched["coordinate_space"] = "dechunked-http-body"
            candidate_failures.append(enriched)


def _deduplicate_evidence(
    evidence: Sequence[CompressionEvidence],
) -> List[CompressionEvidence]:
    """Remove exact duplicate kind/category/region findings from overlapping passes."""

    result: List[CompressionEvidence] = []
    seen: Set[Tuple[Any, ...]] = set()
    for item in evidence:
        region_key = tuple(sorted(
            (
                region.carrier,
                region.direction,
                -1 if region.packet_index is None else region.packet_index,
                region.start,
                region.end,
            )
            for region in item.regions
        ))
        key = (item.category, item.kind, region_key)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _region_covers(outer: ByteRegion, inner: ByteRegion) -> bool:
    return (
        outer.carrier == inner.carrier
        and outer.direction == inner.direction
        and outer.packet_index == inner.packet_index
        and outer.start <= inner.start
        and inner.end <= outer.end
    )


def _suppress_nested_generic_streams(
    evidence: Sequence[CompressionEvidence],
) -> List[CompressionEvidence]:
    """Prefer a validated container detector over its generic inner stream.

    For example, a single PNG IDAT chunk can also look like a standalone zlib
    stream.  Keeping both does not change the byte union, but obscures evidence
    counts and format summaries.  A more weakly validated container never hides
    a stronger generic decode.
    """

    generic_kinds = {"zlib", "bzip2", "xz"}
    result: List[CompressionEvidence] = []
    for item in evidence:
        if item.category != "intrinsic" or item.kind not in generic_kinds:
            result.append(item)
            continue
        item_level = int(item.detail.get("minimum_level", 1))
        covering_regions = [
            region
            for other in evidence
            if (
                other is not item
                and other.category == "intrinsic"
                and other.kind not in generic_kinds
                and int(other.detail.get("minimum_level", 1)) <= item_level
            )
            for region in other.regions
        ]
        if item.regions and all(
            any(_region_covers(outer, inner) for outer in covering_regions)
            for inner in item.regions
        ):
            continue
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Per-flow metrics and dataset aggregation
# ---------------------------------------------------------------------------


FLOW_FIELDS = [
    "dataset",
    "label",
    "relative_pcap",
    "pcap_path",
    "status",
    "error",
    "protocol",
    "payload_packets",
    "window_bytes",
    "unmappable_window_bytes",
    "tcp_reassembled_bytes",
    "tcp_contiguous_chunks",
    "udp_datagrams",
    "payload_entropy_bits",
    "payload_normalized_entropy",
    "payload_r_stat",
    "model_entropy_bits",
    "model_normalized_entropy",
    "model_r_stat",
    "has_encrypted_transport_evidence",
    "encrypted_transport_evidence",
    "protocol_compressed",
    "intrinsic_compressed",
    "both_compression_categories",
    "confirmed_compressed",
    "protocol_algorithms",
    "intrinsic_formats",
    "evidence_count",
    "protocol_window_bytes",
    "intrinsic_window_bytes",
    "compressed_window_bytes",
    "compressed_reaches_window",
    "e_i",
    "e_bin",
    "unsupported_compression_count",
    "unsupported_compression",
    "candidate_failure_count",
    "candidate_failures",
    "analysis_error_count",
    "analysis_errors",
    "compression_evidence",
]


BIN_DEFINITIONS = [
    {
        "id": "e_eq_0",
        "display": "e_i = 0",
        "lower": 0.0,
        "lower_inclusive": True,
        "upper": 0.0,
        "upper_inclusive": True,
    },
    {
        "id": "e_gt_0_le_0_25",
        "display": "0 < e_i <= 0.25",
        "lower": 0.0,
        "lower_inclusive": False,
        "upper": 0.25,
        "upper_inclusive": True,
    },
    {
        "id": "e_gt_0_25_le_0_50",
        "display": "0.25 < e_i <= 0.50",
        "lower": 0.25,
        "lower_inclusive": False,
        "upper": 0.50,
        "upper_inclusive": True,
    },
    {
        "id": "e_gt_0_50_le_1",
        "display": "0.50 < e_i <= 1.00",
        "lower": 0.50,
        "lower_inclusive": False,
        "upper": 1.00,
        "upper_inclusive": True,
    },
]


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _shannon_entropy_bits(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    length = len(values)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def compute_entropy_metrics(window: bytes) -> Dict[str, float]:
    """Return interpretable payload entropy and exact model-side r_stat.

    ``payload_*`` excludes CLS/SEP/PAD and uses only W_i.  ``model_*`` mirrors
    ``compute_flow_reliability_raw``: [CLS] + W_i + [SEP], PAD excluded, with
    H_max = log(min(n, 261)).  Base-2 logs are equivalent to the model's natural
    logs after normalization.
    """

    payload_values = list(window)
    payload_entropy = _shannon_entropy_bits(payload_values)
    payload_n = len(payload_values)
    payload_hmax = math.log2(min(payload_n, 256)) if payload_n > 1 else 0.0
    payload_normalized = (
        min(1.0, max(0.0, payload_entropy / payload_hmax))
        if payload_hmax > 0
        else 0.0
    )

    model_values = [CLS_TOKEN_ID]
    model_values.extend(value + BYTE_TOKEN_OFFSET for value in payload_values)
    model_values.append(SEP_TOKEN_ID)
    model_entropy = _shannon_entropy_bits(model_values)
    model_n = len(model_values)
    model_hmax = math.log2(min(model_n, RAW_VOCAB_SIZE)) if model_n > 1 else 0.0
    model_normalized = (
        min(1.0, max(0.0, model_entropy / model_hmax))
        if model_hmax > 0
        else 0.0
    )

    return {
        "payload_entropy_bits": payload_entropy,
        "payload_normalized_entropy": payload_normalized,
        "payload_r_stat": 1.0 - payload_normalized,
        "model_entropy_bits": model_entropy,
        "model_normalized_entropy": model_normalized,
        "model_r_stat": 1.0 - model_normalized,
    }


def _window_hit_indices(
    refs: Sequence[WindowByteRef],
    evidence: Sequence[CompressionEvidence],
) -> Set[int]:
    regions = [region for item in evidence for region in item.regions]
    if not regions:
        return set()
    return {
        index
        for index, ref in enumerate(refs)
        if any(region.contains(ref) for region in regions)
    }


def _e_bin(hit_count: int, window_length: int) -> str:
    if window_length <= 0:
        raise ValueError("cannot bin e_i with an empty model window")
    if hit_count == 0:
        return "e_eq_0"
    if 4 * hit_count <= window_length:
        return "e_gt_0_le_0_25"
    if 2 * hit_count <= window_length:
        return "e_gt_0_25_le_0_50"
    return "e_gt_0_50_le_1"


def _empty_flow_row(
    dataset_name: str,
    dataset_dir: Path,
    pcap_path: Path,
) -> Dict[str, Any]:
    row = {field_name: "" for field_name in FLOW_FIELDS}
    row.update({
        "dataset": dataset_name,
        "label": pcap_path.parent.name,
        "relative_pcap": str(pcap_path.relative_to(dataset_dir)),
        "pcap_path": str(pcap_path.resolve()),
        "status": "invalid",
        "protocol_compressed": 0,
        "intrinsic_compressed": 0,
        "both_compression_categories": 0,
        "confirmed_compressed": 0,
        "compressed_reaches_window": 0,
        "has_encrypted_transport_evidence": 0,
        "unsupported_compression_count": 0,
        "candidate_failure_count": 0,
        "analysis_error_count": 0,
        "evidence_count": 0,
    })
    return row


def analyze_pcap(
    dataset_name: str,
    dataset_dir: Path,
    pcap_path: Path,
    max_output: int,
    max_header_bytes: int,
    compression_level: int,
) -> Dict[str, Any]:
    row = _empty_flow_row(dataset_name, dataset_dir, pcap_path)
    try:
        parsed = read_pcap_for_audit(pcap_path)
    except (ValueError, OSError) as exc:
        row["error"] = str(exc)
        return row

    evidence: List[CompressionEvidence] = []
    unsupported: List[Dict[str, Any]] = []
    candidate_failures: List[Dict[str, Any]] = []
    analysis_errors: List[str] = []

    for source in parsed.tcp_buffers:
        try:
            detect_http_compression(
                source,
                max_output,
                max_header_bytes,
                compression_level,
                evidence,
                unsupported,
                candidate_failures,
            )
        except Exception as exc:  # keep a detector bug visible without losing the flow
            analysis_errors.append(
                f"http@{source.source_id}: {type(exc).__name__}: {exc}"
            )

    for source in parsed.tcp_buffers + parsed.udp_buffers:
        detect_intrinsic_compression(
            source,
            max_output,
            compression_level,
            evidence,
            unsupported,
            candidate_failures,
            analysis_errors,
        )

    for source in parsed.tcp_buffers:
        try:
            detect_chunked_intrinsic_compression(
                source,
                max_output,
                max_header_bytes,
                compression_level,
                evidence,
                unsupported,
                candidate_failures,
                analysis_errors,
            )
        except Exception as exc:
            analysis_errors.append(
                f"chunked-intrinsic@{source.source_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    evidence = _suppress_nested_generic_streams(evidence)
    evidence = _deduplicate_evidence(evidence)

    protocol_evidence = [item for item in evidence if item.category == "protocol"]
    intrinsic_evidence = [item for item in evidence if item.category == "intrinsic"]
    protocol_hits = _window_hit_indices(parsed.window_refs, protocol_evidence)
    intrinsic_hits = _window_hit_indices(parsed.window_refs, intrinsic_evidence)
    union_hits = protocol_hits | intrinsic_hits

    protocol_compressed = bool(protocol_evidence)
    intrinsic_compressed = bool(intrinsic_evidence)
    confirmed_compressed = protocol_compressed or intrinsic_compressed
    entropy = compute_entropy_metrics(parsed.window_bytes)
    window_length = len(parsed.window_bytes)
    exposure = len(union_hits) / window_length
    exposure_bin = _e_bin(len(union_hits), window_length) if confirmed_compressed else ""

    row.update({
        "status": "valid",
        "error": "",
        "protocol": parsed.protocol_name,
        "payload_packets": parsed.payload_packet_count,
        "window_bytes": window_length,
        "unmappable_window_bytes": sum(
            1 for ref in parsed.window_refs if not ref.canonical
        ),
        "tcp_reassembled_bytes": sum(len(item.data) for item in parsed.tcp_buffers),
        "tcp_contiguous_chunks": len(parsed.tcp_buffers),
        "udp_datagrams": len(parsed.udp_buffers),
        **entropy,
        "has_encrypted_transport_evidence": int(
            bool(parsed.encrypted_transport_evidence)
        ),
        "encrypted_transport_evidence": ";".join(
            parsed.encrypted_transport_evidence
        ),
        "protocol_compressed": int(protocol_compressed),
        "intrinsic_compressed": int(intrinsic_compressed),
        "both_compression_categories": int(
            protocol_compressed and intrinsic_compressed
        ),
        "confirmed_compressed": int(confirmed_compressed),
        "protocol_algorithms": ";".join(sorted({
            item.kind for item in protocol_evidence
        })),
        "intrinsic_formats": ";".join(sorted({
            item.kind for item in intrinsic_evidence
        })),
        "evidence_count": len(evidence),
        "protocol_window_bytes": len(protocol_hits),
        "intrinsic_window_bytes": len(intrinsic_hits),
        "compressed_window_bytes": len(union_hits),
        "compressed_reaches_window": int(bool(union_hits)),
        "e_i": exposure if confirmed_compressed else "",
        "e_bin": exposure_bin,
        "unsupported_compression_count": len(unsupported),
        "unsupported_compression": _json_cell(unsupported),
        "candidate_failure_count": len(candidate_failures),
        "candidate_failures": _json_cell(candidate_failures),
        "analysis_error_count": len(analysis_errors),
        "analysis_errors": _json_cell(analysis_errors),
        "compression_evidence": _json_cell([
            item.to_dict() for item in evidence
        ]),
    })
    return row


def _report_progress(
    completed: int,
    total: int,
    pcap_path: Path,
    quiet: bool,
) -> None:
    if quiet or not (
        completed == 1 or completed % 100 == 0 or completed == total
    ):
        return
    print(
        f"[compression-audit] completed {completed}/{total}: {pcap_path}",
        flush=True,
    )


def analyze_pcaps(
    dataset_name: str,
    dataset_dir: Path,
    pcaps: Sequence[Path],
    max_output: int,
    max_header_bytes: int,
    compression_level: int,
    workers: int,
    quiet: bool,
) -> List[Dict[str, Any]]:
    """Analyze independent PCAPs with bounded process-level parallelism.

    Only a small multiple of ``workers`` is submitted at once.  This avoids a
    large ``Future`` queue when a dataset contains hundreds of thousands of
    flows.  Rows are restored to deterministic discovery order before output.
    """

    total = len(pcaps)
    if workers == 1 or total == 1:
        rows: List[Dict[str, Any]] = []
        for completed, pcap_path in enumerate(pcaps, start=1):
            rows.append(
                analyze_pcap(
                    dataset_name,
                    dataset_dir,
                    pcap_path,
                    max_output,
                    max_header_bytes,
                    compression_level,
                )
            )
            _report_progress(completed, total, pcap_path, quiet)
        return rows

    ordered: List[Optional[Dict[str, Any]]] = [None] * total
    pending: Dict[Future, Tuple[int, Path]] = {}
    pcap_iterator = iter(enumerate(pcaps))
    max_pending = min(total, workers * PENDING_TASKS_PER_WORKER)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        def submit_next() -> bool:
            try:
                index, pcap_path = next(pcap_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                analyze_pcap,
                dataset_name,
                dataset_dir,
                pcap_path,
                max_output,
                max_header_bytes,
                compression_level,
            )
            pending[future] = (index, pcap_path)
            return True

        for _ in range(max_pending):
            submit_next()

        completed = 0
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index, pcap_path = pending.pop(future)
                try:
                    ordered[index] = future.result()
                except BaseException as exc:
                    for remaining in pending:
                        remaining.cancel()
                    raise RuntimeError(
                        f"parallel analysis failed for {pcap_path}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                completed += 1
                _report_progress(completed, total, pcap_path, quiet)
                submit_next()

    rows = []
    for index, row in enumerate(ordered):
        if row is None:
            raise RuntimeError(f"missing analysis result for {pcaps[index]}")
        rows.append(row)
    return rows


def discover_pcaps(dataset_dir: Path) -> List[Path]:
    """Discover exactly DATASET/label/*.{pcap,pcapng}, without recursion."""

    pcaps: List[Path] = []
    for label_dir in sorted(dataset_dir.iterdir(), key=lambda path: path.name):
        if not label_dir.is_dir():
            continue
        for candidate in sorted(label_dir.iterdir(), key=lambda path: path.name):
            if candidate.is_file() and candidate.suffix.lower() in PCAP_SUFFIXES:
                pcaps.append(candidate)
    return pcaps


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _mean(rows: Sequence[Dict[str, Any]], field_name: str) -> Optional[float]:
    if not rows:
        return None
    return statistics.fmean(float(row[field_name]) for row in rows)


def build_exposure_rows(
    dataset_name: str,
    valid_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    confirmed = [row for row in valid_rows if row["confirmed_compressed"] == 1]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in confirmed:
        grouped[str(row["e_bin"])].append(row)

    result: List[Dict[str, Any]] = []
    for definition in BIN_DEFINITIONS:
        bin_rows = grouped.get(definition["id"], [])
        result.append({
            "dataset": dataset_name,
            "e_bin": definition["id"],
            "e_bin_display": definition["display"],
            "lower_bound": definition["lower"],
            "lower_inclusive": int(definition["lower_inclusive"]),
            "upper_bound": definition["upper"],
            "upper_inclusive": int(definition["upper_inclusive"]),
            "flow_count": len(bin_rows),
            "flow_fraction_of_confirmed": _ratio(len(bin_rows), len(confirmed)),
            "mean_e_i": _mean(bin_rows, "e_i"),
            "mean_entropy_bits": _mean(bin_rows, "payload_entropy_bits"),
            "mean_normalized_entropy": _mean(
                bin_rows, "payload_normalized_entropy"
            ),
            "mean_model_entropy_bits": _mean(bin_rows, "model_entropy_bits"),
            "mean_model_r_stat": _mean(bin_rows, "model_r_stat"),
        })

    if sum(row["flow_count"] for row in result) != len(confirmed):
        raise RuntimeError("e_i bin counts do not sum to confirmed compressed flows")
    return result


def build_summary_row(
    dataset_name: str,
    dataset_dir: Path,
    all_rows: Sequence[Dict[str, Any]],
    exposure_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    valid = [row for row in all_rows if row["status"] == "valid"]
    confirmed = [row for row in valid if row["confirmed_compressed"] == 1]
    protocol = [row for row in valid if row["protocol_compressed"] == 1]
    intrinsic = [row for row in valid if row["intrinsic_compressed"] == 1]
    both = [row for row in valid if row["both_compression_categories"] == 1]
    reaches_window = [row for row in confirmed if row["compressed_reaches_window"] == 1]

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_dir.resolve()),
        "total_pcaps": len(all_rows),
        "valid_flows": len(valid),
        "invalid_flows": len(all_rows) - len(valid),
        "flows_with_analysis_errors": sum(
            1 for row in valid if int(row["analysis_error_count"]) > 0
        ),
        "encrypted_transport_evidence_flows": sum(
            int(row["has_encrypted_transport_evidence"]) for row in valid
        ),
        "unsupported_compression_evidence_flows": sum(
            1 for row in valid if int(row["unsupported_compression_count"]) > 0
        ),
        "protocol_compressed": len(protocol),
        "intrinsic_compressed": len(intrinsic),
        "both_compression_categories": len(both),
        "confirmed_compressed": len(confirmed),
        "compression_rate": _ratio(len(confirmed), len(valid)),
        "protocol_compression_rate": _ratio(len(protocol), len(valid)),
        "intrinsic_compression_rate": _ratio(len(intrinsic), len(valid)),
        "compressed_in_window": len(reaches_window),
        "window_exposure_rate": _ratio(len(reaches_window), len(confirmed)),
    }

    for exposure in exposure_rows:
        prefix = str(exposure["e_bin"])
        summary[f"{prefix}_flows"] = exposure["flow_count"]
        summary[f"{prefix}_mean_entropy_bits"] = exposure["mean_entropy_bits"]
        summary[f"{prefix}_mean_model_r_stat"] = exposure["mean_model_r_stat"]
    return summary


EXPOSURE_FIELDS = [
    "dataset",
    "e_bin",
    "e_bin_display",
    "lower_bound",
    "lower_inclusive",
    "upper_bound",
    "upper_inclusive",
    "flow_count",
    "flow_fraction_of_confirmed",
    "mean_e_i",
    "mean_entropy_bits",
    "mean_normalized_entropy",
    "mean_model_entropy_bits",
    "mean_model_r_stat",
]


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _metadata(
    dataset_dir: Path,
    output_dir: Path,
    max_output: int,
    max_header_bytes: int,
    compression_level: int,
    workers: int,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    confirmed_protocol = [
        "HTTP/1.x Content-Encoding gzip",
        "HTTP/1.x Content-Encoding deflate",
        "HTTP/1.x compressed Transfer-Encoding gzip/deflate",
    ]
    confirmed_intrinsic = [
        "gzip deflate region",
        "ZIP deflate/bzip2/LZMA entry data",
        "PNG IDAT data",
        "JPEG entropy-coded scans",
        "WebP VP8/VP8L bitstreams",
    ]
    if compression_level >= 2:
        confirmed_intrinsic.extend([
            "fully decoded standalone no-dictionary zlib stream",
            "fully decoded standalone BZIP2 stream",
            "fully decoded standalone XZ stream",
            "GIF LZW image-data sub-blocks",
            "WOFF1 zlib-compressed tables/metadata",
            "SWF CWS zlib-compressed body",
        ])
    if compression_level >= 3:
        confirmed_protocol.extend([
            "exactly framed HTTP/1.x br declaration",
            "exactly framed HTTP/1.x zstd declaration",
            "exactly framed HTTP/1.x compress/x-compress declaration",
        ])
        confirmed_intrinsic.extend([
            "recognized non-stored/non-encrypted ZIP method entry data",
            "structurally complete Zstandard compressed/RLE blocks",
            "structurally complete LZ4 compressed blocks",
        ])

    explicitly_unsupported = [
        "HTTP/2",
        "HTTP/3",
        "WebSocket compression",
        "ZIP64",
        "encrypted ZIP entries",
        "unrecognized/reserved ZIP methods",
        "HTTP dcb/dcz dictionary content codings",
        "WOFF2, complex PDF filters, and audio/video containers",
        "incomplete objects or signature-only candidates",
        "TLS/QUIC ciphertext as compression evidence",
    ]
    if compression_level < 3:
        explicitly_unsupported.extend([
            "HTTP br",
            "HTTP zstd",
            "HTTP compress/x-compress",
            "ZIP methods outside the runtime decoder scope",
            "Zstandard frames without decoding",
            "LZ4 frames without decoding",
        ])

    return {
        "script": "revision/compression_audit.py",
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_dir.resolve()),
        "output_path": str(output_dir.resolve()),
        "sample_definition": "one DATASET/label/*.pcap file is one bidirectional flow",
        "dataset_selection": (
            "all direct label-directory PCAP/PCAPNG files; no class filtering, "
            "class cap, random split, or resampling"
        ),
        "parallelism": {
            "model": "processes" if workers > 1 else "serial",
            "workers": workers,
            "maximum_in_flight_tasks": (
                min(summary["total_pcaps"], workers * PENDING_TASKS_PER_WORKER)
                if workers > 1
                else 1
            ),
            "output_order": "deterministic dataset path order",
        },
        "window": {
            "payload_bearing_packets": MAX_RAW_PACKETS,
            "bytes_per_packet": BYTES_PER_PACKET,
            "raw_sequence_length": RAW_SEQUENCE_LENGTH,
            "special_tokens": 2,
            "maximum_payload_bytes": MAX_WINDOW_BYTES,
            "definition": (
                "truncate_510(P1[:64] || ... || P8[:64]) in capture order"
            ),
        },
        "compression_scope": {
            "selected_level": compression_level,
            "levels_are_cumulative": True,
            "level_definitions": {
                "1": "original detector scope, unchanged",
                "2": "level 1 plus additional fully decoded/validated formats",
                "3": (
                    "levels 1-2 plus explicit compression declarations or "
                    "complete framed structures with exact compressed regions"
                ),
            },
            "confirmed_protocol": confirmed_protocol,
            "confirmed_intrinsic": confirmed_intrinsic,
            "explicitly_unsupported": explicitly_unsupported,
            "interpretation": (
                "confirmed_compressed / valid_flows is a conservative lower bound "
                "on whole-flow wire-visible compression, not true compression "
                "prevalence"
            ),
            "region_semantics": (
                "HTTP protocol regions are content-coded entity octets with HTTP "
                "chunk framing excluded; intrinsic regions are the format's actual "
                "compressed data sections rather than the whole file container"
            ),
        },
        "validation": {
            "maximum_decoded_bytes_per_object_or_layer": max_output,
            "extended_detector_probe_chunk_bytes": DECODE_PROBE_CHUNK_BYTES,
            "extended_detector_input_chunk_bytes": DECODE_INPUT_CHUNK_BYTES,
            "xz_decoder_memory_limit": max(1024 * 1024, max_output),
            "maximum_http_header_bytes": max_header_bytes,
            "tcp_overlap_policy": "first-seen bytes win",
            "tcp_gaps": "analyzed as separate contiguous buffers",
            "plain_http_chunking": (
                "identity-coded intrinsic objects are de-chunked for validation and "
                "mapped back to the original non-contiguous TCP byte intervals"
            ),
            "retransmission_conflict_mapping": (
                "conflicting model-window bytes are retained in entropy but cannot "
                "count as confirmed compressed-window hits"
            ),
        },
        "metrics": {
            "e_i": (
                "union of model-window byte positions mapped to confirmed protocol "
                "or intrinsic compressed regions, divided by |W_i|"
            ),
            "entropy_bits": (
                "per-flow Shannon entropy of payload bytes in W_i using log2; bin "
                "values are arithmetic means over flows, never pooled-byte entropy"
            ),
            "model_r_stat": (
                "1 - H([CLS]+W_i+[SEP]) / log2(min(|W_i|+2, 261)); "
                "PAD excluded, matching compute_flow_reliability_raw"
            ),
            "empty_bin": "mean fields are empty/JSON null, never zero",
        },
        "summary": summary,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure selected-level wire-visible compression evidence and its "
            "exposure inside MM-TrafficBERT's raw-content window."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="dataset root with DATASET/label/*.pcap",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "output directory (default: sibling <dataset>_compression_audit)"
        ),
    )
    parser.add_argument(
        "--max-decompressed-bytes",
        type=int,
        default=DEFAULT_MAX_DECOMPRESSED_BYTES,
        help=(
            "per-object/per-layer decoded-byte safety cap "
            f"(default: {DEFAULT_MAX_DECOMPRESSED_BYTES})"
        ),
    )
    parser.add_argument(
        "--max-http-header-bytes",
        type=int,
        default=DEFAULT_MAX_HTTP_HEADER_BYTES,
        help=(
            "maximum HTTP header block inspected "
            f"(default: {DEFAULT_MAX_HTTP_HEADER_BYTES})"
        ),
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=COMPRESSION_LEVELS,
        default=DEFAULT_COMPRESSION_LEVEL,
        help=(
            "cumulative compression judgment level: 1 preserves the original "
            "strict scope; 2 adds fully validated formats; 3 adds explicit, "
            "exactly bounded declarations/frames "
            f"(default: {DEFAULT_COMPRESSION_LEVEL})"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "number of PCAP analysis worker processes; use 1 for serial "
            f"execution (default: {DEFAULT_WORKERS})"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress messages (final summary is still printed)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset directory not found: {dataset_dir}")
    if args.max_decompressed_bytes <= 0:
        raise SystemExit("--max-decompressed-bytes must be positive")
    if args.max_http_header_bytes <= 0:
        raise SystemExit("--max-http-header-bytes must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_dir.parent / f"{dataset_dir.name}_compression_audit"
    )
    pcaps = discover_pcaps(dataset_dir)
    if not pcaps:
        raise SystemExit(
            f"no DATASET/label/*.pcap or *.pcapng files found under {dataset_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(pcaps)
    workers = min(args.workers, total)
    if not args.quiet:
        print(
            f"[compression-audit] pcaps={total}, workers={workers}, "
            f"compression_level={args.compression_level}",
            flush=True,
        )
    rows = analyze_pcaps(
        dataset_dir.name,
        dataset_dir,
        pcaps,
        args.max_decompressed_bytes,
        args.max_http_header_bytes,
        args.compression_level,
        workers,
        args.quiet,
    )

    valid_rows = [row for row in rows if row["status"] == "valid"]
    exposure_rows = build_exposure_rows(dataset_dir.name, valid_rows)
    summary = build_summary_row(
        dataset_dir.name,
        dataset_dir,
        rows,
        exposure_rows,
    )

    summary_fields = list(summary.keys())
    _atomic_write_csv(output_dir / "dataset_summary.csv", [summary], summary_fields)
    _atomic_write_csv(output_dir / "exposure_bins.csv", exposure_rows, EXPOSURE_FIELDS)
    _atomic_write_csv(output_dir / "flow_details.csv", rows, FLOW_FIELDS)
    _atomic_write_json(
        output_dir / "audit_metadata.json",
        _metadata(
            dataset_dir,
            output_dir,
            args.max_decompressed_bytes,
            args.max_http_header_bytes,
            args.compression_level,
            workers,
            summary,
        ),
    )

    print(
        "[compression-audit] "
        f"level={args.compression_level}, "
        f"valid={summary['valid_flows']}/{summary['total_pcaps']}, "
        f"confirmed={summary['confirmed_compressed']}, "
        f"rate={summary['compression_rate']}, "
        f"in_window={summary['compressed_in_window']}"
    )
    print(f"[compression-audit] outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

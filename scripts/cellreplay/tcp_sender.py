#!/usr/bin/env python3
import argparse
import csv
import json
import socket
import struct
import time
from pathlib import Path


def recv_exact(connection, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed before acknowledgement")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def tcp_total_retransmissions(connection):
    if not hasattr(socket, "TCP_INFO"):
        return None
    info = connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 192)
    if len(info) < 104:
        return None
    return struct.unpack_from("I", info, 100)[0]


def connect_with_retry(host, port, timeout_s=10):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return socket.create_connection((host, port), timeout=30)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=50127)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--loss-rate", type=float, required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    with connect_with_retry(args.host, args.port) as connection, open(
        output, "a", encoding="utf-8"
    ) as results:
        connection.settimeout(120)
        for row in rows:
            payload_path = Path(row["payload_path"])
            payload = payload_path.read_bytes()
            header = {
                "image_id": row.get("image_id"),
                "question_id": row.get("question_id"),
                "fidelity": row.get("selected_fidelity", row.get("fidelity")),
                "payload_bytes": len(payload),
            }
            encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
            retrans_before = tcp_total_retransmissions(connection)
            started_ns = time.perf_counter_ns()
            connection.sendall(struct.pack("!I", len(encoded_header)))
            connection.sendall(encoded_header)
            connection.sendall(payload)
            acknowledged = struct.unpack("!Q", recv_exact(connection, 8))[0]
            completion_s = (time.perf_counter_ns() - started_ns) / 1e9
            retrans_after = tcp_total_retransmissions(connection)
            result = {
                **header,
                "acknowledged_bytes": acknowledged,
                "completion_s": completion_s,
                "tcp_retransmissions": None
                if retrans_before is None or retrans_after is None
                else max(0, retrans_after - retrans_before),
                "trace": args.trace,
                "loss_rate": args.loss_rate,
                "transport": "reliable TCP",
            }
            results.write(json.dumps(result) + "\n")
            results.flush()
        connection.sendall(struct.pack("!I", 0))


if __name__ == "__main__":
    main()

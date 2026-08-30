#!/usr/bin/env python3
import argparse
import json
import socket
import struct


def recv_exact(connection, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(min(1024 * 1024, remaining))
        if not chunk:
            raise ConnectionError("connection closed before payload completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50127)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        connection, _ = server.accept()
        with connection:
            while True:
                raw_header_size = recv_exact(connection, 4)
                header_size = struct.unpack("!I", raw_header_size)[0]
                if header_size == 0:
                    break
                header = json.loads(recv_exact(connection, header_size))
                payload_size = int(header["payload_bytes"])
                recv_exact(connection, payload_size)
                connection.sendall(struct.pack("!Q", payload_size))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import socket
from typing import Dict, Iterable, Iterator, Optional


def send_jsonl(sock: socket.socket, payload: Dict[str, object]) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)


def iter_jsonl_messages(sock: socket.socket) -> Iterator[Dict[str, object]]:
    buf = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            return
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(buf[:nl]).decode("utf-8", errors="replace").strip()
            del buf[: nl + 1]
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def resolve_host(host: str) -> str:
    try:
        return socket.gethostbyname(host)
    except Exception:
        return host


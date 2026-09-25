"""Lossless removal of JPEG metadata (EXIF, GPS, XMP, maker notes, comments).

Only the container is rewritten: the compressed scan data is copied byte for
byte, so the decoded pixels are identical. The ICC colour profile is kept
(it identifies no one, and without it a Display-P3 photo shows shifted
colours). Anything after the End Of Image
marker is dropped as well: phones append a second JPEG there (an MPF gain map
or preview) that carries its own XMP.
"""

from __future__ import annotations

from pathlib import Path

_SOS, _EOI = 0xDA, 0xD9
_STANDALONE = {0x01, *range(0xD0, 0xD8)}  # TEM, RST0-7: no length field


def _skip_scan(data: bytes, i: int) -> int:
    """Index of the first marker after the entropy-coded data starting at i."""
    n = len(data)
    while i < n - 1:
        if data[i] == 0xFF:
            nxt = data[i + 1]
            if nxt == 0x00 or 0xD0 <= nxt <= 0xD7:  # byte stuffing, restart markers
                i += 2
                continue
            if nxt == 0xFF:  # fill byte
                i += 1
                continue
            return i
        i += 1
    raise ValueError("corrupt JPEG: no End Of Image marker")


def strip_metadata(data: bytes) -> bytes:
    """Drop every APPn (except APP0/JFIF and the ICC profile) and COM segment, and trailing data."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file")
    out = bytearray(b"\xff\xd8")
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError(f"corrupt JPEG: expected a marker at byte {i}")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker == _EOI:
            out += b"\xff\xd9"
            return bytes(out)  # whatever follows (MPF secondary image...) is dropped
        if marker in _STANDALONE:
            out += data[i : i + 2]
            i += 2
            continue
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        end = i + 2 + length
        is_app = 0xE0 <= marker <= 0xEF
        is_icc = marker == 0xE2 and data[i + 4 : i + 16] == b"ICC_PROFILE\x00"
        if not ((is_app and marker != 0xE0 and not is_icc) or marker == 0xFE):
            out += data[i:end]
        i = end
        if marker == _SOS:  # copy the entropy-coded data up to the next marker
            j = _skip_scan(data, i)
            out += data[i:j]
            i = j
    raise ValueError("corrupt JPEG: no End Of Image marker")


def copy_without_metadata(src: str | Path, dst: str | Path) -> None:
    Path(dst).write_bytes(strip_metadata(Path(src).read_bytes()))

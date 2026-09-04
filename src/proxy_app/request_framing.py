from __future__ import annotations


class JSONBoundaryError(ValueError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.public_message = message


def parse_content_length(headers: list[tuple[bytes, bytes]], max_bytes: int) -> int | None:
    lengths = [
        value.decode("ascii", "strict") for name, value in headers if name == b"content-length"
    ]
    encodings = [
        value.decode("ascii", "strict") for name, value in headers if name == b"transfer-encoding"
    ]
    if len(lengths) > 1 or len(encodings) > 1 or (lengths and encodings):
        raise JSONBoundaryError(400, "ambiguous_framing", "Request framing is invalid.")
    if encodings and encodings[0].strip().lower() != "chunked":
        raise JSONBoundaryError(400, "unsupported_transfer_encoding", "Request framing is invalid.")
    if not lengths:
        return None

    value = lengths[0]
    if not value or not value.isascii() or not value.isdecimal():
        raise JSONBoundaryError(400, "invalid_content_length", "Request framing is invalid.")
    if len(value) > 1 and value.startswith("0"):
        raise JSONBoundaryError(400, "invalid_content_length", "Request framing is invalid.")
    declared = int(value)
    if declared > max_bytes:
        raise JSONBoundaryError(413, "body_too_large", "Request body exceeds the 4 MiB limit.")
    return declared

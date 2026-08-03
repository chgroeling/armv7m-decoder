"""Decoding one instruction out of a byte buffer.

:func:`decode` takes a word and the size to read it at; a caller walking a stream
has neither, only bytes and a position. :func:`decode_at` bridges the two, and
reports what it read alongside what it decoded.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

from armv7m_decoder._decoder import Context, InstructionSize, decode
from armv7m_decoder._size import instr_size


@dataclass(frozen=True)
class DecodedWord:
    """One instruction read out of a buffer, and what it decoded to.

    Attributes:
        offset: Where in the buffer the instruction begins.
        size: The size it was decoded at, settled from its first halfword.
        n_bytes: Bytes it occupies -- how far the stream advances past it.
        word: The instruction word, ``size`` bits wide, as the decoder took it.
        halfwords: The halfwords it was assembled from, in memory order.
        instruction: The decoded instruction, or ``NoMatch`` if none matched.
    """

    offset: int
    size: InstructionSize
    n_bytes: int
    word: int
    halfwords: tuple[int, ...]
    instruction: object


def decode_at(data: bytes, offset: int, ctx: Context) -> Optional[DecodedWord]:
    """Decode the instruction that begins at ``offset`` in ``data``.

    ITSTATE reaches the decoder through ``ctx``, which the decode only reads: it
    still holds what the word decoded under when this returns, so the caller can
    hand the same ``ctx.istate`` to ``disassemble`` and ``next_itstate``.

    Args:
        data: The bytes to decode from, little-endian as Thumb code is stored.
        offset: Where in ``data`` the instruction begins.
        ctx: The runtime context the decode blocks thread their state through.

    Returns:
        What the word decoded to, or ``None`` if ``data`` does not hold a whole
        instruction at ``offset``.
    """
    if offset < 0 or offset + 2 > len(data):
        return None

    # The first halfword settles the size, whether or not an encoding matches --
    # skipping only half of a 32-bit word would leave the second halfword to be
    # decoded as an instruction of its own.
    hw1 = struct.unpack_from("<H", data, offset)[0]
    size = instr_size(hw1)
    n_bytes = size // 8
    if offset + n_bytes > len(data):
        return None

    if size == InstructionSize.SIZE_32BIT:
        hw2 = struct.unpack_from("<H", data, offset + 2)[0]
        halfwords = (hw1, hw2)
        word = (hw1 << 16) | hw2
    else:
        halfwords = (hw1,)
        word = hw1

    return DecodedWord(
        offset=offset,
        size=size,
        n_bytes=n_bytes,
        word=word,
        halfwords=halfwords,
        instruction=decode(word, ctx, size),
    )

"""Reading one instruction out of memory.

:func:`decode` takes a word and the size to read it at; a caller walking a stream
has neither, only bytes and a position. Settling the size is what bridges the
two, and Thumb settles it from the first halfword alone (:func:`instr_size`), so
:func:`fetch_and_decode` can take a whole instruction out of a buffer and hand it
to the decoder.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

from armv7m_decoder._decoder import Context, InstructionSize, decode


def instr_size(hw1: int) -> InstructionSize:
    """Size of the instruction that begins with halfword ``hw1``.

    A first halfword whose bits 15:11 read ``0b11101``, ``0b11110`` or
    ``0b11111`` begins a 32-bit instruction, and every other halfword is a
    16-bit instruction in its own right (Armv7-M ARM A5.1). The rule holds
    whether or not an encoding matches, which is what keeps a stream in step: a
    word nothing decodes is still skipped whole, instead of leaving its second
    halfword to be read as an instruction of its own.
    """
    if hw1 & 0xF800 >= 0xE800:
        return InstructionSize.SIZE_32BIT
    return InstructionSize.SIZE_16BIT


def instr_bytes(hw1: int) -> int:
    """Bytes that instruction occupies -- how far a stream advances past it."""
    return instr_size(hw1) // 8


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


def fetch_and_decode(data: bytes, offset: int, ctx: Context) -> Optional[DecodedWord]:
    """Fetch and decode the instruction that begins at ``offset`` in ``data``.

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

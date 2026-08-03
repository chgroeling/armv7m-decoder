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
    """One instruction word, and what it decoded to.

    Attributes:
        offset: Where in the buffer the instruction begins -- 0 for a word that
            came from nowhere but the caller's hand.
        size: The size it was decoded at.
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


def _decoded(
    word: int, size: InstructionSize, offset: int, ctx: Context
) -> DecodedWord:
    """Decode ``word`` at ``size`` and report it with the halfwords it holds."""
    if size == InstructionSize.SIZE_32BIT:
        halfwords = (word >> 16, word & 0xFFFF)
    else:
        halfwords = (word,)

    return DecodedWord(
        offset=offset,
        size=size,
        n_bytes=size // 8,
        word=word,
        halfwords=halfwords,
        instruction=decode(word, ctx, size),
    )


def decode_word(instr: int, ctx: Context) -> DecodedWord:
    """Decode an instruction word the caller already has in hand.

    The word is written the way an architecture manual spells the encoding --
    ``0xBF00`` for a 16-bit instruction, ``0xF3AF8000`` for a 32-bit one -- and
    that settles the size on its own: nothing below ``0x10000`` is a 32-bit
    encoding, since Thumb reserves the halfwords from ``0xE800`` up for the
    first half of one.

    Use :func:`fetch_and_decode` for a word still in memory. Its size has to
    come from the first halfword, which is the only rule that works on bytes:
    a bare ``0xE800`` in a buffer is the beginning of a 32-bit instruction, not
    a 16-bit word.

    Args:
        instr: The instruction word.
        ctx: The runtime context the decode blocks thread their state through.

    Returns:
        What the word decoded to. ``offset`` is 0 -- the word came from no
        buffer -- and ``instruction`` is ``NoMatch`` if no encoding matched.
    """
    if instr > 0xFFFF:
        size = InstructionSize.SIZE_32BIT
    else:
        size = InstructionSize.SIZE_16BIT
    return _decoded(instr, size, 0, ctx)


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
        word = (hw1 << 16) | struct.unpack_from("<H", data, offset + 2)[0]
    else:
        word = hw1

    return _decoded(word, size, offset, ctx)

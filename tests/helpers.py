"""Shared helpers for the decoder tests.

Instruction words are written the way an architecture manual spells the
encoding -- ``0xBF00`` for a 16-bit instruction, ``0xF3AF8000`` for a 32-bit
one -- which is also the form the decoder takes them in. Nothing below 0x10000
is a 32-bit encoding (Thumb reserves the halfwords from 0xE800 up for the first
half of one), so the size a word is decoded at follows from the word itself.
"""

from __future__ import annotations

import struct

from armv7m_decoder import (
    Context,
    InstructionSize,
    decode,
    disassemble,
    instr_size,
    next_itstate,
)


def word_size(instr: int) -> InstructionSize:
    """Size of a test word written in that form."""
    if instr > 0xFFFF:
        return InstructionSize.SIZE_32BIT
    return InstructionSize.SIZE_16BIT


def decode_word(ctx: Context, instr: int) -> object:
    """Decode one instruction word at the size the word itself implies."""
    return decode(instr, ctx, word_size(instr))


def disasm(ctx: Context, instr: int, offset: int = 0, istate: int = 0) -> str:
    """Decode one instruction word and format it as assembler text."""
    size = word_size(instr)
    return disassemble(decode(instr, ctx, size), instr, size, offset, istate)


def disassemble_stream(ctx: Context, halfwords: list[int]) -> list[str]:
    """Disassemble a Thumb stream, carrying ITSTATE across it like the CLI."""
    data = b"".join(struct.pack("<H", hw) for hw in halfwords)
    asm: list[str] = []
    offset = 0
    while offset + 2 <= len(data):
        hw1 = struct.unpack_from("<H", data, offset)[0]
        size = instr_size(hw1)
        n_bytes = size // 8
        if offset + n_bytes > len(data):
            break
        if size == InstructionSize.SIZE_32BIT:
            instr = (hw1 << 16) | struct.unpack_from("<H", data, offset + 2)[0]
        else:
            instr = hw1
        istate = ctx.istate
        result = decode(instr, ctx, size)
        asm.append(disassemble(result, instr, size, offset, istate))
        ctx.istate = next_itstate(istate, result)
        offset += n_bytes
    return asm

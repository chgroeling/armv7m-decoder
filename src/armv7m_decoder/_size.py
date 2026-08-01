"""Instruction size, settled from the first halfword alone.

The decoder does not classify: it decodes a word of exactly the size it is
handed, so the caller has to settle the size first. Thumb makes that possible
without decoding anything -- a first halfword whose bits 15:11 read ``0b11101``,
``0b11110`` or ``0b11111`` begins a 32-bit instruction, and every other halfword
is a 16-bit instruction in its own right (Armv7-M ARM A5.1). The rule holds
whether or not an encoding matches, which is what keeps a stream in step: a word
nothing decodes is still skipped whole, instead of leaving its second halfword to
be read as an instruction of its own.
"""

from __future__ import annotations

from armv7m_decoder._decoder import InstructionSize


def instr_size(hw1: int) -> InstructionSize:
    """Size of the instruction that begins with halfword ``hw1``."""
    if hw1 & 0xF800 >= 0xE800:
        return InstructionSize.SIZE_32BIT
    return InstructionSize.SIZE_16BIT


def instr_bytes(hw1: int) -> int:
    """Bytes that instruction occupies -- how far a stream advances past it."""
    return instr_size(hw1) // 8

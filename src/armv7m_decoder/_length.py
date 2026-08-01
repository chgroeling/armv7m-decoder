"""Instruction length for words the decoder matches no encoding for.

``decode`` reports the length of the encoding it matched. When nothing matches
it has no encoding to report and falls back to two bytes -- but Thumb settles
the length before any decoding: a first halfword of ``0b11101``, ``0b11110`` or
``0b11111`` in bits 15:11 begins a 32-bit instruction whether or not anything
decodes it (Armv7-M ARM A5.1). A caller walking a stream has to skip all four
bytes of one, or it disassembles the second halfword as an instruction in its
own right and every line after it is suspect.
"""

from __future__ import annotations

from armv7m_decoder._decoder import NoMatch


def instr_bytes(instr: int) -> int:
    """Bytes the instruction beginning at the top halfword of ``instr`` occupies."""
    return 4 if (instr >> 16) & 0xF800 >= 0xE800 else 2


def decoded_bytes(instr: int, result: object, n_bytes: int) -> int:
    """``n_bytes`` from ``decode``, corrected where it had no encoding to go on."""
    if isinstance(result, NoMatch):
        return instr_bytes(instr)
    return n_bytes

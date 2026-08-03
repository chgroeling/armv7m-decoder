"""Tests for decoding straight out of a byte buffer.

``decode`` takes a word and the size to read it at; ``decode_at`` takes bytes and
a position, and settles the rest. What it has to get right is the size -- a word
is read whole or not at all -- and what it has to report is enough for a caller
to spell the line and step the stream without going back to the bytes.
"""

from __future__ import annotations

import struct

from armv7m_decoder import (
    IT,
    Context,
    InstructionSize,
    Opcode,
    decode_at,
    next_itstate,
)


def buffer(halfwords: list[int]) -> bytes:
    return b"".join(struct.pack("<H", hw) for hw in halfwords)


class TestWordFromBuffer:
    def test_narrow_word_is_one_halfword(self, ctx: Context) -> None:
        word = decode_at(buffer([0x0001]), 0, ctx)
        assert word is not None
        assert word.size == InstructionSize.SIZE_16BIT
        assert word.n_bytes == 2
        assert word.word == 0x0001
        assert word.halfwords == (0x0001,)
        assert word.instruction.opcode == Opcode.OP_MOV_REGISTER

    def test_wide_word_joins_both_halfwords(self, ctx: Context) -> None:
        word = decode_at(buffer([0xF20D, 0x154F]), 0, ctx)
        assert word is not None
        assert word.size == InstructionSize.SIZE_32BIT
        assert word.n_bytes == 4
        assert word.word == 0xF20D154F
        assert word.halfwords == (0xF20D, 0x154F)

    def test_offset_is_where_it_reads_and_what_it_reports(self, ctx: Context) -> None:
        word = decode_at(buffer([0x0001, 0xF20D, 0x154F]), 2, ctx)
        assert word is not None
        assert word.offset == 2
        assert word.word == 0xF20D154F

    def test_a_word_nothing_matches_is_still_read_whole(self, ctx: Context) -> None:
        # The size holds whether or not an encoding matches: a caller stepping
        # by ``n_bytes`` stays in step where nothing decodes.
        word = decode_at(buffer([0xF2E5, 0x3EFF]), 0, ctx)
        assert word is not None
        assert word.n_bytes == 4
        assert word.instruction.opcode == Opcode.OP_NO_MATCH


class TestTruncation:
    def test_nothing_left_reads_as_nothing(self, ctx: Context) -> None:
        assert decode_at(buffer([0x0001]), 2, ctx) is None

    def test_a_lone_byte_is_not_a_halfword(self, ctx: Context) -> None:
        assert decode_at(b"\x01", 0, ctx) is None

    def test_a_wide_word_missing_its_tail_is_not_read(self, ctx: Context) -> None:
        # 0xF20D announces four bytes and only two are there. Reading it as the
        # halfword it is not would put every line after it out of step.
        assert decode_at(buffer([0xF20D]), 0, ctx) is None


class TestContext:
    def test_the_decode_leaves_the_context_alone(self, ctx: Context) -> None:
        # Nothing is carried back on the result because nothing needs to be: the
        # caller still holds the ITSTATE the word decoded under, which is what
        # ``disassemble`` and ``next_itstate`` both want.
        ctx.istate = 0x08
        ctx.apsr.C = 1
        word = decode_at(buffer([0xBF08]), 0, ctx)  # it eq -- loads nothing here
        assert word is not None
        assert ctx.istate == 0x08
        assert ctx.apsr.C == 1

    def test_advancing_itstate_stays_the_callers(self, ctx: Context) -> None:
        data = buffer([0xBF08, 0x0001])  # it eq; moveq r1, r0

        it = decode_at(data, 0, ctx)
        assert it is not None
        assert isinstance(it.instruction, IT)

        # The block opens only once the caller advances ITSTATE itself.
        ctx.istate = next_itstate(ctx.istate, it.instruction)
        assert ctx.istate == 0x08

    def test_the_block_reaches_the_decoder(self, ctx: Context) -> None:
        # 0x4001 is ``ands`` outside a block and ``and`` inside one -- the S bit
        # of a 16-bit data-processing instruction comes from ITSTATE, which
        # reaches the decoder through ``ctx``.
        outside = decode_at(buffer([0x4001]), 0, ctx)
        assert outside is not None
        assert outside.instruction.setflags is True

        ctx.istate = 0x08
        inside = decode_at(buffer([0x4001]), 0, ctx)
        assert inside is not None
        assert inside.instruction.setflags is False

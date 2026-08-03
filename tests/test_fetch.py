"""Tests for the two ways in: a word in hand, or a word still in memory.

Both settle the size the caller used to have to settle itself, and they settle
it from different things -- ``decode_word`` from the value it is handed,
``fetch_and_decode`` from the first halfword, the only rule that works on bytes.
What either has to report is enough for a caller to spell the line and step the
stream without going back to the word.
"""

from __future__ import annotations

import struct

import pytest

from armv7m_decoder import (
    IT,
    Context,
    InstructionSize,
    Opcode,
    decode_word,
    fetch_and_decode,
    instr_bytes,
    instr_size,
    next_itstate,
)

from .helpers import disassemble_stream


def buffer(halfwords: list[int]) -> bytes:
    return b"".join(struct.pack("<H", hw) for hw in halfwords)


class TestInstrSize:
    @pytest.mark.parametrize(
        ("halfword", "expected"),
        [
            (0xB717, InstructionSize.SIZE_16BIT),  # undefined, but plainly 16-bit
            (0x38D1, InstructionSize.SIZE_16BIT),
            (0xE7FF, InstructionSize.SIZE_16BIT),  # last halfword that stands alone
            (0xE800, InstructionSize.SIZE_32BIT),  # first of the three prefixes
            (0xF2E5, InstructionSize.SIZE_32BIT),
            (0xFFFF, InstructionSize.SIZE_32BIT),
        ],
    )
    def test_size_from_the_halfword(self, halfword: int, expected: int) -> None:
        assert instr_size(halfword) == expected

    @pytest.mark.parametrize(
        ("halfword", "expected"),
        [(0x38D1, 2), (0xE7FF, 2), (0xE800, 4), (0xFFFF, 4)],
    )
    def test_bytes_follow_the_size(self, halfword: int, expected: int) -> None:
        assert instr_bytes(halfword) == expected


class TestStreamResync:
    def test_stream_skips_both_halfwords(self, ctx) -> None:
        # The size holds whether or not an encoding matches: without it the
        # second halfword disassembles as an instruction of its own and every
        # line after it is suspect.
        assert disassemble_stream(ctx, [0x0001, 0xF2E5, 0x3EFF, 0x068E]) == [
            "movs\tr1, r0",
            "<no_match>",
            "lsls\tr6, r1, #26",
        ]


class TestWordInHand:
    def test_narrow_word_is_a_bare_halfword(self, ctx: Context) -> None:
        word = decode_word(0xBF00, ctx)
        assert word.size == InstructionSize.SIZE_16BIT
        assert word.n_bytes == 2
        assert word.word == 0xBF00
        assert word.halfwords == (0xBF00,)
        assert word.instruction.opcode == Opcode.OP_NOP

    def test_wide_word_splits_into_its_halfwords(self, ctx: Context) -> None:
        word = decode_word(0xF3AF8000, ctx)
        assert word.size == InstructionSize.SIZE_32BIT
        assert word.n_bytes == 4
        assert word.halfwords == (0xF3AF, 0x8000)
        assert word.instruction.opcode == Opcode.OP_NOP

    def test_it_came_from_no_buffer(self, ctx: Context) -> None:
        assert decode_word(0xBF00, ctx).offset == 0

    def test_the_block_reaches_the_decoder_here_too(self, ctx: Context) -> None:
        assert decode_word(0x4001, ctx).instruction.setflags is True
        ctx.istate = 0x08
        assert decode_word(0x4001, ctx).instruction.setflags is False

    def test_a_lone_first_halfword_is_the_word_it_is(self, ctx: Context) -> None:
        # 0xE800 out of a buffer begins a 32-bit instruction, but as a word in
        # hand it is all there is, so it decodes at the width it has -- and
        # matches nothing, 16-bit encodings stopping below 0xE800.
        word = decode_word(0xE800, ctx)
        assert word.size == InstructionSize.SIZE_16BIT
        assert word.instruction.opcode == Opcode.OP_NO_MATCH


class TestWordFromBuffer:
    def test_narrow_word_is_one_halfword(self, ctx: Context) -> None:
        word = fetch_and_decode(buffer([0x0001]), 0, ctx)
        assert word is not None
        assert word.size == InstructionSize.SIZE_16BIT
        assert word.n_bytes == 2
        assert word.word == 0x0001
        assert word.halfwords == (0x0001,)
        assert word.instruction.opcode == Opcode.OP_MOV_REGISTER

    def test_wide_word_joins_both_halfwords(self, ctx: Context) -> None:
        word = fetch_and_decode(buffer([0xF20D, 0x154F]), 0, ctx)
        assert word is not None
        assert word.size == InstructionSize.SIZE_32BIT
        assert word.n_bytes == 4
        assert word.word == 0xF20D154F
        assert word.halfwords == (0xF20D, 0x154F)

    def test_offset_is_where_it_reads_and_what_it_reports(self, ctx: Context) -> None:
        word = fetch_and_decode(buffer([0x0001, 0xF20D, 0x154F]), 2, ctx)
        assert word is not None
        assert word.offset == 2
        assert word.word == 0xF20D154F

    def test_a_word_nothing_matches_is_still_read_whole(self, ctx: Context) -> None:
        # The size holds whether or not an encoding matches: a caller stepping
        # by ``n_bytes`` stays in step where nothing decodes.
        word = fetch_and_decode(buffer([0xF2E5, 0x3EFF]), 0, ctx)
        assert word is not None
        assert word.n_bytes == 4
        assert word.instruction.opcode == Opcode.OP_NO_MATCH


class TestTruncation:
    def test_nothing_left_reads_as_nothing(self, ctx: Context) -> None:
        assert fetch_and_decode(buffer([0x0001]), 2, ctx) is None

    def test_a_lone_byte_is_not_a_halfword(self, ctx: Context) -> None:
        assert fetch_and_decode(b"\x01", 0, ctx) is None

    def test_a_wide_word_missing_its_tail_is_not_read(self, ctx: Context) -> None:
        # 0xF20D announces four bytes and only two are there. Reading it as the
        # halfword it is not would put every line after it out of step.
        assert fetch_and_decode(buffer([0xF20D]), 0, ctx) is None


class TestContext:
    def test_the_decode_leaves_the_context_alone(self, ctx: Context) -> None:
        # Nothing is carried back on the result because nothing needs to be: the
        # caller still holds the ITSTATE the word decoded under, which is what
        # ``disassemble`` and ``next_itstate`` both want.
        ctx.istate = 0x08
        ctx.apsr.C = 1
        word = fetch_and_decode(buffer([0xBF08]), 0, ctx)  # it eq -- loads nothing
        assert word is not None
        assert ctx.istate == 0x08
        assert ctx.apsr.C == 1

    def test_advancing_itstate_stays_the_callers(self, ctx: Context) -> None:
        data = buffer([0xBF08, 0x0001])  # it eq; moveq r1, r0

        it = fetch_and_decode(data, 0, ctx)
        assert it is not None
        assert isinstance(it.instruction, IT)

        # The block opens only once the caller advances ITSTATE itself.
        ctx.istate = next_itstate(ctx.istate, it.instruction)
        assert ctx.istate == 0x08

    def test_the_block_reaches_the_decoder(self, ctx: Context) -> None:
        # 0x4001 is ``ands`` outside a block and ``and`` inside one -- the S bit
        # of a 16-bit data-processing instruction comes from ITSTATE, which
        # reaches the decoder through ``ctx``.
        outside = fetch_and_decode(buffer([0x4001]), 0, ctx)
        assert outside is not None
        assert outside.instruction.setflags is True

        ctx.istate = 0x08
        inside = fetch_and_decode(buffer([0x4001]), 0, ctx)
        assert inside is not None
        assert inside.instruction.setflags is False

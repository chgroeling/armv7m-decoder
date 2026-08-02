"""Instruction-size tests: how wide a word is, settled before it is decoded."""

import pytest

from armv7m_decoder import InstructionSize, instr_bytes, instr_size

from .helpers import disassemble_stream


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
            "\t\t@ <UNDEFINED> instruction: 0xf2e53eff",
            "lsls\tr6, r1, #26",
        ]

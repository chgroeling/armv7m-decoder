"""Instruction-length tests: how far a caller advances past a decoded word."""

import pytest

from armv7m_decoder import decode, decoded_bytes, instr_bytes

from .test_disasm import disassemble_stream


class TestInstrBytes:
    @pytest.mark.parametrize(
        ("halfword", "expected"),
        [
            (0xB717, 2),  # undefined, but plainly 16-bit
            (0x38D1, 2),
            (0xE7FF, 2),  # last halfword that stands on its own
            (0xE800, 4),  # first of the three 32-bit prefixes
            (0xF2E5, 4),
            (0xFFFF, 4),
        ],
    )
    def test_length_from_the_halfword(self, halfword: int, expected: int) -> None:
        assert instr_bytes(halfword << 16) == expected


class TestDecodedBytes:
    def test_matched_encodings_keep_their_length(self, ctx) -> None:
        for instr in (0x38D1 << 16, 0xF2400000):
            result, n_bytes = decode(instr, ctx)
            assert decoded_bytes(instr, result, n_bytes) == n_bytes

    def test_unmatched_wide_word_takes_four_bytes(self, ctx) -> None:
        # The decoder has no encoding to report a length from and falls back
        # to two; Thumb settles it at four.
        result, n_bytes = decode(0xF2E53EFF, ctx)
        assert n_bytes == 2
        assert decoded_bytes(0xF2E53EFF, result, n_bytes) == 4

    def test_unmatched_narrow_word_takes_two_bytes(self, ctx) -> None:
        result, n_bytes = decode(0xB7170000, ctx)
        assert decoded_bytes(0xB7170000, result, n_bytes) == 2


class TestStreamResync:
    def test_stream_skips_both_halfwords(self, ctx) -> None:
        # Without the correction the second halfword disassembles as an
        # instruction of its own and every line after it is suspect.
        assert disassemble_stream(ctx, [0x0001, 0xF2E5, 0x3EFF, 0x068E]) == [
            "movs\tr1, r0",
            "<nomatch>",
            "lsls\tr6, r1, #26",
        ]

"""Decode correctness tests for the ARMv7-M instruction decoder.

An instruction word is decoded at the size the caller settles on beforehand,
and holds exactly that many bits: a 16-bit encoding is a bare halfword, a
32-bit one a full word.
"""

import pytest

from armv7m_decoder import (
    SIDEFFECT_NONE,
    SIDEFFECT_SEE,
    SIDEFFECT_UNDEFINED,
    SIDEFFECT_UNPREDICTABLE,
    Encoding,
    InstructionSize,
    NoMatch,
    decode_word,
    get_supported_sizes,
)
from armv7m_decoder._decoder import IT, decode


class TestDecoderBasics:
    def test_supported_sizes(self) -> None:
        assert get_supported_sizes() == (
            InstructionSize.SIZE_16BIT,
            InstructionSize.SIZE_32BIT,
        )

    def test_decode_empty_word_is_mov_register(self, ctx) -> None:
        """0x0000 matches MOV (register) T2 — not NoMatch."""
        from armv7m_decoder import MOV_register

        assert isinstance(decode_word(0x0000, ctx).instruction, MOV_register)

    def test_decode_nomatch(self, ctx) -> None:
        """0xFFFF0000 is not a valid ARMv7-M encoding."""
        assert isinstance(decode_word(0xFFFF0000, ctx).instruction, NoMatch)

    def test_a_size_the_encodings_do_not_use_matches_nothing(self, ctx) -> None:
        # 0xBF00 is NOP T1, but only as a 16-bit encoding. Reaching past the
        # public API on purpose: nothing there can ask for a size this set does
        # not use, which is the point of it not being public.
        assert isinstance(decode(0xBF00, ctx, InstructionSize.SIZE_8BIT), NoMatch)


class TestKnownEncodings:
    """Smoke tests for a representative sample of ARMv7-M instructions."""

    def test_nop_16bit(self, ctx) -> None:
        """NOP T1: 1011111100000000 = 0xBF00"""
        from armv7m_decoder import NOP

        assert isinstance(decode_word(0xBF00, ctx).instruction, NOP)

    def test_nop_32bit(self, ctx) -> None:
        """NOP T2: 111100111010xxxx10x0x00000000000 = 0xF3AF8000"""
        from armv7m_decoder import NOP

        assert isinstance(decode_word(0xF3AF8000, ctx).instruction, NOP)

    def test_mov_immediate_t1(self, ctx) -> None:
        """MOV (immediate) T1: 00100xxx... = 0x2000 (Rd=0, imm8=0)"""
        from armv7m_decoder import MOV_immediate

        assert isinstance(decode_word(0x2000, ctx).instruction, MOV_immediate)

    def test_adc_register_t1(self, ctx) -> None:
        """ADC (register) T1: 0100000101xxxxxx = 0x4140 (Rdn=0, Rm=0)"""
        from armv7m_decoder import ADC_register

        assert isinstance(decode_word(0x4140, ctx).instruction, ADC_register)

    def test_add_immediate_t1(self, ctx) -> None:
        """ADD (immediate) T1: 0001110xxxxxxxxx = 0x1C00 (Rd=0, Rn=0, imm3=0)"""
        from armv7m_decoder import ADD_immediate

        assert isinstance(decode_word(0x1C00, ctx).instruction, ADD_immediate)

    def test_sub_immediate_t1(self, ctx) -> None:
        """SUB (immediate) T1: 0001111xxxxxxxxx = 0x1E00"""
        from armv7m_decoder import SUB_immediate

        assert isinstance(decode_word(0x1E00, ctx).instruction, SUB_immediate)

    def test_push_t1(self, ctx) -> None:
        """PUSH T1: 1011010xxxxxxxxx, push {r0} = 0xB401"""
        from armv7m_decoder import PUSH

        assert isinstance(decode_word(0xB401, ctx).instruction, PUSH)

    def test_bkpt_t1(self, ctx) -> None:
        """BKPT T1: 10111110xxxxxxxx = 0xBE00 (imm8=0)"""
        from armv7m_decoder import BKPT

        assert isinstance(decode_word(0xBE00, ctx).instruction, BKPT)

    def test_ldr_immediate_t1(self, ctx) -> None:
        """LDR (immediate) T1: 01101xxxxxxxxxxx = 0x6800 (Rt=0, Rn=0, imm5=0)"""
        from armv7m_decoder import LDR_immediate

        assert isinstance(decode_word(0x6800, ctx).instruction, LDR_immediate)

    def test_str_immediate_t1(self, ctx) -> None:
        """STR (immediate) T1: 01100xxxxxxxxxxx = 0x6000 (Rt=0, Rn=0, imm5=0)"""
        from armv7m_decoder import STR_immediate

        assert isinstance(decode_word(0x6000, ctx).instruction, STR_immediate)

    def test_b_t2(self, ctx) -> None:
        """B T2: 11100xxxxxxxxxxx, unconditional branch forward 32 bytes = 0xE010"""
        from armv7m_decoder import B

        assert isinstance(decode_word(0xE010, ctx).instruction, B)


class TestEncodingMember:
    """Every instruction says which of its encodings matched.

    All encodings of an instruction share one class, so the fields alone cannot
    tell T3 from T4 -- `add.w r5, sp, #1` and `addw r5, sp, #335` differ in no
    member but this one. The disassembler spells several forms from it.
    """

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xF20D154F, Encoding.T4),  # addw r5, sp, #335
            (0xF10D0501, Encoding.T3),  # add.w r5, sp, #1
            (0xF2AD154F, Encoding.T3),  # subw r5, sp, #335
            (0xF1AD0501, Encoding.T2),  # sub.w r5, sp, #1
            (0xF240154F, Encoding.T3),  # movw r5, #335
            (0xF04F0501, Encoding.T2),  # mov.w r5, #1
            (0x3501, Encoding.T2),  # adds r5, #1  -- the narrow <Rdn> form
            (0x1CAD, Encoding.T1),  # adds r5, r5, #2 -- narrow, three operands
        ],
    )
    def test_encoding_is_reported(self, ctx, instr: int, expected: Encoding) -> None:
        assert decode_word(instr, ctx).instruction.encoding == expected

    def test_nomatch_carries_no_encoding(self) -> None:
        # No encoding matched, so there is no form to name.
        assert not hasattr(NoMatch(), "encoding")


class TestSideEffects:
    """A flagged side effect is reported on the instruction it belongs to.

    The decoder returns the instruction with every field decoded and the
    condition on `sideeffects`, so a caller can see both what the word decodes
    to and what the architecture says about it.
    """

    def test_undefined_is_reported_on_the_instruction(self, ctx) -> None:
        result = decode_word(0xF81DBAA1, ctx).instruction
        assert result.sideeffects & SIDEFFECT_UNDEFINED
        # ...and every field of the instruction is there to inspect.
        assert result.opcode >= 0
        assert hasattr(result, "encoding")

    def test_unpredictable_is_reported_on_the_instruction(self, ctx) -> None:
        # IT with firstcond 0b1111 is UNPREDICTABLE, but is still an IT.
        result = decode_word(0xBFF8, ctx).instruction
        assert result.sideeffects & SIDEFFECT_UNPREDICTABLE
        assert isinstance(result, IT)
        assert result.firstcond == 0xF

    def test_see_is_reported_on_the_instruction(self, ctx) -> None:
        # An IT whose mask is 0000 is the hint space, not an IT: SEE NOP.
        result = decode_word(0xBF50, ctx).instruction
        assert result.sideeffects & SIDEFFECT_SEE
        assert result.mask == 0

    def test_clean_decode_flags_nothing(self, ctx) -> None:
        assert decode_word(0xBF08, ctx).instruction.sideeffects == SIDEFFECT_NONE

    def test_nomatch_is_not_an_instruction(self, ctx) -> None:
        # NoMatch is the one result that is not an instruction, so it has
        # neither an encoding nor side effects to report.
        result = decode_word(0xF2E53EFF, ctx).instruction
        assert isinstance(result, NoMatch)
        assert not hasattr(result, "sideeffects")


class TestIDToName:
    """Verify the _opcode_to_name lookup table."""

    def test_id_to_name_has_instructions(self) -> None:
        from armv7m_decoder import _opcode_to_name

        assert len(_opcode_to_name) > 0
        nop_ids = [i for i, n in _opcode_to_name.items() if n == "NOP"]
        assert len(nop_ids) == 1

"""Decode correctness tests for the ARMv7-M instruction decoder.

An instruction word is decoded at the size the caller settles on beforehand,
and holds exactly that many bits: a 16-bit encoding is a bare halfword, a
32-bit one a full word.
"""

from armv7m_decoder import (
    InstructionSize,
    NoMatch,
    Undefined,
    Unpredictable,
    decode,
    get_supported_sizes,
)

from .helpers import decode_word


class TestDecoderBasics:
    def test_supported_sizes(self) -> None:
        assert get_supported_sizes() == (
            InstructionSize.SIZE_16BIT,
            InstructionSize.SIZE_32BIT,
        )

    def test_decode_empty_word_is_mov_register(self, ctx) -> None:
        """0x0000 matches MOV (register) T2 — not NoMatch."""
        from armv7m_decoder import MOV_register

        assert isinstance(decode_word(ctx, 0x0000), MOV_register)

    def test_decode_nomatch(self, ctx) -> None:
        """0xFFFF0000 is not a valid ARMv7-M encoding."""
        assert isinstance(decode_word(ctx, 0xFFFF0000), NoMatch)

    def test_a_size_the_encodings_do_not_use_matches_nothing(self, ctx) -> None:
        # 0xBF00 is NOP T1, but only as a 16-bit encoding.
        assert isinstance(decode(0xBF00, ctx, InstructionSize.SIZE_8BIT), NoMatch)


class TestKnownEncodings:
    """Smoke tests for a representative sample of ARMv7-M instructions."""

    def test_nop_16bit(self, ctx) -> None:
        """NOP T1: 1011111100000000 = 0xBF00"""
        from armv7m_decoder import NOP

        assert isinstance(decode_word(ctx, 0xBF00), NOP)

    def test_nop_32bit(self, ctx) -> None:
        """NOP T2: 111100111010xxxx10x0x00000000000 = 0xF3AF8000"""
        from armv7m_decoder import NOP

        assert isinstance(decode_word(ctx, 0xF3AF8000), NOP)

    def test_mov_immediate_t1(self, ctx) -> None:
        """MOV (immediate) T1: 00100xxx... = 0x2000 (Rd=0, imm8=0)"""
        from armv7m_decoder import MOV_immediate

        assert isinstance(decode_word(ctx, 0x2000), MOV_immediate)

    def test_adc_register_t1(self, ctx) -> None:
        """ADC (register) T1: 0100000101xxxxxx = 0x4140 (Rdn=0, Rm=0)"""
        from armv7m_decoder import ADC_register

        assert isinstance(decode_word(ctx, 0x4140), ADC_register)

    def test_add_immediate_t1(self, ctx) -> None:
        """ADD (immediate) T1: 0001110xxxxxxxxx = 0x1C00 (Rd=0, Rn=0, imm3=0)"""
        from armv7m_decoder import ADD_immediate

        assert isinstance(decode_word(ctx, 0x1C00), ADD_immediate)

    def test_sub_immediate_t1(self, ctx) -> None:
        """SUB (immediate) T1: 0001111xxxxxxxxx = 0x1E00"""
        from armv7m_decoder import SUB_immediate

        assert isinstance(decode_word(ctx, 0x1E00), SUB_immediate)

    def test_push_t1(self, ctx) -> None:
        """PUSH T1: 1011010xxxxxxxxx, push {r0} = 0xB401"""
        from armv7m_decoder import PUSH

        assert isinstance(decode_word(ctx, 0xB401), PUSH)

    def test_bkpt_t1(self, ctx) -> None:
        """BKPT T1: 10111110xxxxxxxx = 0xBE00 (imm8=0)"""
        from armv7m_decoder import BKPT

        assert isinstance(decode_word(ctx, 0xBE00), BKPT)

    def test_ldr_immediate_t1(self, ctx) -> None:
        """LDR (immediate) T1: 01101xxxxxxxxxxx = 0x6800 (Rt=0, Rn=0, imm5=0)"""
        from armv7m_decoder import LDR_immediate

        assert isinstance(decode_word(ctx, 0x6800), LDR_immediate)

    def test_str_immediate_t1(self, ctx) -> None:
        """STR (immediate) T1: 01100xxxxxxxxxxx = 0x6000 (Rt=0, Rn=0, imm5=0)"""
        from armv7m_decoder import STR_immediate

        assert isinstance(decode_word(ctx, 0x6000), STR_immediate)

    def test_b_t2(self, ctx) -> None:
        """B T2: 11100xxxxxxxxxxx, unconditional branch forward 32 bytes = 0xE010"""
        from armv7m_decoder import B

        assert isinstance(decode_word(ctx, 0xE010), B)


class TestPseudoInstructions:
    """Verify pseudo-instruction classes exist and decode may produce them."""

    def test_pseudo_instruction_classes_exist(self) -> None:
        assert issubclass(NoMatch, object)
        assert issubclass(Undefined, object)
        assert issubclass(Unpredictable, object)
        from armv7m_decoder._decoder import See

        assert issubclass(See, object)

    def test_decode_can_return_undefined(self, ctx) -> None:
        """Some operand combinations flag the UNDEFINED side effect."""
        # 0xF81D:xxx :xxx :xxx with specific fields triggers UNDEFINED.
        result = decode_word(ctx, 0xF81DBAA1)
        assert isinstance(result, Undefined), f"Got {type(result).__name__}"


class TestIDToName:
    """Verify the _opcode_to_name lookup table."""

    def test_id_to_name_has_instructions(self) -> None:
        from armv7m_decoder import _opcode_to_name

        assert len(_opcode_to_name) > 0
        nop_ids = [i for i, n in _opcode_to_name.items() if n == "NOP"]
        assert len(nop_ids) == 1

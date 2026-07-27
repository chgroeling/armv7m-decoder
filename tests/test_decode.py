"""Decode correctness tests for the ARMv7-M instruction decoder.

The decoder operates on 32-bit MSB-aligned instruction words.  16-bit
Thumb instructions are passed left-shifted by 16 bits (``val16 << 16``).
"""

from armv7m_decoder import (
    NoMatch,
    Undefined,
    Unpredictable,
    decode,
    get_decoder_eval_bytes,
    get_min_instr_bytes,
)


class TestDecoderBasics:
    def test_eval_bytes(self) -> None:
        assert get_decoder_eval_bytes() == 4

    def test_min_instr_bytes(self) -> None:
        assert get_min_instr_bytes() == 2

    def test_decode_empty_word_is_mov_register(self, ctx) -> None:
        """0x00000000 matches MOV (register) T2 — not NoMatch."""
        from armv7m_decoder import MOV_register

        result, _ = decode(0x0000, ctx)
        assert isinstance(result, MOV_register)

    def test_decode_nomatch(self, ctx) -> None:
        """0xFFFF0000 is not a valid ARMv7-M encoding."""
        result, n_bytes = decode(0xFFFF0000, ctx)
        assert isinstance(result, NoMatch)
        assert n_bytes == 2


class TestKnownEncodings:
    """Smoke tests for a representative sample of ARMv7-M instructions.

    All 16-bit values are left-shifted by 16 for the 32-bit MSB-aligned decoder.
    """

    def _decode_16(self, instr_16: int, ctx):
        return decode(instr_16 << 16, ctx)

    def test_nop_16bit(self, ctx) -> None:
        """NOP T1: 1011111100000000 = 0xBF00"""
        from armv7m_decoder import NOP

        result, n_bytes = self._decode_16(0xBF00, ctx)
        assert isinstance(result, NOP)
        assert n_bytes == 2

    def test_nop_32bit(self, ctx) -> None:
        """NOP T2: 111100111010xxxx10x0x00000000000 = 0xF3AF8000"""
        from armv7m_decoder import NOP

        result, n_bytes = decode(0xF3AF8000, ctx)
        assert isinstance(result, NOP)
        assert n_bytes == 4

    def test_mov_immediate_t1(self, ctx) -> None:
        """MOV (immediate) T1: 00100xxx... = 0x2000 (Rd=0, imm8=0)"""
        from armv7m_decoder import MOV_immediate

        result, n_bytes = self._decode_16(0x2000, ctx)
        assert isinstance(result, MOV_immediate)
        assert n_bytes == 2

    def test_adc_register_t1(self, ctx) -> None:
        """ADC (register) T1: 0100000101xxxxxx = 0x4140 (Rdn=0, Rm=0)"""
        from armv7m_decoder import ADC_register

        result, n_bytes = self._decode_16(0x4140, ctx)
        assert isinstance(result, ADC_register)
        assert n_bytes == 2

    def test_add_immediate_t1(self, ctx) -> None:
        """ADD (immediate) T1: 0001110xxxxxxxxx = 0x1C00 (Rd=0, Rn=0, imm3=0)"""
        from armv7m_decoder import ADD_immediate

        result, n_bytes = self._decode_16(0x1C00, ctx)
        assert isinstance(result, ADD_immediate)
        assert n_bytes == 2

    def test_sub_immediate_t1(self, ctx) -> None:
        """SUB (immediate) T1: 0001111xxxxxxxxx = 0x1E00"""
        from armv7m_decoder import SUB_immediate

        result, n_bytes = self._decode_16(0x1E00, ctx)
        assert isinstance(result, SUB_immediate)
        assert n_bytes == 2

    def test_push_t1(self, ctx) -> None:
        """PUSH T1: 1011010xxxxxxxxx, push {r0} = 0xB401"""
        from armv7m_decoder import PUSH

        result, n_bytes = self._decode_16(0xB401, ctx)
        assert isinstance(result, PUSH)
        assert n_bytes == 2

    def test_bkpt_t1(self, ctx) -> None:
        """BKPT T1: 10111110xxxxxxxx = 0xBE00 (imm8=0)"""
        from armv7m_decoder import BKPT

        result, n_bytes = self._decode_16(0xBE00, ctx)
        assert isinstance(result, BKPT)
        assert n_bytes == 2

    def test_ldr_immediate_t1(self, ctx) -> None:
        """LDR (immediate) T1: 01101xxxxxxxxxxx = 0x6800 (Rt=0, Rn=0, imm5=0)"""
        from armv7m_decoder import LDR_immediate

        result, n_bytes = self._decode_16(0x6800, ctx)
        assert isinstance(result, LDR_immediate)
        assert n_bytes == 2

    def test_str_immediate_t1(self, ctx) -> None:
        """STR (immediate) T1: 01100xxxxxxxxxxx = 0x6000 (Rt=0, Rn=0, imm5=0)"""
        from armv7m_decoder import STR_immediate

        result, n_bytes = self._decode_16(0x6000, ctx)
        assert isinstance(result, STR_immediate)
        assert n_bytes == 2

    def test_b_t2(self, ctx) -> None:
        """B T2: 11100xxxxxxxxxxx, unconditional branch forward 32 bytes = 0xE010"""
        from armv7m_decoder import B

        result, n_bytes = self._decode_16(0xE010, ctx)
        assert isinstance(result, B)
        assert n_bytes == 2


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
        result, n_bytes = decode(0xF81DBAA1, ctx)
        assert isinstance(result, Undefined), f"Got {type(result).__name__}"
        assert n_bytes == 4


class TestIDToName:
    """Verify the _id_to_name lookup table."""

    def test_id_to_name_has_instructions(self) -> None:
        from armv7m_decoder import _id_to_name

        assert len(_id_to_name) > 0
        nop_ids = [i for i, n in _id_to_name.items() if n == "NOP"]
        assert len(nop_ids) == 1

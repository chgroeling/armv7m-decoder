"""Disassembly formatting tests for the ARMv7-M instruction decoder."""

from armv7m_decoder import decode, disassemble


class TestDisasmBasics:
    def test_nop(self, ctx) -> None:
        result, _ = decode(0xBF00 << 16, ctx)
        assert disassemble(result) == "nop"

    def test_mov_immediate(self, ctx) -> None:
        result, _ = decode(0x2000 << 16, ctx)
        assert disassemble(result) == "movs\tr0, #0"

    def test_mov_register(self, ctx) -> None:
        result, _ = decode(0x0000 << 16, ctx)
        assert disassemble(result) == "movs\tr0, r0"

    def test_add_immediate(self, ctx) -> None:
        result, _ = decode(0x1C00 << 16, ctx)
        assert disassemble(result) == "adds\tr0, r0, #0"

    def test_sub_immediate(self, ctx) -> None:
        result, _ = decode(0x1E00 << 16, ctx)
        assert disassemble(result) == "subs\tr0, r0, #0"

    def test_adc_register(self, ctx) -> None:
        result, _ = decode(0x4140 << 16, ctx)
        assert disassemble(result) == "adcs\tr0, r0, r0"

    def test_bkpt(self, ctx) -> None:
        result, _ = decode(0xBE00 << 16, ctx)
        assert disassemble(result) == "bkpt\t0x0000"

    def test_push(self, ctx) -> None:
        result, _ = decode(0xB401 << 16, ctx)
        assert disassemble(result) == "push\t{r0}"

    def test_ldr_immediate(self, ctx) -> None:
        result, _ = decode(0x6800 << 16, ctx)
        assert disassemble(result) == "ldr\tr0, [r0, #0]"

    def test_str_immediate(self, ctx) -> None:
        result, _ = decode(0x6000 << 16, ctx)
        assert disassemble(result) == "str\tr0, [r0, #0]"

    def test_mul(self, ctx) -> None:
        result, _ = decode(0x4340 << 16, ctx)
        assert disassemble(result) == "muls\tr0, r0, r0"


class TestDisasmBranch:
    def test_b_t2(self, ctx) -> None:
        result, _ = decode(0xE010 << 16, ctx)
        assert disassemble(result) == "b.n\t0x24"

    def test_b_t2_backwards(self, ctx) -> None:
        result, _ = decode(0xE7FE << 16, ctx)
        assert "b" in disassemble(result)

    def test_blx_register(self, ctx) -> None:
        result, _ = decode(0x4780 << 16, ctx)
        assert disassemble(result) == "blx\tr0"

    def test_bx(self, ctx) -> None:
        result, _ = decode(0x4700 << 16, ctx)
        assert disassemble(result) == "bx\tr0"


class TestDisasmRegisterNames:
    def test_sp(self, ctx) -> None:
        result, _ = decode((0xF10D << 16) | 0x0D00, ctx)
        assert "sp" in disassemble(result)

    def test_pc_in_branch(self, ctx) -> None:
        result, _ = decode(0x4778 << 16, ctx)
        assert "pc" in disassemble(result)


class TestDisasmPseudoInstructions:
    def test_nomatch(self, ctx) -> None:
        result, _ = decode(0xFFFF0000, ctx)
        assert disassemble(result).startswith("<nomatch")

    def test_undefined(self, ctx) -> None:
        result, _ = decode(0xF81DBAA1, ctx)
        assert disassemble(result).startswith("<undefined")


class TestDisasmCoprocessor:
    pass


class TestDisasmDMB:
    def test_dmb(self, ctx) -> None:
        result, _ = decode(0xF3BF8F5F, ctx)
        assert "dmb" in disassemble(result)

    def test_dsb(self, ctx) -> None:
        result, _ = decode(0xF3BF8F4F, ctx)
        assert "dsb" in disassemble(result)

    def test_isb(self, ctx) -> None:
        result, _ = decode(0xF3BF8F6F, ctx)
        assert "isb" in disassemble(result)


class TestDisasmVFP:
    pass

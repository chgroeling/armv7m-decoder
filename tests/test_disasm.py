"""Disassembly formatting tests for the ARMv7-M instruction decoder."""

import pytest

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
        assert disassemble(result) == "adds\tr0, #0"

    def test_sub_immediate(self, ctx) -> None:
        result, _ = decode(0x1E00 << 16, ctx)
        assert disassemble(result) == "subs\tr0, #0"

    def test_adc_register(self, ctx) -> None:
        result, _ = decode(0x4140 << 16, ctx)
        assert disassemble(result) == "adcs\tr0, r0"

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


class TestDisasmAdr:
    """ADR prints as the PC-relative add/sub it encodes, matching objdump."""

    def test_narrow_resolves_target(self, ctx) -> None:
        result, _ = decode(0xA5D8 << 16, ctx)
        assert (
            disassemble(result, 0xA5D8 << 16, 0x60)
            == "add\tr5, pc, #864\t@ (adr r5, 0x3c4)"
        )

    def test_narrow_target_aligns_pc(self, ctx) -> None:
        # PC is offset + 4, then rounded down to a word boundary.
        instr = 0xA000 << 16
        result, _ = decode(instr, ctx)
        assert disassemble(result, instr, 0x62) == "add\tr0, pc, #0\t@ (adr r0, 0x64)"

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xF20F0508, "addw\tr5, pc, #8"),
            (0xF2AF0508, "subw\tr5, pc, #8"),
            (0xF60F70FF, "addw\tr0, pc, #4095\t@ 0xfff"),
            (0xF6AF70FF, "subw\tr0, pc, #4095\t@ 0xfff"),
        ],
    )
    def test_wide_is_addw_subw(self, ctx, instr: int, expected: str) -> None:
        result, _ = decode(instr, ctx)
        assert disassemble(result, instr, 0x64) == expected


class TestDisasmWidthSuffix:
    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # S-forms whose spelling ends in something that reads like a
            # condition code: movs/vs, bics/cs, adcs/cs, sbcs/cs, lsls/ls.
            (0xEA5F0000, "movs.w\tr0, r0"),
            (0xEA300000, "bics.w\tr0, r0"),
            (0xEB500000, "adcs.w\tr0, r0"),
            (0xEB700000, "sbcs.w\tr0, r0, r0"),
            (0xF04F0000, "mov.w\tr0, #0"),
            (0xF1100F00, "cmn.w\tr0, #0"),
        ],
    )
    def test_wide_gets_suffix(self, ctx, instr: int, expected: str) -> None:
        result, _ = decode(instr, ctx)
        assert disassemble(result) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # 32-bit only, so there is no narrow form to disambiguate from.
            (0xF3C00000, "ubfx\tr0, r0, #0, #1"),
            (0xEA900F00, "teq\tr0, r0"),
            (0xE9130003, "ldmdb\tr3, {r0, r1}"),
            (0xFB000000, "mla\tr0, r0, r0, r0"),
        ],
    )
    def test_wide_only_gets_no_suffix(self, ctx, instr: int, expected: str) -> None:
        result, _ = decode(instr, ctx)
        assert disassemble(result) == expected

    def test_narrow_gets_no_suffix(self, ctx) -> None:
        result, _ = decode(0x4140 << 16, ctx)
        assert disassemble(result) == "adcs\tr0, r0"


class TestDisasmStackAndMultiTransfer:
    """Only the 16-bit T1 forms are spelled push/pop, matching objdump."""

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xB401 << 16, "push\t{r0}"),
            (0xB501 << 16, "push\t{r0, lr}"),
            (0xBC01 << 16, "pop\t{r0}"),
            (0xBD01 << 16, "pop\t{r0, pc}"),
        ],
    )
    def test_narrow_push_pop(self, ctx, instr: int, expected: str) -> None:
        result, _ = decode(instr, ctx)
        assert disassemble(result) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xE8BD4091, "ldmia.w\tsp!, {r0, r4, r7, lr}"),
            (0xE8BD8091, "ldmia.w\tsp!, {r0, r4, r7, pc}"),
            (0xE92D4091, "stmdb\tsp!, {r0, r4, r7, lr}"),
            (0xF85D0B04, "ldr.w\tr0, [sp], #4"),
            (0xF85DFB04, "ldr.w\tpc, [sp], #4"),
            (0xF84D0D04, "str.w\tr0, [sp, #-4]!"),
        ],
    )
    def test_wide_push_pop(self, ctx, instr: int, expected: str) -> None:
        result, _ = decode(instr, ctx)
        assert disassemble(result) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xC901 << 16, "ldmia\tr1!, {r0}"),
            (0xC101 << 16, "stmia\tr1!, {r0}"),
            (0xE8B34091, "ldmia.w\tr3!, {r0, r4, r7, lr}"),
            (0xE8930003, "ldmia.w\tr3, {r0, r1}"),
            (0xE8A30003, "stmia.w\tr3!, {r0, r1}"),
            # ldmdb/stmdb have no narrow form, so they take no .w suffix.
            (0xE9130003, "ldmdb\tr3, {r0, r1}"),
            (0xE9230003, "stmdb\tr3!, {r0, r1}"),
        ],
    )
    def test_multi_transfer_width_suffix(self, ctx, instr: int, expected: str) -> None:
        result, _ = decode(instr, ctx)
        assert disassemble(result) == expected


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

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            (0x0, "dmb\t#0"),
            (0x1, "dmb\toshld"),
            (0x2, "dmb\toshst"),
            (0x3, "dmb\tosh"),
            (0x4, "dmb\t#4"),
            (0x5, "dmb\tnshld"),
            (0x6, "dmb\tunst"),
            (0x7, "dmb\tun"),
            (0x8, "dmb\t#8"),
            (0x9, "dmb\tishld"),
            (0xA, "dmb\tishst"),
            (0xB, "dmb\tish"),
            (0xC, "dmb\t#12"),
            (0xD, "dmb\tld"),
            (0xE, "dmb\tst"),
            (0xF, "dmb\tsy"),
        ],
    )
    def test_dmb_options(self, ctx, option: int, expected: str) -> None:
        result, _ = decode(0xF3BF8F50 | option, ctx)
        assert disassemble(result) == expected

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            (0x3, "dsb\tosh"),
            (0xB, "dsb\tish"),
            (0xC, "dfb"),
            (0xF, "dsb\tsy"),
        ],
    )
    def test_dsb_options(self, ctx, option: int, expected: str) -> None:
        result, _ = decode(0xF3BF8F40 | option, ctx)
        assert disassemble(result) == expected

    def test_isb_only_names_sy(self, ctx) -> None:
        # ISB leaves every option but SY numeric, matching objdump.
        result, _ = decode(0xF3BF8F6B, ctx)
        assert disassemble(result) == "isb\t#11"
        result, _ = decode(0xF3BF8F6F, ctx)
        assert disassemble(result) == "isb\tsy"

    def test_dbg(self, ctx) -> None:
        result, _ = decode(0xF3AF80F5, ctx)
        assert disassemble(result) == "dbg\t#5"


class TestDisasmVFP:
    pass

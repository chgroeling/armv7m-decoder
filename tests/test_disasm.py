"""Disassembly formatting tests for the ARMv7-M instruction decoder.

Instruction words are written as an architecture manual spells the encoding --
``0xBF00`` for a 16-bit one, ``0xF3AF8000`` for a 32-bit one -- and `disasm`
decodes each at the size that width implies.
"""

import pytest

from .helpers import disasm, disassemble_stream


class TestDisasmBasics:
    def test_nop(self, ctx) -> None:
        assert disasm(ctx, 0xBF00) == "nop"

    def test_mov_immediate(self, ctx) -> None:
        assert disasm(ctx, 0x2000) == "movs\tr0, #0"

    def test_mov_register(self, ctx) -> None:
        assert disasm(ctx, 0x0000) == "movs\tr0, r0"

    def test_add_immediate(self, ctx) -> None:
        assert disasm(ctx, 0x1C00) == "adds\tr0, r0, #0"

    def test_sub_immediate(self, ctx) -> None:
        assert disasm(ctx, 0x1E00) == "subs\tr0, r0, #0"

    def test_adc_register(self, ctx) -> None:
        assert disasm(ctx, 0x4140) == "adcs\tr0, r0"

    def test_bkpt(self, ctx) -> None:
        assert disasm(ctx, 0xBE00) == "bkpt\t0x0000"

    def test_push(self, ctx) -> None:
        assert disasm(ctx, 0xB401) == "push\t{r0}"

    def test_ldr_immediate(self, ctx) -> None:
        assert disasm(ctx, 0x6800) == "ldr\tr0, [r0, #0]"

    def test_str_immediate(self, ctx) -> None:
        assert disasm(ctx, 0x6000) == "str\tr0, [r0, #0]"

    def test_mul(self, ctx) -> None:
        assert disasm(ctx, 0x4340) == "muls\tr0, r0"


class TestDisasmBranch:
    def test_b_t2(self, ctx) -> None:
        assert disasm(ctx, 0xE010) == "b.n\t0x24"

    def test_b_t2_backwards(self, ctx) -> None:
        assert "b" in disasm(ctx, 0xE7FE)

    def test_blx_register(self, ctx) -> None:
        assert disasm(ctx, 0x4780) == "blx\tr0"

    def test_bx(self, ctx) -> None:
        assert disasm(ctx, 0x4700) == "bx\tr0"


class TestDisasmRegisterNames:
    def test_sp(self, ctx) -> None:
        assert "sp" in disasm(ctx, 0xF10D0D00)

    def test_pc_in_branch(self, ctx) -> None:
        assert "pc" in disasm(ctx, 0x4778)


class TestDisasmPseudoInstructions:
    def test_nomatch_comments_the_word(self, ctx) -> None:
        # No encoding matches, so there is no mnemonic to spell -- objdump
        # writes the word itself as a comment, and so do we.
        assert disasm(ctx, 0xF2E53EFF) == "@ <UNDEFINED> instruction: 0xf2e53eff"

    def test_narrow_nomatch_is_a_halfword_wide(self, ctx) -> None:
        assert disasm(ctx, 0xB717) == "@ <UNDEFINED> instruction: 0xb717"

    def test_undefined(self, ctx) -> None:
        assert disasm(ctx, 0xF81DBAA1).startswith("<undefined")


class TestDisasmAdr:
    """ADR prints as the PC-relative add/sub it encodes, matching objdump."""

    def test_narrow_resolves_target(self, ctx) -> None:
        assert disasm(ctx, 0xA5D8, 0x60) == "add\tr5, pc, #864\t@ (adr r5, 0x3c4)"

    def test_narrow_target_aligns_pc(self, ctx) -> None:
        # PC is offset + 4, then rounded down to a word boundary.
        assert disasm(ctx, 0xA000, 0x62) == "add\tr0, pc, #0\t@ (adr r0, 0x64)"

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
        assert disasm(ctx, instr, 0x64) == expected


class TestDisasmWidthSuffix:
    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # S-forms whose spelling ends in something that reads like a
            # condition code: movs/vs, bics/cs, adcs/cs, sbcs/cs, lsls/ls.
            (0xEA5F0000, "movs.w\tr0, r0"),
            (0xEA300000, "bics.w\tr0, r0, r0"),
            (0xEB500000, "adcs.w\tr0, r0, r0"),
            (0xEB700000, "sbcs.w\tr0, r0, r0"),
            (0xF04F0000, "mov.w\tr0, #0"),
            (0xF1100F00, "cmn.w\tr0, #0"),
        ],
    )
    def test_wide_gets_suffix(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected

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
        assert disasm(ctx, instr) == expected

    def test_narrow_gets_no_suffix(self, ctx) -> None:
        assert disasm(ctx, 0x4140) == "adcs\tr0, r0"


class TestDisasmImm12Form:
    """`addw` is a mnemonic of its own, not `add.w` with a different immediate.

    ADD, SUB and MOV each have two wide immediate encodings: one takes a
    modified immediate and is spelled with `.w`, the other a bare 12-bit value
    and is spelled `addw`/`subw`/`movw`. Both are 32 bits, so only bit 25 of
    the word tells them apart. Getting it wrong produces a line that will not
    assemble back: 335 has no modified-immediate form.
    """

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xF20D154F, "addw\tr5, sp, #335\t@ 0x14f"),
            (0xF20D2DBC, "addw\tsp, sp, #700\t@ 0x2bc"),
            (0xF2AD154F, "subw\tr5, sp, #335\t@ 0x14f"),
            (0xF2AD2DBC, "subw\tsp, sp, #700\t@ 0x2bc"),
            (0xF206154F, "addw\tr5, r6, #335\t@ 0x14f"),
            (0xF2A6154F, "subw\tr5, r6, #335\t@ 0x14f"),
            (0xF240154F, "movw\tr5, #335\t@ 0x14f"),
        ],
    )
    def test_imm12_form_is_spelled_w(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xF10D0501, "add.w\tr5, sp, #1"),
            (0xF50D5D80, "add.w\tsp, sp, #4096\t@ 0x1000"),
            (0xF1AD0501, "sub.w\tr5, sp, #1"),
            (0xF1060501, "add.w\tr5, r6, #1"),
            (0xF1A60501, "sub.w\tr5, r6, #1"),
            (0xF04F0501, "mov.w\tr5, #1"),
        ],
    )
    def test_modified_immediate_keeps_dot_w(
        self, ctx, instr: int, expected: str
    ) -> None:
        assert disasm(ctx, instr) == expected


class TestDisasmStackAndMultiTransfer:
    """Only the 16-bit T1 forms are spelled push/pop, matching objdump."""

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xB401, "push\t{r0}"),
            (0xB501, "push\t{r0, lr}"),
            (0xBC01, "pop\t{r0}"),
            (0xBD01, "pop\t{r0, pc}"),
        ],
    )
    def test_narrow_push_pop(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected

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
        assert disasm(ctx, instr) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xC901, "ldmia\tr1!, {r0}"),
            (0xC101, "stmia\tr1!, {r0}"),
            (0xE8B34091, "ldmia.w\tr3!, {r0, r4, r7, lr}"),
            (0xE8930003, "ldmia.w\tr3, {r0, r1}"),
            (0xE8A30003, "stmia.w\tr3!, {r0, r1}"),
            # ldmdb/stmdb have no narrow form, so they take no .w suffix.
            (0xE9130003, "ldmdb\tr3, {r0, r1}"),
            (0xE9230003, "stmdb\tr3!, {r0, r1}"),
        ],
    )
    def test_multi_transfer_width_suffix(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected


class TestDisasmDMB:
    def test_dmb(self, ctx) -> None:
        assert "dmb" in disasm(ctx, 0xF3BF8F5F)

    def test_dsb(self, ctx) -> None:
        assert "dsb" in disasm(ctx, 0xF3BF8F4F)

    def test_isb(self, ctx) -> None:
        assert "isb" in disasm(ctx, 0xF3BF8F6F)

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
        assert disasm(ctx, 0xF3BF8F50 | option) == expected

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
        assert disasm(ctx, 0xF3BF8F40 | option) == expected

    def test_isb_only_names_sy(self, ctx) -> None:
        # ISB leaves every option but SY numeric, matching objdump.
        assert disasm(ctx, 0xF3BF8F6B) == "isb\t#11"
        assert disasm(ctx, 0xF3BF8F6F) == "isb\tsy"

    def test_dbg(self, ctx) -> None:
        assert disasm(ctx, 0xF3AF80F5) == "dbg\t#5"


class TestDisasmOperandForm:
    """How many operands an instruction writes is a property of its encoding,
    not of whether two of its register fields happen to hold the same number."""

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # 16-bit T1: three operands, sharing a register or not.
            (0x18AD, "adds\tr5, r5, r2"),
            (0x18C5, "adds\tr5, r0, r3"),
            (0x1C40, "adds\tr0, r0, #1"),
            (0x1E00, "subs\tr0, r0, #0"),
            # 16-bit <Rdn> encodings: two, the destination standing in for the
            # first operand.
            (0x3001, "adds\tr0, #1"),
            (0x3801, "subs\tr0, #1"),
            (0x4148, "adcs\tr0, r1"),
            (0x4088, "lsls\tr0, r1"),
            (0x4408, "add\tr0, r1"),
            # MUL T1 spells its destination as <Rdm>, so it is the multiplicand
            # that goes unwritten, not the multiplier.
            (0x4341, "muls\tr1, r0"),
            # Wide encodings always write all three.
            (0xEA000000, "and.w\tr0, r0, r0"),
            (0xEB100000, "adds.w\tr0, r0, r0"),
            (0xF1000001, "add.w\tr0, r0, #1"),
            (0xFA00F001, "lsl.w\tr0, r0, r1"),
            (0xFB00F000, "mul.w\tr0, r0, r0"),
        ],
    )
    def test_operand_count(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # SP arithmetic leaves SP out of the operands only when narrow.
            (0x4468, "add\tr0, sp"),
            (0x4485, "add\tsp, r0"),
            (0xB001, "add\tsp, #4"),
            (0xB081, "sub\tsp, #4"),
            (0xEB0D0000, "add.w\tr0, sp, r0"),
            (0xEB0D0D00, "add.w\tsp, sp, r0"),
            (0xF10D0D01, "add.w\tsp, sp, #1"),
            (0xF1AD0D01, "sub.w\tsp, sp, #1"),
        ],
    )
    def test_sp_operand_form(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected


class TestDisasmCoprocessor:
    """The coprocessor instructions spell their operands their own way."""

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # The D bit is the long form, spelled after the 2 of the variant.
            (0xECF00102, "ldcl\t1, cr0, [r0], #8"),
            (0xECE00102, "stcl\t1, cr0, [r0], #8"),
            (0xECF08102, "ldcl\t1, cr8, [r0], #8"),
            (0xFCF50E02, "ldc2l\t14, cr0, [r5], #8"),
            (0xEDD50E02, "ldcl\t14, cr0, [r5, #8]"),
            (0xED850E02, "stc\t14, cr0, [r5, #8]"),
            # The unindexed form has no offset: its imm8 is an option code.
            (0xEC950E05, "ldc\t14, cr0, [r5], {5}"),
            # A post-indexed zero goes unwritten unless it is subtracted.
            (0xECB50E00, "ldc\t14, cr0, [r5]"),
            (0xEC350E00, "ldc\t14, cr0, [r5], #-0"),
        ],
    )
    def test_transfer(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xEE210E13, "mcr\t14, 1, r0, cr1, cr3, {0}"),
            (0xEE310E13, "mrc\t14, 1, r0, cr1, cr3, {0}"),
            (0xEE210E03, "cdp\t14, 2, cr0, cr1, cr3, {0}"),
            (0xEC450E12, "mcrr\t14, 1, r0, r5, cr2"),
            (0xEC550E12, "mrrc\t14, 1, r0, r5, cr2"),
        ],
    )
    def test_register_transfer(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected


class TestDisasmShiftImmediate:
    """A wide shift by an immediate is MOV (register) T3 with its shift filled
    in, and objdump spells it that way."""

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xEA4F0787, "mov.w\tr7, r7, lsl #2"),
            (0xEA4F0797, "mov.w\tr7, r7, lsr #2"),
            (0xEA4F07A7, "mov.w\tr7, r7, asr #2"),
            (0xEA4F07B7, "mov.w\tr7, r7, ror #2"),
            (0xEA4F0017, "mov.w\tr0, r7, lsr #32"),
            (0xEA5F0787, "movs.w\tr7, r7, lsl #2"),
            # RRX is a rotate by one through carry, 32-bit only.
            (0xEA4F0037, "mov.w\tr0, r7, rrx"),
            (0xEA5F0037, "movs.w\tr0, r7, rrx"),
            # No shift at all, and the narrow encodings, keep their own name.
            (0xEA4F0007, "mov.w\tr0, r7"),
            (0x0087, "lsls\tr7, r0, #2"),
            (0x0887, "lsrs\tr7, r0, #2"),
            (0x1087, "asrs\tr7, r0, #2"),
            # A shift by a register is a shift in its own right at any width.
            (0xFA07F006, "lsl.w\tr0, r7, r6"),
        ],
    )
    def test_shift_immediate(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected


class TestDisasmTableBranch:
    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # TBH indexes halfwords, so its index register is doubled.
            (0xE8DFF013, "tbh\t[pc, r3, lsl #1]"),
            (0xE8D1F012, "tbh\t[r1, r2, lsl #1]"),
            (0xE8DFF003, "tbb\t[pc, r3]"),
            (0xE8D1F002, "tbb\t[r1, r2]"),
        ],
    )
    def test_table_branch(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected


class TestDisasmNeg:
    """RSB (immediate) T1 is spelled as the negate it performs."""

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0x4252, "negs\tr2, r2"),
            (0x4241, "negs\tr1, r0"),
            # The wide forms keep the rsb spelling -- and take no .w, having no
            # narrow form of that name to be told apart from.
            (0xF1C10200, "rsb\tr2, r1, #0"),
            (0xF1D10200, "rsbs\tr2, r1, #0"),
            (0xEBC10200, "rsb\tr2, r1, r0"),
        ],
    )
    def test_neg(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected

    def test_neg_in_it_block(self, ctx) -> None:
        # T1 sets flags only outside an IT block, so the s gives way to the
        # block's condition.
        assert disassemble_stream(ctx, [0xBF08, 0x4252]) == ["it\teq", "negeq\tr2, r2"]


class TestDisasmZeroOffset:
    """A wide encoding drops a zero offset; a narrow one spells it out."""

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            (0xF8DC3000, "ldr.w\tr3, [ip]"),
            (0xF8500C00, "ldr.w\tr0, [r0]"),  # single-word: the sign goes too
            (0xF8510F00, "ldr.w\tr0, [r1]!"),
            # Post-indexed: the offset is what advances Rn, so it stays.
            (0xF8510B00, "ldr.w\tr0, [r1], #0"),
            # 16-bit forms always spell it.
            (0x6800, "ldr\tr0, [r0, #0]"),
            (0x9800, "ldr\tr0, [sp, #0]"),
        ],
    )
    def test_zero_offset(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # Coprocessor, VFP and dual transfers carry the U bit into the
            # syntax, so a subtracted zero stays visible as #-0.
            (0xED00E000, "stc\t0, cr14, [r0, #-0]"),
            (0xED80E000, "stc\t0, cr14, [r0]"),
            (0xED100A00, "vldr\ts0, [r0, #-0]"),
            (0xED900A00, "vldr\ts0, [r0]"),
            (0xED000A00, "vstr\ts0, [r0, #-0]"),
            (0xE9400000, "strd\tr0, r0, [r0, #-0]"),
            (0xE9C00000, "strd\tr0, r0, [r0]"),
        ],
    )
    def test_negative_zero_offset(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # The core forms gloss the plain positive offset -- the encodings
            # holding an imm12 -- and say nothing about an imm8, however
            # indexed and whichever way it goes.
            (0xF8841021, "strb.w\tr1, [r4, #33]\t@ 0x21"),
            (0x6B00, "ldr\tr0, [r0, #48]\t@ 0x30"),
            (0xF8041C58, "strb.w\tr1, [r4, #-88]"),
            (0xF8041F21, "strb.w\tr1, [r4, #33]!"),
            (0xF8041B21, "strb.w\tr1, [r4], #33"),
            (0xF8541E21, "ldrt\tr1, [r4, #33]"),
            # Dual transfers gloss the magnitude whichever way it goes.
            (0xE9440110, "strd\tr0, r1, [r4, #-64]\t@ 0x40"),
            # Coprocessor-class ones gloss a subtracted offset as the 32-bit
            # value it adds.
            (0xED04E010, "stc\t0, cr14, [r4, #-64]\t@ 0xffffffc0"),
            (0xED040A10, "vstr\ts0, [r4, #-64]\t@ 0xffffffc0"),
        ],
    )
    def test_offset_gloss(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr) == expected

    @pytest.mark.parametrize(
        ("instr", "expected"),
        [
            # A PC-relative load is glossed with the address it reads, in
            # brackets for the narrow form only.
            (0x4A06, "ldr\tr2, [pc, #24]\t@ (0x3c)"),
            (0xF85F1C21, "ldr.w\tr1, [pc, #-3105]\t@ 0xfffff403"),
            (0xED1F0A10, "vldr\ts0, [pc, #-64]\t@ 0xffffffe4"),
        ],
    )
    def test_literal_gloss(self, ctx, instr: int, expected: str) -> None:
        assert disasm(ctx, instr, 0x20) == expected

    def test_dual_literal_drops_zero(self, ctx) -> None:
        assert disasm(ctx, 0xE9DF0B00).startswith("ldrd\tr0, fp, [pc]")

    def test_wide_literal_drops_zero(self, ctx) -> None:
        assert disasm(ctx, 0xF8DF0000).startswith("ldr.w\tr0, [pc]")


class TestDisasmIT:
    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            (0xBF08, "it\teq"),
            (0xBFB8, "it\tlt"),
            (0xBFE8, "it\tal"),
            # A set mask bit repeats firstcond<0>, so the same bit pattern is
            # "then" for an odd firstcond and "else" for an even one.
            (0xBF0C, "ite\teq"),
            (0xBF1C, "itt\tne"),
            (0xBF04, "itt\teq"),
            (0xBF14, "ite\tne"),
            (0xBF42, "ittt\tmi"),
            (0xBF2B, "itete\tcs"),
            (0xBFC7, "ittee\tgt"),
            (0xBFDF, "itttt\tle"),
        ],
    )
    def test_it_mnemonic(self, ctx, word: int, expected: str) -> None:
        assert disasm(ctx, word) == expected

    def test_block_conditions_alternate(self, ctx) -> None:
        assert disassemble_stream(ctx, [0xBF0C, 0x2001, 0x2102]) == [
            "ite\teq",
            "moveq\tr0, #1",
            "movne\tr1, #2",
        ]

    def test_condition_precedes_width(self, ctx) -> None:
        # ADD (immediate) T3 and LDR (immediate) T3: cond before the .w.
        assert disassemble_stream(ctx, [0xBF1C, 0x2001, 0xF103, 0x0204]) == [
            "itt\tne",
            "movne\tr0, #1",
            "addne.w\tr2, r3, #4",
        ]
        assert disassemble_stream(ctx, [0xBF44, 0xEB11, 0x0002, 0xF8D1, 0x0004]) == [
            "itt\tmi",
            "addsmi.w\tr0, r1, r2",
            "ldrmi.w\tr0, [r1, #4]",
        ]

    def test_condition_precedes_data_type(self, ctx) -> None:
        assert disassemble_stream(ctx, [0xBF08, 0xEEB0, 0x0A60]) == [
            "it\teq",
            "vmoveq.f32\ts0, s1",
        ]

    def test_block_covers_four_slots_then_ends(self, ctx) -> None:
        assert disassemble_stream(ctx, [0xBF2B, 0xBF00, 0xBF00, 0xBF00, 0xBF00]) == [
            "itete\tcs",
            "nopcs",
            "nopcc",
            "nopcs",
            "nopcc",
        ]
        assert disassemble_stream(ctx, [0xBF08, 0xBF00, 0xBF00]) == [
            "it\teq",
            "nopeq",
            "nop",
        ]

    def test_al_block_spells_its_condition(self, ctx) -> None:
        # "al" is the one condition an encoding of its own leaves unspelled --
        # an IT block writes it out, as objdump does.
        assert disassemble_stream(ctx, [0xBFE8, 0xBF00, 0xBF00]) == [
            "it\tal",
            "nopal",
            "nop",
        ]

    def test_16bit_data_processing_drops_s_in_block(self, ctx) -> None:
        # ADD (register) T1 decodes with setflags = !InITBlock(), so the
        # running ITSTATE has to reach the decoder, not just the formatter.
        assert disassemble_stream(ctx, [0x1800]) == ["adds\tr0, r0, r0"]
        assert disassemble_stream(ctx, [0xBF2C, 0x1800, 0x1800]) == [
            "ite\tcs",
            "addcs\tr0, r0, r0",
            "addcc\tr0, r0, r0",
        ]

    def test_unconditional_branch_takes_the_block_condition(self, ctx) -> None:
        # B T2 encodes cond = AL, so the block's condition applies.
        assert disassemble_stream(ctx, [0xBF08, 0xE7FF]) == ["it\teq", "beq.n\t0x4"]

    def test_branch_keeps_its_own_condition(self, ctx) -> None:
        # B T1 carries the condition CurrentCond reports, so a block's
        # condition is never appended on top of it.
        assert disasm(ctx, 0xD0FE, istate=0x18) == "beq.n\t0x0"


class TestDisasmVFP:
    pass

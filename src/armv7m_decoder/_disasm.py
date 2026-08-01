"""ARMv7-M instruction disassembler.

Converts decoded instruction dataclass instances to UAL (Unified Assembler
Language) syntax strings, suitable for display as disassembled output.
"""

from __future__ import annotations

import inspect as _inspect
from typing import Any

from armv7m_decoder._decoder import DecoderState, Opcode
from armv7m_decoder._itstate import COND_AL, current_cond, in_it_block

# Separates the mnemonic from its operands. Formatters emit it directly, so a
# formatted instruction never has to be re-parsed to find the mnemonic boundary.
_SEP: str = "\t"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHIFT_NAMES = {1: "lsl", 2: "lsr", 3: "asr", 4: "ror", 5: "rrx"}
_COND_CODES = [
    "eq",
    "ne",
    "cs",
    "cc",
    "mi",
    "pl",
    "vs",
    "vc",
    "hi",
    "ls",
    "ge",
    "lt",
    "gt",
    "le",
    "al",
]
_MNEMONICS_WITH_BOTH_WIDTHS: frozenset[str] = frozenset({
    "adc",
    "add",
    "and",
    "asr",
    "b",
    "bic",
    "cmn",
    "cmp",
    "eor",
    "ldmia",
    "ldr",
    "ldrb",
    "ldrh",
    "ldrsb",
    "ldrsh",
    "lsl",
    "lsr",
    "mov",
    "mul",
    "mvn",
    "nop",
    "orr",
    "rev",
    "rev16",
    "revsh",
    "ror",
    "rsb",
    "sbc",
    "sev",
    "stmia",
    "str",
    "strb",
    "strh",
    "sub",
    "sxtb",
    "sxth",
    "tst",
    "udf",
    "uxtb",
    "uxth",
    "wfe",
    "wfi",
    "yield",
})
_BARRIER_OPTIONS: dict[int, str] = {
    0x1: "oshld",
    0x2: "oshst",
    0x3: "osh",
    0x5: "nshld",
    0x6: "unst",
    0x7: "un",
    0x9: "ishld",
    0xA: "ishst",
    0xB: "ish",
    0xD: "ld",
    0xE: "st",
    0xF: "sy",
}
_SPEC_REGS: dict[int, str] = {
    0x00: "APSR",
    0x01: "IAPSR",
    0x02: "EAPSR",
    0x03: "XPSR",
    0x05: "IPSR",
    0x06: "EPSR",
    0x07: "IEPSR",
    0x08: "MSP",
    0x09: "PSP",
    0x10: "PRIMASK",
    0x11: "BASEPRI",
    0x12: "BASEPRI_MAX",
    0x13: "FAULTMASK",
    0x14: "CONTROL",
}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _reg(r: int) -> str:
    if r == 10:
        return "sl"
    if r == 11:
        return "fp"
    if r == 12:
        return "ip"
    if r == 13:
        return "sp"
    if r == 14:
        return "lr"
    if r == 15:
        return "pc"
    return f"r{r}"


def _sreg(r: int) -> str:
    return f"s{r}"


def _dreg(r: int) -> str:
    return f"d{r}"


def _shift(shift_t: int, shift_n: int) -> str:
    if shift_t == 5:
        return ", rrx"
    if shift_t == 1 and shift_n == 0:
        return ""
    name = _SHIFT_NAMES.get(shift_t, f"shift{shift_t}")
    return f", {name} #{shift_n}"


def _cond(cond: int) -> str:
    # A cond field of 0b1110 in an encoding of its own marks the instruction
    # unconditional and goes unspelled; only an IT block writes out "al".
    if cond >= COND_AL:
        return ""
    return _COND_CODES[cond]


def _flags(setflags: bool) -> str:
    return "s" if setflags else ""


def _width(result: Any, mnemonic: str) -> str:
    """Width suffix for `mnemonic`, appended after any `s`/condition suffix.

    A 32-bit encoding takes `.w` only where the same mnemonic also has a
    16-bit encoding and the suffix is what tells the two apart. Mnemonics
    that exist in one width only (`ubfx`, `stmdb`, `teq`, ...) take nothing.
    """
    if mnemonic in _MNEMONICS_WITH_BOTH_WIDTHS and (
        result.decoder_state & DecoderState.DECODED_32BIT
    ):
        return ".w"
    return ""


def _is_narrow(result: Any) -> bool:
    """Whether `result` came from a 16-bit encoding."""
    return bool(result.decoder_state & DecoderState.DECODED_16BIT)


def _barrier(opt: int) -> str:
    return _BARRIER_OPTIONS.get(opt, f"#{opt}")


def _reg_list(registers: int) -> str:
    names = [_reg(i) for i in range(16) if registers & (1 << i)]
    if not names:
        return "{}"
    return "{" + ", ".join(names) + "}"


def _vfp_reg_list(single_regs: bool, d: int, count: int) -> str:
    if single_regs:
        names = [_sreg(d + i) for i in range(count)]
    else:
        names = [_dreg(d + i) for i in range(count)]
    return "{" + ", ".join(names) + "}"


def _vfp_reg(dp_operation: bool, r: int) -> str:
    return _dreg(r) if dp_operation else _sreg(r)


# ---------------------------------------------------------------------------
# Addressing-mode helpers
# ---------------------------------------------------------------------------


def _addr_imm(result: Any) -> str:
    """`[Rn, #imm]` in the shape the encoding calls for.

    A 32-bit encoding leaves a zero offset out of the offset and pre-indexed
    forms -- `[ip]`, `[ip]!` -- because it has room to encode one and nothing
    else can be meant. The 16-bit forms spell it (`[r0, #0]`), as does every
    post-indexed form (`[r0], #0`), where the offset is what advances Rn.
    """
    n, imm32 = result.n, result.imm32
    sign = "" if result.add else "-"
    hc = "" if n == 15 else _hex_comment(imm32)
    if not result.index:
        return f"[{_reg(n)}], #{sign}{imm32}{hc}"
    offset_text = "" if imm32 == 0 and not _is_narrow(result) else f", #{sign}{imm32}"
    wb = "!" if result.wback else ""
    return f"[{_reg(n)}{offset_text}]{wb}{hc}"


def _addr_imm_dual(result: Any) -> str:
    return f"{_reg(result.t)}, {_reg(result.t2)}, {_addr_imm(result)}"


def _addr_reg(t: int, n: int, m: int, shift_t: int, shift_n: int) -> str:
    if shift_n == 0:
        return f"{_reg(t)}, [{_reg(n)}, {_reg(m)}]"
    return f"{_reg(t)}, [{_reg(n)}, {_reg(m)}, lsl #{shift_n}]"


def _addr_excl(d: int, t: int, n: int, imm32: int) -> str:
    if imm32 == 0:
        return f"{_reg(d)}, {_reg(t)}, [{_reg(n)}]"
    return f"{_reg(d)}, {_reg(t)}, [{_reg(n)}, #{imm32}]{_hex_comment(imm32)}"


def _addr_excl_single(t: int, n: int, imm32: int) -> str:
    if imm32 == 0:
        return f"{_reg(t)}, [{_reg(n)}]"
    return f"{_reg(t)}, [{_reg(n)}, #{imm32}]{_hex_comment(imm32)}"


def _addr_literal(result: Any) -> str:
    sign = "" if result.add else "-"
    imm32 = result.imm32
    offset_text = "" if imm32 == 0 and not _is_narrow(result) else f", #{sign}{imm32}"
    return f"{_reg(result.t)}, [pc{offset_text}]"


def _addr_unpriv(t: int, n: int, imm32: int, register_form: bool) -> str:
    if register_form:
        return f"{_reg(t)}, [{_reg(n)}], {_reg(imm32)}"
    if imm32 == 0:
        return f"{_reg(t)}, [{_reg(n)}]"
    return f"{_reg(t)}, [{_reg(n)}, #{imm32}]{_hex_comment(imm32)}"


def _addr_unpriv_ldr(t: int, n: int, imm32: int) -> str:
    if imm32 == 0:
        return f"{_reg(t)}, [{_reg(n)}]"
    return f"{_reg(t)}, [{_reg(n)}, #{imm32}]{_hex_comment(imm32)}"


def _branch_target(offset: int, imm32: int) -> str:
    return f"0x{((offset + 4 + imm32) & 0xFFFFFFFF):x}"



def _hex_comment(imm: int) -> str:
    return f"\t@ 0x{imm:x}" if imm > 32 or imm < -16 else ""


def _hex_target(offset: int, imm32: int, add: bool = True) -> str:
    base = ((offset + 4) & ~3) & 0xFFFFFFFF
    target = (base + (imm32 if add else -imm32)) & 0xFFFFFFFF
    return f"\t@ (0x{target:x})"


# ---------------------------------------------------------------------------
# Combined-mnemonic helpers
# ---------------------------------------------------------------------------


def _is_coproc2(instr: int) -> bool:
    return bool(instr & 0x10000000)


def _is_rdn_encoding(result: Any, instr: int) -> bool:
    """Whether the encoding names its destination once, as `<Rdn>`.

    Three 16-bit groups do, and write two operands where the wide forms
    write three: ADD/SUB (immediate) T2 (0x3000), the data-processing group
    (0x4000) and ADD (register) T2 (0x4400). Nothing else does -- `d == n`
    is not the test, because the 16-bit T1 forms take three registers and
    two of them may well be the same one: 18ad is `adds r5, r5, r2`.

    Which encoding produced `result` is not in `result`, so this needs the
    instruction word; without one the wide, always-valid form is used.
    """
    if not _is_narrow(result):
        return False
    return 0x3000 <= (instr >> 16) & 0xFFFF < 0x4700


def _nhigh_mhigh_mnemonic(base: str, n_high: bool, m_high: bool) -> str:
    if not n_high and not m_high:
        return base + "bb"
    if not n_high and m_high:
        return base + "bt"
    if n_high and not m_high:
        return base + "tb"
    return base + "tt"


def _mswap_mnemonic(base: str, m_swap: bool) -> str:
    return base + ("x" if m_swap else "")


def _mhigh_mnemonic(base: str, m_high: bool) -> str:
    return base + ("t" if m_high else "b")


def _round_mnemonic(base: str, round_val: bool) -> str:
    return base + ("r" if round_val else "")


# ---------------------------------------------------------------------------
# Instruction formatters — organized by category
# ---------------------------------------------------------------------------


# --- Data-processing immediate ---


def _fmt_dp_imm(result: Any, mnemonic: str, instr: int = 0) -> str:
    s = _flags(result.setflags) + _width(result, mnemonic)
    if _is_rdn_encoding(result, instr):
        return (
            f"{mnemonic}{s}{_SEP}{_reg(result.d)}, #{result.imm32}"
            f"{_hex_comment(result.imm32)}"
        )
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.imm32}"
        f"{_hex_comment(result.imm32)}"
    )


def _fmt_mov_imm(result: Any, instr: int = 0) -> str:
    s = _flags(result.setflags) + _width(result, "mov")
    # T2 (mov.w) and T3 (movw) are both 32-bit, so the width alone cannot tell
    # them apart -- bits 25:20 pick out T3. The width test keeps a 16-bit T1
    # word, whose low halfword is zeroed, from aliasing onto that bit pattern.
    if (result.decoder_state & DecoderState.DECODED_32BIT) and (
        instr >> 20
    ) & 0x3F == 0x24:
        return (
            f"movw{_SEP}{_reg(result.d)}, #{result.imm32}{_hex_comment(result.imm32)}"
        )
    return f"mov{s}{_SEP}{_reg(result.d)}, #{result.imm32}{_hex_comment(result.imm32)}"


def _fmt_mvn_imm(result: Any, mnemonic: str = "mvn") -> str:
    s = _flags(result.setflags) + _width(result, mnemonic)
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)},"
        f" #{result.imm32}{_hex_comment(result.imm32)}"
    )


def _fmt_and_imm(result: Any, mnemonic: str = "and") -> str:
    s = _flags(getattr(result, "setflags", False)) + _width(result, mnemonic)
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.imm32}"
        f"{_hex_comment(result.imm32)}"
    )


# --- Data-processing register ---


def _fmt_dp_reg(result: Any, mnemonic: str, instr: int = 0) -> str:
    s = _flags(result.setflags) + _width(result, mnemonic)
    sh = _shift(result.shift_t, result.shift_n)
    if _is_rdn_encoding(result, instr):
        return f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}{sh}"
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}{sh}"
    )


def _fmt_mov_reg(result: Any) -> str:
    s = _flags(result.setflags) + _width(result, "mov")
    return f"mov{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_mvn_reg(result: Any) -> str:
    s = _flags(result.setflags) + _width(result, "mvn")
    sh = _shift(result.shift_t, result.shift_n)
    return f"mvn{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}{sh}"


def _fmt_rrx(result: Any) -> str:
    s = _flags(result.setflags)
    return f"rrx{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}"


# --- Test/compare ---


def _fmt_test_imm(result: Any, mnemonic: str) -> str:
    w = _width(result, mnemonic)
    return (
        f"{mnemonic}{w}{_SEP}{_reg(result.n)},"
        f" #{result.imm32}{_hex_comment(result.imm32)}"
    )


def _fmt_test_reg(result: Any, mnemonic: str) -> str:
    sh = _shift(result.shift_t, result.shift_n)
    m = mnemonic + _width(result, mnemonic)
    return f"{m}{_SEP}{_reg(result.n)}, {_reg(result.m)}{sh}"


# --- Shift immediate ---


def _fmt_shift_imm(result: Any, mnemonic: str) -> str:
    s = _flags(result.setflags) + _width(result, mnemonic)
    return f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}, #{result.shift_n}"


def _fmt_ror_imm(result: Any) -> str:
    s = _flags(result.setflags) + _width(result, "ror")
    return f"ror{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}, #{result.shift_n}"


# --- Shift register ---


def _fmt_shift_reg(result: Any, mnemonic: str, instr: int = 0) -> str:
    s = _flags(result.setflags) + _width(result, mnemonic)
    if _is_rdn_encoding(result, instr):
        return f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}"
    return f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# --- SP arithmetic ---


# Only the 16-bit encodings of SP arithmetic leave SP out of the operands
# ("add sp, #4", "add r0, sp"); the wide forms spell it as the second operand
# even when that repeats the destination.


def _fmt_add_sp_imm(result: Any) -> str:
    s = _flags(result.setflags) + _width(result, "add")
    if result.d == 13 and _is_narrow(result):
        return f"add{s}{_SEP}sp, #{result.imm32}{_hex_comment(result.imm32)}"
    return (
        f"add{s}{_SEP}{_reg(result.d)}, sp, #{result.imm32}{_hex_comment(result.imm32)}"
    )


def _fmt_add_sp_reg(result: Any) -> str:
    s = _flags(result.setflags) + _width(result, "add")
    sh = _shift(result.shift_t, result.shift_n)
    if _is_narrow(result):
        if result.d == 13:
            return f"add{s}{_SEP}sp, {_reg(result.m)}{sh}"
        if result.d == result.m:
            return f"add{s}{_SEP}{_reg(result.d)}, sp{sh}"
    return f"add{s}{_SEP}{_reg(result.d)}, sp, {_reg(result.m)}{sh}"


def _fmt_sub_sp_imm(result: Any) -> str:
    s = _flags(result.setflags) + _width(result, "sub")
    if result.d == 13 and _is_narrow(result):
        return f"sub{s}{_SEP}sp, #{result.imm32}{_hex_comment(result.imm32)}"
    return (
        f"sub{s}{_SEP}{_reg(result.d)}, sp, #{result.imm32}{_hex_comment(result.imm32)}"
    )


def _fmt_sub_sp_reg(result: Any) -> str:
    s = _flags(result.setflags) + _width(result, "sub")
    sh = _shift(result.shift_t, result.shift_n)
    if _is_narrow(result):
        if result.d == 13:
            return f"sub{s}{_SEP}sp, {_reg(result.m)}{sh}"
        if result.d == result.m:
            return f"sub{s}{_SEP}{_reg(result.d)}, sp{sh}"
    return f"sub{s}{_SEP}{_reg(result.d)}, sp, {_reg(result.m)}{sh}"


# --- ADR ---


def _fmt_adr(result: Any, offset: int = 0) -> str:
    """ADR renders as the PC-relative add/sub it encodes, never as `adr`.

    The wide encodings are the unshifted-immediate `addw`/`subw` forms; the
    narrow one is `add Rd, pc, #imm` and carries the resolved target as a
    comment, since the immediate is relative to Align(PC, 4).
    """
    d = _reg(result.d)
    if result.decoder_state & DecoderState.DECODED_32BIT:
        mnemonic = "addw" if result.add else "subw"
        return f"{mnemonic}{_SEP}{d}, pc, #{result.imm32}{_hex_comment(result.imm32)}"
    base = ((offset + 4) & ~3) & 0xFFFFFFFF
    target = (base + (result.imm32 if result.add else -result.imm32)) & 0xFFFFFFFF
    return f"add{_SEP}{d}, pc, #{result.imm32}\t@ (adr {d}, 0x{target:x})"


# --- Load/store immediate ---


def _fmt_ldst_imm(result: Any, mnemonic: str) -> str:
    addr = _addr_imm(result)
    return f"{mnemonic}{_width(result, mnemonic)}{_SEP}{addr}"


def _fmt_ldst_imm_t(result: Any, mnemonic: str, offset: int = 0) -> str:
    addr = _addr_imm(result)
    asm = f"{mnemonic}{_width(result, mnemonic)}{_SEP}{_reg(result.t)}, {addr}"
    if result.n == 15:
        asm += _hex_target(offset, result.imm32, result.add)
    return asm


def _fmt_ldst_imm_dual(result: Any, mnemonic: str) -> str:
    addr = _addr_imm_dual(result)
    return f"{mnemonic}{_width(result, mnemonic)}{_SEP}{addr}"


# --- Load/store register ---


def _fmt_ldst_reg(result: Any, mnemonic: str) -> str:
    addr = _addr_reg(result.t, result.n, result.m, result.shift_t, result.shift_n)
    return f"{mnemonic}{_width(result, mnemonic)}{_SEP}{addr}"


def _fmt_ldst_reg_rt(result: Any, mnemonic: str) -> str:
    addr = _addr_reg(result.t, result.n, result.m, result.shift_t, result.shift_n)
    return f"{mnemonic}{_width(result, mnemonic)}{_SEP}{addr}"


# --- Load/store literal ---


def _fmt_ldst_lit(result: Any, mnemonic: str, offset: int = 0) -> str:
    m = mnemonic + _width(result, mnemonic)
    asm = f"{m}{_SEP}{_addr_literal(result)}"
    return f"{asm}{_hex_target(offset, result.imm32, result.add)}"


def _fmt_ldst_lit_dual(result: Any, mnemonic: str = "ldrd", offset: int = 0) -> str:
    sign = "" if result.add else "-"
    return (
        f"{mnemonic}{_SEP}{_reg(result.t)}, {_reg(result.t2)},"
        f" [pc, #{sign}{result.imm32}]{_hex_target(offset, result.imm32, result.add)}"
    )


# --- Preload (PLD / PLI) ---


def _fmt_pld_imm(result: Any) -> str:
    sign = "" if result.add else "-"
    return (
        f"pld{_SEP}[{_reg(result.n)},"
        f" #{sign}{result.imm32}]{_hex_comment(result.imm32)}"
    )


def _fmt_pld_lit(result: Any, offset: int = 0) -> str:
    sign = "" if result.add else "-"
    return (
        f"pld{_SEP}[pc, #{sign}{result.imm32}]"
        f"{_hex_target(offset, result.imm32, result.add)}"
    )


def _fmt_pld_reg(result: Any) -> str:
    if result.shift_n == 0:
        return f"pld{_SEP}[{_reg(result.n)}, {_reg(result.m)}]"
    return f"pld{_SEP}[{_reg(result.n)}, {_reg(result.m)}, lsl #{result.shift_n}]"


def _fmt_pli_imm_lit(result: Any) -> str:
    sign = "" if result.add else "-"
    return (
        f"pli{_SEP}[{_reg(result.n)},"
        f" #{sign}{result.imm32}]{_hex_comment(result.imm32)}"
    )


def _fmt_pli_reg(result: Any) -> str:
    if result.shift_n == 0:
        return f"pli{_SEP}[{_reg(result.n)}, {_reg(result.m)}]"
    return f"pli{_SEP}[{_reg(result.n)}, {_reg(result.m)}, lsl #{result.shift_n}]"


# --- Exclusive load/store ---


def _fmt_strex(result: Any) -> str:
    return f"strex{_SEP}{_addr_excl(result.d, result.t, result.n, result.imm32)}"


def _fmt_ldrex(result: Any) -> str:
    return f"ldrex{_SEP}{_addr_excl_single(result.t, result.n, result.imm32)}"


def _fmt_strexb(result: Any) -> str:
    return f"strexb{_SEP}{_reg(result.d)}, {_reg(result.t)}, [{_reg(result.n)}]"


def _fmt_ldrexb(result: Any) -> str:
    return f"ldrexb{_SEP}{_reg(result.t)}, [{_reg(result.n)}]"


def _fmt_strexh(result: Any) -> str:
    return f"strexh{_SEP}{_reg(result.d)}, {_reg(result.t)}, [{_reg(result.n)}]"


def _fmt_ldrexh(result: Any) -> str:
    return f"ldrexh{_SEP}{_reg(result.t)}, [{_reg(result.n)}]"


# --- Unprivileged load/store ---


def _fmt_unpriv_ldr(result: Any, mnemonic: str) -> str:
    if result.register_form:
        return (
            f"{mnemonic}{_SEP}{_reg(result.t)},"
            f" [{_reg(result.n)}], {_reg(result.imm32)}"
        )
    if result.imm32 == 0:
        return f"{mnemonic}{_SEP}{_reg(result.t)}, [{_reg(result.n)}]"
    return (
        f"{mnemonic}{_SEP}{_reg(result.t)}, [{_reg(result.n)}, #{result.imm32}]"
        f"{_hex_comment(result.imm32)}"
    )


def _fmt_unpriv_str(result: Any, mnemonic: str) -> str:
    return _fmt_unpriv_ldr(result, mnemonic)


# --- Multiple transfer ---


def _fmt_multi_xfer(result: Any, mnemonic: str) -> str:
    wb = "!" if result.wback else ""
    m = mnemonic + _width(result, mnemonic)
    return f"{m}{_SEP}{_reg(result.n)}{wb}, {_reg_list(result.registers)}"


# --- Stack ---


def _fmt_pop(result: Any) -> str:
    # Only the 16-bit T1 form is spelled "pop"; the wide forms render as the
    # LDM/LDR they encode. UnalignedAllowed marks the single-register T3.
    if result.decoder_state == DecoderState.DECODED_16BIT:
        return f"pop{_SEP}{_reg_list(result.registers)}"
    if result.UnalignedAllowed:
        return f"ldr{_width(result, 'ldr')}{_SEP}{_reg(result.t)}, [sp], #4"
    return f"ldmia{_width(result, 'ldmia')}{_SEP}sp!, {_reg_list(result.registers)}"


def _fmt_push(result: Any) -> str:
    if result.decoder_state == DecoderState.DECODED_16BIT:
        return f"push{_SEP}{_reg_list(result.registers)}"
    if result.UnalignedAllowed:
        return f"str{_width(result, 'str')}{_SEP}{_reg(result.t)}, [sp, #-4]!"
    return f"stmdb{_SEP}sp!, {_reg_list(result.registers)}"


# --- Branch ---


def _fmt_b(result: Any, offset: int = 0, instr: int = 0) -> str:
    c = _cond(result.cond)
    # .n and .w occupy the same slot; B is the only mnemonic that spells both.
    if result.decoder_state == DecoderState.DECODED_16BIT:
        width = ".n"
    else:
        width = _width(result, "b")
    return f"b{c}{width}{_SEP}{_branch_target(offset, result.imm32)}"


def _fmt_bl(result: Any, offset: int = 0) -> str:
    return f"bl{_SEP}{_branch_target(offset, result.imm32)}"


def _fmt_blx_reg(result: Any) -> str:
    return f"blx{_SEP}{_reg(result.m)}"


def _fmt_bx(result: Any) -> str:
    return f"bx{_SEP}{_reg(result.m)}"


def _fmt_cbnz_cbz(result: Any, offset: int = 0) -> str:
    mnemonic = "cbnz" if result.nonzero else "cbz"
    return f"{mnemonic}{_SEP}{_reg(result.n)}, {_branch_target(offset, result.imm32)}"


def _fmt_tbb_tbh(result: Any, offset: int = 0) -> str:
    mnemonic = "tbh" if result.is_tbh else "tbb"
    return f"{mnemonic}{_SEP}[{_reg(result.n)}, {_reg(result.m)}]"


# --- Barrier ---


def _fmt_dmb(result: Any) -> str:
    return f"dmb{_SEP}{_barrier(result.option)}"


def _fmt_dsb(result: Any) -> str:
    if result.option == 0xC:
        return "dfb"
    return f"dsb{_SEP}{_barrier(result.option)}"


def _fmt_isb(result: Any) -> str:
    # ISB names only the full-system option; everything else stays numeric.
    option = "sy" if result.option == 0xF else f"#{result.option}"
    return f"isb{_SEP}{option}"


# --- Bitfield ---


def _fmt_bfi(result: Any) -> str:
    width = result.msbit - result.lsbit + 1
    return f"bfi{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.lsbit}, #{width}"


def _fmt_bfc(result: Any) -> str:
    width = result.msbit - result.lsbit + 1
    return f"bfc{_SEP}{_reg(result.d)}, #{result.lsbit}, #{width}"


def _fmt_ubfx(result: Any) -> str:
    width = result.widthminus1 + 1
    return f"ubfx{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.lsbit}, #{width}"


def _fmt_sbfx(result: Any) -> str:
    width = result.widthminus1 + 1
    return f"sbfx{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.lsbit}, #{width}"


# --- Extend ---


def _fmt_extend(result: Any, mnemonic: str) -> str:
    m = mnemonic + _width(result, mnemonic)
    if result.rotation == 0:
        return f"{m}{_SEP}{_reg(result.d)}, {_reg(result.m)}"
    rot = f"ror #{result.rotation}"
    return f"{m}{_SEP}{_reg(result.d)}, {_reg(result.m)}, {rot}"


def _fmt_extab(result: Any, mnemonic: str) -> str:
    if result.rotation == 0:
        return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"
    rot = f"ror #{result.rotation}"
    return (
        f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {rot}"
    )


# --- Saturate ---


def _fmt_ssat(result: Any) -> str:
    sh = _shift(result.shift_t, result.shift_n)
    return f"ssat{_SEP}{_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}{sh}"


def _fmt_usat(result: Any) -> str:
    sh = _shift(result.shift_t, result.shift_n)
    return f"usat{_SEP}{_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}{sh}"


def _fmt_ssat16(result: Any) -> str:
    return f"ssat16{_SEP}{_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}"


def _fmt_usat16(result: Any) -> str:
    return f"usat16{_SEP}{_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}"


# --- SIMD 3-register ---


def _fmt_simd3(result: Any, mnemonic: str) -> str:
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# --- SIMD 3-register + accumulator ---


def _fmt_smla(result: Any, mnemonic: str, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)},"
        f" {_reg(result.m)}, {_reg(result.a)}"
    )


def _fmt_smlal(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"smlal{s}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


# --- SIMD long multiply ---


def _fmt_smull(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"smull{s}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


def _fmt_umull(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"umull{s}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


def _fmt_umlal(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"umlal{s}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


def _fmt_umaal(result: Any) -> str:
    return (
        f"umaal{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


# --- Multiply ---


def _fmt_mul(result: Any, setflags: bool = False, instr: int = 0) -> str:
    s = _flags(setflags) + _width(result, "mul")
    # T1 spells the destination as <Rdm>, so it is the multiplicand that goes
    # unwritten here, not the multiplier the other Rdn encodings drop.
    if _is_rdn_encoding(result, instr):
        return f"mul{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}"
    return f"mul{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_mla(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"mla{s}{_SEP}{_reg(result.d)}, {_reg(result.n)},"
        f" {_reg(result.m)}, {_reg(result.a)}"
    )


def _fmt_mls(result: Any) -> str:
    return (
        f"mls{_SEP}{_reg(result.d)}, {_reg(result.n)},"
        f" {_reg(result.m)}, {_reg(result.a)}"
    )


def _fmt_sdiv(result: Any) -> str:
    return f"sdiv{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_udiv(result: Any) -> str:
    return f"udiv{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# --- Miscellaneous ---


def _fmt_clz(result: Any) -> str:
    return f"clz{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_rev(result: Any) -> str:
    w = _width(result, "rev")
    return f"rev{w}{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_rev16(result: Any) -> str:
    w = _width(result, "rev16")
    return f"rev16{w}{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_revsh(result: Any) -> str:
    w = _width(result, "revsh")
    return f"revsh{w}{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_rbit(result: Any) -> str:
    return f"rbit{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_sel(result: Any) -> str:
    return f"sel{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_pkhbt_pkhtb(result: Any) -> str:
    mnemonic = "pkhtb" if result.tbform else "pkhbt"
    sh = _shift(result.shift_t, result.shift_n)
    # When tbform is False, shift type is LSL; when True, ASR
    if mnemonic == "pkhbt" and result.shift_t == 0 and result.shift_n == 0:
        return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}{sh}"


# --- MOVT ---


def _fmt_movt(result: Any) -> str:
    return f"movt{_SEP}{_reg(result.d)}, #{result.imm16}{_hex_comment(result.imm16)}"


# --- USAD8 / USADA8 ---


def _fmt_usad8(result: Any) -> str:
    return f"usad8{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_usada8(result: Any) -> str:
    return (
        f"usada8{_SEP}{_reg(result.d)}, {_reg(result.n)},"
        f" {_reg(result.m)}, {_reg(result.a)}"
    )


# --- System registers ---


def _fmt_mrs(result: Any) -> str:
    spec = _SPEC_REGS.get(result.SYSm, f"spec_reg_{result.SYSm:#x}")
    return f"mrs{_SEP}{_reg(result.d)}, {spec}"


def _fmt_msr(result: Any) -> str:
    spec = _SPEC_REGS.get(result.SYSm, f"spec_reg_{result.SYSm:#x}")
    return f"msr{_SEP}{spec}, {_reg(result.n)}"


# --- CPS ---


def _fmt_cps(result: Any) -> str:
    effect = "ie" if result.enable else "id"
    flags = ""
    if result.affectPRI:
        flags += "i"
    if result.affectFAULT:
        flags += "f"
    return f"cps{effect}{_SEP}{flags}"


# --- IT ---


def _fmt_it(result: Any) -> str:
    # The lowest set bit of the mask terminates the block; the bits above it
    # are the conditions of the second, third and fourth slot, one bit each.
    mask = result.mask & 0xF
    if mask & 0x1:
        slots = 3
    elif mask & 0x2:
        slots = 2
    elif mask & 0x4:
        slots = 1
    else:
        slots = 0
    suffix = "".join(_it_te(mask, 3 - i, result.firstcond) for i in range(slots))
    return f"it{suffix}{_SEP}{_COND_CODES[result.firstcond]}"


def _it_te(mask: int, bit: int, firstcond: int) -> str:
    # A slot runs on `firstcond` ("then") when its mask bit repeats
    # firstcond<0>, and on the inverse condition ("else") when it flips it --
    # ITAdvance shifts the bit into ITSTATE<4>, which is cond<0>.
    return "t" if (mask >> bit) & 1 == firstcond & 1 else "e"


# --- Misc zero-operand ---


def _fmt_noargs(result: Any, mnemonic: str) -> str:
    return f"{mnemonic}{_width(result, mnemonic)}"


def _fmt_bkpt(result: Any) -> str:
    return f"bkpt{_SEP}0x{result.imm32:04x}"


def _fmt_svc(result: Any) -> str:
    return f"svc{_SEP}{result.imm32}{_hex_comment(result.imm32)}"


def _fmt_udf(result: Any) -> str:
    w = _width(result, "udf")
    return f"udf{w}{_SEP}#{result.imm32}{_hex_comment(result.imm32)}"


def _fmt_dbg(result: Any) -> str:
    return f"dbg{_SEP}#{result.option}"


# --- VFP data-processing 3-reg ---


def _fmt_vfp_dp3(result: Any, mnemonic: str) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


# --- VFP data-processing 2-reg ---


def _fmt_vfp_dp2(result: Any, mnemonic: str) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {m}"


# --- VFP combined mnemonics ---


def _fmt_vfma_vfms(result: Any) -> str:
    mnemonic = "vfms" if result.op1_neg else "vfma"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


def _fmt_vfnma_vfnms(result: Any) -> str:
    mnemonic = "vfnms" if result.op1_neg else "vfnma"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


def _fmt_vmla_vmls(result: Any) -> str:
    mnemonic = "vmls" if not result.add else "vmla"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


def _fmt_vnmla_vnmls_vnmul(result: Any) -> str:
    types = {0: "vnmla", 1: "vnmls", 2: "vnmul"}
    mnemonic = types.get(result.type, "vnmla")
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


# --- VFP compare ---


def _fmt_vcmp_vcmpe(result: Any) -> str:
    mnemonic = "vcmpe" if result.quiet_nan_exc else "vcmp"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    if result.with_zero:
        return f"{mnemonic}{precision}{_SEP}{d}, #0.0"
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {m}"


# --- VFP convert ---

_VCVTA_MODES = {0: "vcvta", 1: "vcvtn", 2: "vcvtp", 3: "vcvtm"}


def _fmt_vcvta_round(result: Any) -> str:
    mnemonic = _VCVTA_MODES.get(result.round_mode, "vcvta")
    signed = "u" if result.unsigned else "s"
    src_prec = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}.{signed}32{src_prec}{_SEP}{d}, {m}"


def _fmt_vcvt_integer(result: Any) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    unsigned = "u" if result.unsigned else ""
    rounding = "r" if (result.round_zero or result.round_nearest) else ""
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    if result.to_integer:
        return f"vcvt{rounding}{unsigned}.s32{precision}{_SEP}{d}, {m}"
    return f"vcvt{rounding}{unsigned}{precision}.s32{_SEP}{d}, {m}"


def _fmt_vcvt_fixed(result: Any) -> str:
    unsigned = "u" if result.unsigned else ""
    d = _sreg(result.d)
    if result.to_fixed:
        return f"vcvt{unsigned}.s32.f32{_SEP}{d}, {d}, #{result.frac_bits}"
    return f"vcvt{unsigned}.f32.s32{_SEP}{d}, {d}, #{result.frac_bits}"


def _fmt_vcvt_double_single(result: Any) -> str:
    if result.double_to_single:
        return f"vcvt.f32.f64{_SEP}{_sreg(result.d)}, {_dreg(result.m)}"
    return f"vcvt.f64.f32{_SEP}{_dreg(result.d)}, {_sreg(result.m)}"


def _fmt_vcvtb_vcvtt(result: Any) -> str:
    mnemonic = "vcvtt" if result.lowbit else "vcvtb"
    return f"{mnemonic}.f32.f16{_SEP}{_sreg(result.d)}, {_sreg(result.m)}"


# --- VFP move ---


def _fmt_vmov_imm(result: Any) -> str:
    return (
        f"vmov{_SEP}{_vfp_reg(result.dp_operation, result.d)}, #{result.imm32}"
        f"{_hex_comment(result.imm32)}"
    )


def _fmt_vmov_reg(result: Any) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"vmov{precision}{_SEP}{d}, {m}"


def _fmt_vmov_core_to_scalar(result: Any) -> str:
    return f"vmov{_SEP}{_sreg(result.d)}, {_reg(result.t)}"


def _fmt_vmov_scalar_to_core(result: Any) -> str:
    return f"vmov{_SEP}{_reg(result.t)}, {_sreg(result.n)}"


def _fmt_vmov_core_and_single(result: Any) -> str:
    if result.to_arm_register:
        return f"vmov{_SEP}{_reg(result.t)}, {_sreg(result.n)}"
    return f"vmov{_SEP}{_sreg(result.n)}, {_reg(result.t)}"


def _fmt_vmov_two_core_and_single(result: Any) -> str:
    if result.to_arm_registers:
        return (
            f"vmov{_SEP}{_reg(result.t)}, {_reg(result.t2)},"
            f" {_sreg(result.m)}, {_sreg(result.m + 1)}"
        )
    return (
        f"vmov{_SEP}{_sreg(result.m)}, {_sreg(result.m + 1)},"
        f" {_reg(result.t)}, {_reg(result.t2)}"
    )


def _fmt_vmov_two_core_and_doubleword(result: Any) -> str:
    if result.to_arm_registers:
        return f"vmov{_SEP}{_reg(result.t)}, {_reg(result.t2)}, {_dreg(result.m)}"
    return f"vmov{_SEP}{_dreg(result.m)}, {_reg(result.t)}, {_reg(result.t2)}"


# --- VFP system ---


def _fmt_vmrs(result: Any) -> str:
    if result.t == 15:
        return f"vmrs{_SEP}apsr_nzcv, fpscr"
    return f"vmrs{_SEP}{_reg(result.t)}, fpscr"


def _fmt_vmsr(result: Any) -> str:
    return f"vmsr{_SEP}fpscr, {_reg(result.t)}"


# --- VFP conditional select ---


def _fmt_vsel(result: Any) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    c = _cond(result.cond)
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"vsel{c}{precision}{_SEP}{d}, {n}, {m}"


# --- VFP round ---

_VRINT_MODES = {0: "vrinta", 1: "vrintn", 2: "vrintp", 3: "vrintm"}
_VRINTZ_MODES = {0: "vrintz", 1: "vrintr"}


def _fmt_vrint_round(result: Any) -> str:
    mnemonic = _VRINT_MODES.get(result.rmode, "vrinta")
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {m}"


def _fmt_vrint_zr(result: Any) -> str:
    mnemonic = _VRINTZ_MODES.get(result.rmode, "vrintz")
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {m}"


def _fmt_vrintx(result: Any) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"vrintx{precision}{_SEP}{d}, {m}"


# --- VFP load/store ---


def _fmt_vldr_vstr(result: Any, mnemonic: str) -> str:
    sign = "" if result.add else "-"
    single_reg = result.single_reg
    reg_name_fn = _sreg if single_reg else _dreg
    # 32-bit only, and an offset form throughout: a zero offset goes unwritten.
    offset_text = "" if result.imm32 == 0 else f", #{sign}{result.imm32}"
    return (
        f"{mnemonic}{_SEP}{reg_name_fn(result.d)},"
        f" [{_reg(result.n)}{offset_text}]"
        f"{_hex_comment(result.imm32)}"
    )


def _fmt_vldm_vstm(result: Any, mnemonic: str) -> str:
    wb = "!" if result.wback else ""
    reg_str = _vfp_reg_list(result.single_regs, result.d, result.regs)
    return f"{mnemonic}{_SEP}{_reg(result.n)}{wb}, {reg_str}"


def _fmt_vpush_vpop(result: Any, mnemonic: str) -> str:
    reg_str = _vfp_reg_list(result.single_regs, result.d, result.regs)
    return f"{mnemonic}{_SEP}{reg_str}"


# --- VFP max/min ---


def _fmt_vmaxnm_vminnm(result: Any) -> str:
    mnemonic = "vmaxnm" if result.maximum else "vminnm"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


# --- SIMD combined mnemonics ---


def _fmt_smlabb_variants(result: Any) -> str:
    mnemonic = _nhigh_mhigh_mnemonic("smla", result.n_high, result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlalbb_variants(result: Any) -> str:
    mnemonic = _nhigh_mhigh_mnemonic("smlal", result.n_high, result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)}, {_reg(result.n)}, {_reg(result.m)}"  # noqa: E501


def _fmt_smulbb_variants(result: Any) -> str:
    mnemonic = _nhigh_mhigh_mnemonic("smul", result.n_high, result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smlad_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smlad", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlald_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smlald", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)}, {_reg(result.n)}, {_reg(result.m)}"  # noqa: E501


def _fmt_smlaw_variants(result: Any) -> str:
    mnemonic = _mhigh_mnemonic("smlaw", result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlsd_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smlsd", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlsld_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smlsld", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)}, {_reg(result.n)}, {_reg(result.m)}"  # noqa: E501


def _fmt_smmla_variants(result: Any) -> str:
    mnemonic = _round_mnemonic("smmla", result.round)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smmls_variants(result: Any) -> str:
    mnemonic = _round_mnemonic("smmls", result.round)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smmul_variants(result: Any) -> str:
    mnemonic = _round_mnemonic("smmul", result.round)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smuad_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smuad", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smulw_variants(result: Any) -> str:
    mnemonic = _mhigh_mnemonic("smulw", result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smusd_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smusd", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# ---------------------------------------------------------------------------
# Coprocessor formatters
# ---------------------------------------------------------------------------


def _fmt_cdp_cdp2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"cdp{suffix}{_SEP}p{result.cp}, #{result.opc1}, {_reg(result.CRd)}, c{result.CRn}, c{result.CRm}, #{result.opc2}"  # noqa: E501


def _fmt_mcr_mcr2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"mcr{suffix}{_SEP}p{result.cp}, #{result.opc1}, {_reg(result.t)}, c{result.CRn}, c{result.CRm}, #{result.opc2}"  # noqa: E501


def _fmt_mrc_mrc2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"mrc{suffix}{_SEP}p{result.cp}, #{result.opc1}, {_reg(result.t)}, c{result.CRn}, c{result.CRm}, #{result.opc2}"  # noqa: E501


def _fmt_mcrr_mcrr2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"mcrr{suffix}{_SEP}p{result.cp}, #{result.opc1}, {_reg(result.t)}, {_reg(result.t2)}, c{result.CRm}"  # noqa: E501


def _fmt_mrrc_mrrc2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"mrrc{suffix}{_SEP}p{result.cp}, #{result.opc1}, {_reg(result.t)}, {_reg(result.t2)}, c{result.CRm}"  # noqa: E501


def _fmt_stc_stc2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    addr = _addr_imm(result)
    return f"stc{suffix}{_SEP}{result.cp}, cr{result.CRd}, {addr}"


def _fmt_ldc_ldc2_imm(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    addr = _addr_imm(result)
    return f"ldc{suffix}{_SEP}p{result.cp}, c{result.CRd}, {addr}"


def _fmt_ldc_ldc2_lit(result: Any, instr: int = 0, offset: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    sign = "" if result.add else "-"
    return (
        f"ldc{suffix}{_SEP}p{result.cp}, c{result.CRd},"
        f" [pc, #{sign}{result.imm32}]{_hex_target(offset, result.imm32, result.add)}"
    )


# ---------------------------------------------------------------------------
# Shadow aliases (CPY, NEG, MOV_shifted_register)
# ---------------------------------------------------------------------------


def _fmt_cpy(result: Any) -> str:
    return f"cpy{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_neg(result: Any) -> str:
    return f"neg{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_mov_shifted(result: Any) -> str:
    s = _flags(result.setflags) + _width(result, "mov")
    sh = _shift(result.shift_t, result.shift_n)
    return f"mov{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}{sh}"


# ---------------------------------------------------------------------------
# Dispatch table — maps class name → (formatter_fn, extra_args)
# ---------------------------------------------------------------------------

_DISPATCH: dict[int, Any] = {
    Opcode.OP_ADC_IMMEDIATE: lambda r, instr=0: _fmt_dp_imm(r, "adc", instr),
    Opcode.OP_ADC_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "adc", instr),
    Opcode.OP_ADD_IMMEDIATE: lambda r, instr=0: _fmt_dp_imm(r, "add", instr),
    Opcode.OP_ADD_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "add", instr),
    Opcode.OP_ADD_SP_PLUS_IMMEDIATE: _fmt_add_sp_imm,
    Opcode.OP_ADD_SP_PLUS_REGISTER: _fmt_add_sp_reg,
    Opcode.OP_ADR: _fmt_adr,
    Opcode.OP_AND_IMMEDIATE: _fmt_and_imm,
    Opcode.OP_AND_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "and", instr),
    Opcode.OP_ASR_IMMEDIATE: lambda r: _fmt_shift_imm(r, "asr"),
    Opcode.OP_ASR_REGISTER: lambda r, instr=0: _fmt_shift_reg(r, "asr", instr),
    Opcode.OP_B: _fmt_b,
    Opcode.OP_BFC: _fmt_bfc,
    Opcode.OP_BFI: _fmt_bfi,
    Opcode.OP_BIC_IMMEDIATE: lambda r: _fmt_and_imm(r, "bic"),
    Opcode.OP_BIC_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "bic", instr),
    Opcode.OP_BKPT: _fmt_bkpt,
    Opcode.OP_BL: _fmt_bl,
    Opcode.OP_BLX_REGISTER: _fmt_blx_reg,
    Opcode.OP_BX: _fmt_bx,
    Opcode.OP_CBNZ_CBZ: _fmt_cbnz_cbz,
    Opcode.OP_CDP_CDP2: _fmt_cdp_cdp2,
    Opcode.OP_CLREX: lambda r: _fmt_noargs(r, "clrex"),
    Opcode.OP_CLZ: _fmt_clz,
    Opcode.OP_CMN_IMMEDIATE: lambda r: _fmt_test_imm(r, "cmn"),
    Opcode.OP_CMN_REGISTER: lambda r: _fmt_test_reg(r, "cmn"),
    Opcode.OP_CMP_IMMEDIATE: lambda r: _fmt_test_imm(r, "cmp"),
    Opcode.OP_CMP_REGISTER: lambda r: _fmt_test_reg(r, "cmp"),
    Opcode.OP_CPS: _fmt_cps,
    "CPY": _fmt_cpy,
    Opcode.OP_CSDB: lambda r: _fmt_noargs(r, "csdb"),
    Opcode.OP_DBG: _fmt_dbg,
    Opcode.OP_DMB: _fmt_dmb,
    Opcode.OP_DSB: _fmt_dsb,
    Opcode.OP_EOR_IMMEDIATE: lambda r: _fmt_and_imm(r, "eor"),
    Opcode.OP_EOR_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "eor", instr),
    Opcode.OP_ISB: _fmt_isb,
    Opcode.OP_IT: _fmt_it,
    Opcode.OP_LDC_LDC2_IMMEDIATE: _fmt_ldc_ldc2_imm,
    Opcode.OP_LDC_LDC2_LITERAL: _fmt_ldc_ldc2_lit,
    Opcode.OP_LDM: lambda r: _fmt_multi_xfer(r, "ldmia"),
    Opcode.OP_LDMDB: lambda r: _fmt_multi_xfer(r, "ldmdb"),
    Opcode.OP_LDR_IMMEDIATE: lambda r, offset=0: _fmt_ldst_imm_t(
        r, "ldr", offset=offset
    ),
    Opcode.OP_LDR_LITERAL: lambda r, offset=0: _fmt_ldst_lit(r, "ldr", offset=offset),
    Opcode.OP_LDR_REGISTER: lambda r: _fmt_ldst_reg(r, "ldr"),
    Opcode.OP_LDRB_IMMEDIATE: lambda r, offset=0: _fmt_ldst_imm_t(
        r, "ldrb", offset=offset
    ),
    Opcode.OP_LDRB_LITERAL: lambda r, offset=0: _fmt_ldst_lit(r, "ldrb", offset=offset),
    Opcode.OP_LDRB_REGISTER: lambda r: _fmt_ldst_reg(r, "ldrb"),
    Opcode.OP_LDRBT: lambda r: _fmt_unpriv_ldr(r, "ldrbt"),
    Opcode.OP_LDRD_IMMEDIATE: lambda r: _fmt_ldst_imm_dual(r, "ldrd"),
    Opcode.OP_LDRD_LITERAL: _fmt_ldst_lit_dual,
    Opcode.OP_LDREX: _fmt_ldrex,
    Opcode.OP_LDREXB: _fmt_ldrexb,
    Opcode.OP_LDREXH: _fmt_ldrexh,
    Opcode.OP_LDRH_IMMEDIATE: lambda r, offset=0: _fmt_ldst_imm_t(
        r, "ldrh", offset=offset
    ),
    Opcode.OP_LDRH_LITERAL: lambda r, offset=0: _fmt_ldst_lit(r, "ldrh", offset=offset),
    Opcode.OP_LDRH_REGISTER: lambda r: _fmt_ldst_reg(r, "ldrh"),
    Opcode.OP_LDRHT: lambda r: _fmt_unpriv_ldr(r, "ldrht"),
    Opcode.OP_LDRSB_IMMEDIATE: lambda r, offset=0: _fmt_ldst_imm_t(
        r, "ldrsb", offset=offset
    ),
    Opcode.OP_LDRSB_LITERAL: lambda r, offset=0: _fmt_ldst_lit(
        r, "ldrsb", offset=offset
    ),
    Opcode.OP_LDRSB_REGISTER: lambda r: _fmt_ldst_reg(r, "ldrsb"),
    Opcode.OP_LDRSBT: lambda r: _fmt_unpriv_ldr(r, "ldrsbt"),
    Opcode.OP_LDRSH_IMMEDIATE: lambda r, offset=0: _fmt_ldst_imm_t(
        r, "ldrsh", offset=offset
    ),
    Opcode.OP_LDRSH_LITERAL: lambda r, offset=0: _fmt_ldst_lit(
        r, "ldrsh", offset=offset
    ),
    Opcode.OP_LDRSH_REGISTER: lambda r: _fmt_ldst_reg(r, "ldrsh"),
    Opcode.OP_LDRSHT: lambda r: _fmt_unpriv_ldr(r, "ldrsht"),
    Opcode.OP_LDRT: lambda r: _fmt_unpriv_ldr(r, "ldrt"),
    Opcode.OP_LSL_IMMEDIATE: lambda r: _fmt_shift_imm(r, "lsl"),
    Opcode.OP_LSL_REGISTER: lambda r, instr=0: _fmt_shift_reg(r, "lsl", instr),
    Opcode.OP_LSR_IMMEDIATE: lambda r: _fmt_shift_imm(r, "lsr"),
    Opcode.OP_LSR_REGISTER: lambda r, instr=0: _fmt_shift_reg(r, "lsr", instr),
    Opcode.OP_MCR_MCR2: _fmt_mcr_mcr2,
    Opcode.OP_MCRR_MCRR2: _fmt_mcrr_mcrr2,
    Opcode.OP_MLA: lambda r: _fmt_mla(r, getattr(r, "setflags", False)),
    Opcode.OP_MLS: _fmt_mls,
    Opcode.OP_MOV_IMMEDIATE: _fmt_mov_imm,
    Opcode.OP_MOV_REGISTER: _fmt_mov_reg,
    Opcode.OP_MOVT: _fmt_movt,
    Opcode.OP_MRC_MRC2: _fmt_mrc_mrc2,
    Opcode.OP_MRRC_MRRC2: _fmt_mrrc_mrrc2,
    Opcode.OP_MRS: _fmt_mrs,
    Opcode.OP_MSR: _fmt_msr,
    Opcode.OP_MUL: lambda r, instr=0: _fmt_mul(r, getattr(r, "setflags", False), instr),
    Opcode.OP_MVN_IMMEDIATE: _fmt_mvn_imm,
    Opcode.OP_MVN_REGISTER: _fmt_mvn_reg,
    Opcode.OP_NOP: lambda r: _fmt_noargs(r, "nop"),
    Opcode.OP_ORN_IMMEDIATE: lambda r: _fmt_and_imm(r, "orn"),
    Opcode.OP_ORN_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "orn", instr),
    Opcode.OP_ORR_IMMEDIATE: lambda r: _fmt_and_imm(r, "orr"),
    Opcode.OP_ORR_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "orr", instr),
    Opcode.OP_PKHBT_PKHTB: _fmt_pkhbt_pkhtb,
    Opcode.OP_PLD_IMMEDIATE: _fmt_pld_imm,
    Opcode.OP_PLD_LITERAL: _fmt_pld_lit,
    Opcode.OP_PLD_REGISTER: _fmt_pld_reg,
    Opcode.OP_PLI_IMMEDIATE_LITERAL: _fmt_pli_imm_lit,
    Opcode.OP_PLI_REGISTER: _fmt_pli_reg,
    Opcode.OP_POP: _fmt_pop,
    Opcode.OP_PSSBB: lambda r: _fmt_noargs(r, "pssbb"),
    Opcode.OP_PUSH: _fmt_push,
    Opcode.OP_QADD: lambda r: _fmt_simd3(r, "qadd"),
    Opcode.OP_QADD16: lambda r: _fmt_simd3(r, "qadd16"),
    Opcode.OP_QADD8: lambda r: _fmt_simd3(r, "qadd8"),
    Opcode.OP_QASX: lambda r: _fmt_simd3(r, "qasx"),
    Opcode.OP_QDADD: lambda r: _fmt_simd3(r, "qdadd"),
    Opcode.OP_QDSUB: lambda r: _fmt_simd3(r, "qdsub"),
    Opcode.OP_QSAX: lambda r: _fmt_simd3(r, "qsax"),
    Opcode.OP_QSUB: lambda r: _fmt_simd3(r, "qsub"),
    Opcode.OP_QSUB16: lambda r: _fmt_simd3(r, "qsub16"),
    Opcode.OP_QSUB8: lambda r: _fmt_simd3(r, "qsub8"),
    Opcode.OP_RBIT: _fmt_rbit,
    Opcode.OP_REV: _fmt_rev,
    Opcode.OP_REV16: _fmt_rev16,
    Opcode.OP_REVSH: _fmt_revsh,
    Opcode.OP_ROR_IMMEDIATE: _fmt_ror_imm,
    Opcode.OP_ROR_REGISTER: lambda r, instr=0: _fmt_shift_reg(r, "ror", instr),
    Opcode.OP_RRX: _fmt_rrx,
    Opcode.OP_RSB_IMMEDIATE: lambda r, instr=0: _fmt_dp_imm(r, "rsb", instr),
    Opcode.OP_RSB_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "rsb", instr),
    Opcode.OP_SADD16: lambda r: _fmt_simd3(r, "sadd16"),
    Opcode.OP_SADD8: lambda r: _fmt_simd3(r, "sadd8"),
    Opcode.OP_SASX: lambda r: _fmt_simd3(r, "sasx"),
    Opcode.OP_SBC_IMMEDIATE: lambda r, instr=0: _fmt_dp_imm(r, "sbc", instr),
    Opcode.OP_SBC_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "sbc", instr),
    Opcode.OP_SBFX: _fmt_sbfx,
    Opcode.OP_SDIV: _fmt_sdiv,
    Opcode.OP_SEL: _fmt_sel,
    Opcode.OP_SEV: lambda r: _fmt_noargs(r, "sev"),
    Opcode.OP_SHADD16: lambda r: _fmt_simd3(r, "shadd16"),
    Opcode.OP_SHADD8: lambda r: _fmt_simd3(r, "shadd8"),
    Opcode.OP_SHASX: lambda r: _fmt_simd3(r, "shasx"),
    Opcode.OP_SHSAX: lambda r: _fmt_simd3(r, "shsax"),
    Opcode.OP_SHSUB16: lambda r: _fmt_simd3(r, "shsub16"),
    Opcode.OP_SHSUB8: lambda r: _fmt_simd3(r, "shsub8"),
    Opcode.OP_SMLABB_SMLABT_SMLATB_SMLATT: _fmt_smlabb_variants,
    Opcode.OP_SMLAD_SMLADX: _fmt_smlad_variants,
    Opcode.OP_SMLAL: lambda r: _fmt_smlal(r, getattr(r, "setflags", False)),
    Opcode.OP_SMLALBB_SMLALBT_SMLALTB_SMLALTT: _fmt_smlalbb_variants,
    Opcode.OP_SMLALD_SMLALDX: _fmt_smlald_variants,
    Opcode.OP_SMLAWB_SMLAWT: _fmt_smlaw_variants,
    Opcode.OP_SMLSD_SMLSDX: _fmt_smlsd_variants,
    Opcode.OP_SMLSLD_SMLSLDX: _fmt_smlsld_variants,
    Opcode.OP_SMMLA_SMMLAR: _fmt_smmla_variants,
    Opcode.OP_SMMLS_SMMLSR: _fmt_smmls_variants,
    Opcode.OP_SMMUL_SMMULR: _fmt_smmul_variants,
    Opcode.OP_SMUAD_SMUADX: _fmt_smuad_variants,
    Opcode.OP_SMULBB_SMULBT_SMULTB_SMULTT: _fmt_smulbb_variants,
    Opcode.OP_SMULL: lambda r: _fmt_smull(r, getattr(r, "setflags", False)),
    Opcode.OP_SMULWB_SMULWT: _fmt_smulw_variants,
    Opcode.OP_SMUSD_SMUSDX: _fmt_smusd_variants,
    Opcode.OP_SSAT: _fmt_ssat,
    Opcode.OP_SSAT16: _fmt_ssat16,
    Opcode.OP_SSAX: lambda r: _fmt_simd3(r, "ssax"),
    Opcode.OP_SSBB: lambda r: _fmt_noargs(r, "ssbb"),
    Opcode.OP_SSUB16: lambda r: _fmt_simd3(r, "ssub16"),
    Opcode.OP_SSUB8: lambda r: _fmt_simd3(r, "ssub8"),
    Opcode.OP_STC_STC2: _fmt_stc_stc2,
    Opcode.OP_STM: lambda r: _fmt_multi_xfer(r, "stmia"),
    Opcode.OP_STMDB: lambda r: _fmt_multi_xfer(r, "stmdb"),
    Opcode.OP_STR_IMMEDIATE: lambda r, offset=0: _fmt_ldst_imm_t(
        r, "str", offset=offset
    ),
    Opcode.OP_STR_REGISTER: lambda r: _fmt_ldst_reg(r, "str"),
    Opcode.OP_STRB_IMMEDIATE: lambda r, offset=0: _fmt_ldst_imm_t(
        r, "strb", offset=offset
    ),
    Opcode.OP_STRB_REGISTER: lambda r: _fmt_ldst_reg(r, "strb"),
    Opcode.OP_STRBT: lambda r: _fmt_unpriv_str(r, "strbt"),
    Opcode.OP_STRD_IMMEDIATE: lambda r: _fmt_ldst_imm_dual(r, "strd"),
    Opcode.OP_STREX: _fmt_strex,
    Opcode.OP_STREXB: _fmt_strexb,
    Opcode.OP_STREXH: _fmt_strexh,
    Opcode.OP_STRH_IMMEDIATE: lambda r, offset=0: _fmt_ldst_imm_t(
        r, "strh", offset=offset
    ),
    Opcode.OP_STRH_REGISTER: lambda r: _fmt_ldst_reg(r, "strh"),
    Opcode.OP_STRHT: lambda r: _fmt_unpriv_str(r, "strht"),
    Opcode.OP_STRT: lambda r: _fmt_unpriv_str(r, "strt"),
    Opcode.OP_SUB_IMMEDIATE: lambda r, instr=0: _fmt_dp_imm(r, "sub", instr),
    Opcode.OP_SUB_REGISTER: lambda r, instr=0: _fmt_dp_reg(r, "sub", instr),
    Opcode.OP_SUB_SP_MINUS_IMMEDIATE: _fmt_sub_sp_imm,
    Opcode.OP_SUB_SP_MINUS_REGISTER: _fmt_sub_sp_reg,
    Opcode.OP_SVC: _fmt_svc,
    Opcode.OP_SXTAB: lambda r: _fmt_extab(r, "sxtab"),
    Opcode.OP_SXTAB16: lambda r: _fmt_extab(r, "sxtab16"),
    Opcode.OP_SXTAH: lambda r: _fmt_extab(r, "sxtah"),
    Opcode.OP_SXTB: lambda r: _fmt_extend(r, "sxtb"),
    Opcode.OP_SXTB16: lambda r: _fmt_extend(r, "sxtb16"),
    Opcode.OP_SXTH: lambda r: _fmt_extend(r, "sxth"),
    Opcode.OP_TBB_TBH: _fmt_tbb_tbh,
    Opcode.OP_TEQ_IMMEDIATE: lambda r: _fmt_test_imm(r, "teq"),
    Opcode.OP_TEQ_REGISTER: lambda r: _fmt_test_reg(r, "teq"),
    Opcode.OP_TST_IMMEDIATE: lambda r: _fmt_test_imm(r, "tst"),
    Opcode.OP_TST_REGISTER: lambda r: _fmt_test_reg(r, "tst"),
    Opcode.OP_UADD16: lambda r: _fmt_simd3(r, "uadd16"),
    Opcode.OP_UADD8: lambda r: _fmt_simd3(r, "uadd8"),
    Opcode.OP_UASX: lambda r: _fmt_simd3(r, "uasx"),
    Opcode.OP_UBFX: _fmt_ubfx,
    Opcode.OP_UDF: _fmt_udf,
    Opcode.OP_UDIV: _fmt_udiv,
    Opcode.OP_UHADD16: lambda r: _fmt_simd3(r, "uhadd16"),
    Opcode.OP_UHADD8: lambda r: _fmt_simd3(r, "uhadd8"),
    Opcode.OP_UHASX: lambda r: _fmt_simd3(r, "uhasx"),
    Opcode.OP_UHSAX: lambda r: _fmt_simd3(r, "uhsax"),
    Opcode.OP_UHSUB16: lambda r: _fmt_simd3(r, "uhsub16"),
    Opcode.OP_UHSUB8: lambda r: _fmt_simd3(r, "uhsub8"),
    Opcode.OP_UMAAL: _fmt_umaal,
    Opcode.OP_UMLAL: lambda r: _fmt_umlal(r, getattr(r, "setflags", False)),
    Opcode.OP_UMULL: lambda r: _fmt_umull(r, getattr(r, "setflags", False)),
    Opcode.OP_UQADD16: lambda r: _fmt_simd3(r, "uqadd16"),
    Opcode.OP_UQADD8: lambda r: _fmt_simd3(r, "uqadd8"),
    Opcode.OP_UQASX: lambda r: _fmt_simd3(r, "uqasx"),
    Opcode.OP_UQSAX: lambda r: _fmt_simd3(r, "uqsax"),
    Opcode.OP_UQSUB16: lambda r: _fmt_simd3(r, "uqsub16"),
    Opcode.OP_UQSUB8: lambda r: _fmt_simd3(r, "uqsub8"),
    Opcode.OP_USAD8: _fmt_usad8,
    Opcode.OP_USADA8: _fmt_usada8,
    Opcode.OP_USAT: _fmt_usat,
    Opcode.OP_USAT16: _fmt_usat16,
    Opcode.OP_USAX: lambda r: _fmt_simd3(r, "usax"),
    Opcode.OP_USUB16: lambda r: _fmt_simd3(r, "usub16"),
    Opcode.OP_USUB8: lambda r: _fmt_simd3(r, "usub8"),
    Opcode.OP_UXTAB: lambda r: _fmt_extab(r, "uxtab"),
    Opcode.OP_UXTAB16: lambda r: _fmt_extab(r, "uxtab16"),
    Opcode.OP_UXTAH: lambda r: _fmt_extab(r, "uxtah"),
    Opcode.OP_UXTB: lambda r: _fmt_extend(r, "uxtb"),
    Opcode.OP_UXTB16: lambda r: _fmt_extend(r, "uxtb16"),
    Opcode.OP_UXTH: lambda r: _fmt_extend(r, "uxth"),
    Opcode.OP_VABS: lambda r: _fmt_vfp_dp2(r, "vabs"),
    Opcode.OP_VADD: lambda r: _fmt_vfp_dp3(r, "vadd"),
    Opcode.OP_VCMP_VCMPE: _fmt_vcmp_vcmpe,
    Opcode.OP_VCVTA_VCVTN_VCVTP_VCVTM: _fmt_vcvta_round,
    Opcode.OP_VCVT_VCVTR_INTEGER: _fmt_vcvt_integer,
    Opcode.OP_VCVT_FIXED_POINT: _fmt_vcvt_fixed,
    Opcode.OP_VCVT_DOUBLE_SINGLE: _fmt_vcvt_double_single,
    Opcode.OP_VCVTB_VCVTT: _fmt_vcvtb_vcvtt,
    Opcode.OP_VDIV: lambda r: _fmt_vfp_dp3(r, "vdiv"),
    Opcode.OP_VFMA_VFMS: _fmt_vfma_vfms,
    Opcode.OP_VFNMA_VFNMS: _fmt_vfnma_vfnms,
    Opcode.OP_VLDM: lambda r: _fmt_vldm_vstm(r, "vldm"),
    Opcode.OP_VLDR: lambda r: _fmt_vldr_vstr(r, "vldr"),
    Opcode.OP_VMAXNM_VMINNM: _fmt_vmaxnm_vminnm,
    Opcode.OP_VMLA_VMLS: _fmt_vmla_vmls,
    Opcode.OP_VMOV_IMMEDIATE: _fmt_vmov_imm,
    Opcode.OP_VMOV_REGISTER: _fmt_vmov_reg,
    Opcode.OP_VMOV_CORE_TO_SCALAR: _fmt_vmov_core_to_scalar,
    Opcode.OP_VMOV_SCALAR_TO_CORE: _fmt_vmov_scalar_to_core,
    Opcode.OP_VMOV_CORE_AND_SINGLE: _fmt_vmov_core_and_single,
    Opcode.OP_VMOV_TWO_CORE_AND_TWO_SINGLE: _fmt_vmov_two_core_and_single,
    Opcode.OP_VMOV_TWO_CORE_AND_DOUBLEWORD: _fmt_vmov_two_core_and_doubleword,
    Opcode.OP_VMRS: _fmt_vmrs,
    Opcode.OP_VMSR: _fmt_vmsr,
    Opcode.OP_VMUL: lambda r: _fmt_vfp_dp3(r, "vmul"),
    Opcode.OP_VNEG: lambda r: _fmt_vfp_dp2(r, "vneg"),
    Opcode.OP_VNMLA_VNMLS_VNMUL: _fmt_vnmla_vnmls_vnmul,
    Opcode.OP_VPOP: lambda r: _fmt_vpush_vpop(r, "vpop"),
    Opcode.OP_VPUSH: lambda r: _fmt_vpush_vpop(r, "vpush"),
    Opcode.OP_VRINTA_VRINTN_VRINTP_VRINTM: _fmt_vrint_round,
    Opcode.OP_VRINTX: _fmt_vrintx,
    Opcode.OP_VRINTZ_VRINTR: _fmt_vrint_zr,
    Opcode.OP_VSEL: _fmt_vsel,
    Opcode.OP_VSQRT: lambda r: _fmt_vfp_dp2(r, "vsqrt"),
    Opcode.OP_VSTM: lambda r: _fmt_vldm_vstm(r, "vstm"),
    Opcode.OP_VSTR: lambda r: _fmt_vldr_vstr(r, "vstr"),
    Opcode.OP_VSUB: lambda r: _fmt_vfp_dp3(r, "vsub"),
    Opcode.OP_WFE: lambda r: _fmt_noargs(r, "wfe"),
    Opcode.OP_WFI: lambda r: _fmt_noargs(r, "wfi"),
    Opcode.OP_YIELD: lambda r: _fmt_noargs(r, "yield"),
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _in_cond_block(asm: str, istate: int) -> str:
    """Re-spell `asm` as the conditional instruction an IT block makes of it.

    The condition goes between the mnemonic and any dotted qualifier it
    carries -- `add` becomes `addeq`, `adds.w` `addseq.w`, `vmov.f32`
    `vmoveq.f32`. `_SEP` bounds the mnemonic, so the operands are never
    searched.
    """
    if not in_it_block(istate):
        return asm
    cond = current_cond(istate)
    if cond > COND_AL:
        return asm  # UNPREDICTABLE ITSTATE; nothing sensible to spell
    mnemonic, sep, operands = asm.partition(_SEP)
    base, dot, qualifier = mnemonic.partition(".")
    return f"{base}{_COND_CODES[cond]}{dot}{qualifier}{sep}{operands}"


def disassemble(
    result: object, instr: int = 0, offset: int = 0, istate: int = 0
) -> str:
    """Convert a decoded instruction dataclass to an assembler mnemonic string.

    Args:
        result: Decoded instruction dataclass instance (or pseudo-instruction).
        instr: Raw 32-bit instruction word (needed to disambiguate coprocessor
               variants like MCR vs MCR2).
        offset: Address of the instruction in memory (for computing absolute
                branch targets).
        istate: ITSTATE in force for this instruction, as tracked across the
                stream by :func:`armv7m_decoder.next_itstate`. Zero (the
                default) means no IT block is open, so no condition suffix.

    Returns:
        UAL assembler syntax string, e.g. ``"adds r0, r1, #42"``.
    """
    opc = result.opcode
    if opc == Opcode.OP_NO_MATCH:
        return "<nomatch>"
    if opc == Opcode.OP_UNDEFINED:
        return "<undefined>"
    if opc == Opcode.OP_UNPREDICTABLE:
        return "<unpredictable>"
    if opc == Opcode.OP_SEE:
        return "<see>"

    fmt_func = _DISPATCH.get(opc)

    if fmt_func is None:
        return repr(result)

    try:
        sig = _inspect.signature(fmt_func)
        kwargs: dict[str, Any] = {}
        if "offset" in sig.parameters:
            kwargs["offset"] = offset
        if "instr" in sig.parameters:
            kwargs["instr"] = instr
        asm = fmt_func(result, **kwargs)
    except (ValueError, TypeError):
        asm = fmt_func(result)

    # B (T1/T3) and VSEL carry their own condition, which is what the
    # architecture reports as CurrentCond -- they never take the block's. (Both
    # are UNPREDICTABLE in a block, so the decoder rarely lets one through.)
    if getattr(result, "cond", COND_AL) == COND_AL:
        asm = _in_cond_block(asm, istate)
    return asm

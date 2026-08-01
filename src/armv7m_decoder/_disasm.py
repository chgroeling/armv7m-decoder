"""ARMv7-M instruction disassembler.

Converts decoded instruction dataclass instances to UAL (Unified Assembler
Language) syntax strings, suitable for display as disassembled output.
"""

from __future__ import annotations

from typing import Any

from armv7m_decoder._decoder import InstructionSize, Opcode
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
_MNEMONICS_WITH_BOTH_WIDTHS: frozenset[str] = frozenset(
    {
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
    }
)
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


def _width(size: int, mnemonic: str) -> str:
    """Width suffix for `mnemonic`, appended after any `s`/condition suffix.

    A 32-bit encoding takes `.w` only where the same mnemonic also has a
    16-bit encoding and the suffix is what tells the two apart. Mnemonics
    that exist in one width only (`ubfx`, `stmdb`, `teq`, ...) take nothing.
    """
    if mnemonic in _MNEMONICS_WITH_BOTH_WIDTHS and size == InstructionSize.SIZE_32BIT:
        return ".w"
    return ""


def _is_narrow(size: int) -> bool:
    """Whether the instruction was decoded from a 16-bit encoding."""
    return size == InstructionSize.SIZE_16BIT


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


# How a transfer renders its immediate. The core single-word forms (LDR/STR)
# drop the sign of a zero offset and gloss only the plain positive offset --
# the encodings holding an imm12 -- saying nothing about an imm8 however it is
# indexed. The dual (LDRD/STRD) and coprocessor-class forms (LDC/STC,
# VLDR/VSTR) keep `#-0` and gloss either sign, the coprocessor ones showing a
# subtracted offset as the 32-bit value it adds.
_XFER_CORE, _XFER_DUAL, _XFER_COPROC = range(3)


def _xfer_comment(result: Any, style: int) -> str:
    """The `@ 0x..` gloss on a transfer's immediate, where objdump prints one."""
    if result.n == 15:
        return ""  # a literal is glossed with its target instead
    if style == _XFER_CORE and not (result.index and result.add and not result.wback):
        return ""
    if style == _XFER_COPROC and not result.add:
        return _hex_comment_signed(-result.imm32)
    return _hex_comment(result.imm32)


def _offset_text(
    imm32: int, add: bool, *, spell_zero: bool, negative_zero: bool
) -> str:
    """The `, #imm` of an addressing mode, or "" where it goes unwritten.

    A zero offset is left out of the wide offset and pre-indexed forms --
    `[ip]` -- because nothing else can be meant. It survives where it still
    says something: `spell_zero` for the 16-bit forms and the post-indexed
    ones, where the offset is what advances Rn, and `negative_zero` for the
    coprocessor, VFP and dual transfers, whose syntax carries the U bit into
    `[r0, #-0]` and would otherwise lose it.
    """
    if imm32 != 0 or spell_zero:
        return f", #{'' if add else '-'}{imm32}"
    return "" if add or not negative_zero else ", #-0"


def _addr_imm(result: Any, style: int = _XFER_CORE, size: int = 0) -> str:
    """`[Rn, #imm]` in the shape the encoding calls for."""
    n, imm32 = result.n, result.imm32
    sign = "" if result.add else "-"
    hc = _xfer_comment(result, style)
    if not result.index:
        if style == _XFER_COPROC:
            post = _offset_text(imm32, result.add, spell_zero=False, negative_zero=True)
            return f"[{_reg(n)}]{post}{hc}"
        return f"[{_reg(n)}], #{sign}{imm32}{hc}"
    offset_text = _offset_text(
        imm32,
        result.add,
        spell_zero=_is_narrow(size),
        negative_zero=style != _XFER_CORE,
    )
    wb = "!" if result.wback else ""
    return f"[{_reg(n)}{offset_text}]{wb}{hc}"


def _addr_imm_dual(result: Any, size: int = 0) -> str:
    addr = _addr_imm(result, _XFER_DUAL, size=size)
    return f"{_reg(result.t)}, {_reg(result.t2)}, {addr}"


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


def _addr_literal(result: Any, size: int = 0) -> str:
    offset_text = _offset_text(
        result.imm32, result.add, spell_zero=_is_narrow(size), negative_zero=False
    )
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


def _hex_comment_signed(value: int) -> str:
    """`_hex_comment` for a value objdump glosses as a 32-bit word."""
    if not (value > 32 or value < -16):
        return ""
    return f"\t@ 0x{value & 0xFFFFFFFF:x}"


def _hex_comment(imm: int) -> str:
    return f"\t@ 0x{imm:x}" if imm > 32 or imm < -16 else ""


def _hex_target(offset: int, imm32: int, add: bool = True, narrow: bool = True) -> str:
    base = ((offset + 4) & ~3) & 0xFFFFFFFF
    target = (base + (imm32 if add else -imm32)) & 0xFFFFFFFF
    # Only the narrow PC-relative loads have their target parenthesised.
    return f"\t@ (0x{target:x})" if narrow else f"\t@ 0x{target:x}"


# ---------------------------------------------------------------------------
# Combined-mnemonic helpers
# ---------------------------------------------------------------------------


def _is_coproc2(instr: int) -> bool:
    return bool(instr & 0x10000000)


def _is_rdn_encoding(instr: int, size: int = 0) -> bool:
    """Whether the encoding names its destination once, as `<Rdn>`.

    Three 16-bit groups do, and write two operands where the wide forms
    write three: ADD/SUB (immediate) T2 (0x3000), the data-processing group
    (0x4000) and ADD (register) T2 (0x4400). Nothing else does -- `d == n`
    is not the test, because the 16-bit T1 forms take three registers and
    two of them may well be the same one: 18ad is `adds r5, r5, r2`.

    Which of an instruction's encodings was matched is not in the decoded
    result, so this needs the instruction word; without one the wide,
    always-valid form is used.
    """
    if not _is_narrow(size):
        return False
    return 0x3000 <= instr < 0x4700


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


def _fmt_dp_imm(result: Any, mnemonic: str, instr: int = 0, size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, mnemonic)
    if _is_rdn_encoding(instr, size):
        return (
            f"{mnemonic}{s}{_SEP}{_reg(result.d)}, #{result.imm32}"
            f"{_hex_comment(result.imm32)}"
        )
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.imm32}"
        f"{_hex_comment(result.imm32)}"
    )


def _fmt_mov_imm(result: Any, instr: int = 0, size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, "mov")
    # T2 (mov.w) and T3 (movw) are both 32-bit, so the width alone cannot tell
    # them apart -- bits 25:20 pick out T3, which only a 32-bit word has.
    if size == InstructionSize.SIZE_32BIT and (instr >> 20) & 0x3F == 0x24:
        return (
            f"movw{_SEP}{_reg(result.d)}, #{result.imm32}{_hex_comment(result.imm32)}"
        )
    return f"mov{s}{_SEP}{_reg(result.d)}, #{result.imm32}{_hex_comment(result.imm32)}"


def _fmt_rsb_imm(result: Any, instr: int = 0, size: int = 0, **_) -> str:
    # T1 subtracts from a literal zero it has no room to encode, and is spelled
    # as the negate it performs: "negs r2, r2".
    if _is_narrow(size):
        return f"neg{_flags(result.setflags)}{_SEP}{_reg(result.d)}, {_reg(result.n)}"
    return _fmt_dp_imm(result, "rsb", instr, size=size)


def _fmt_mvn_imm(result: Any, mnemonic: str = "mvn", size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, mnemonic)
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)},"
        f" #{result.imm32}{_hex_comment(result.imm32)}"
    )


def _fmt_and_imm(result: Any, mnemonic: str = "and", size: int = 0, **_) -> str:
    s = _flags(getattr(result, "setflags", False)) + _width(size, mnemonic)
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.imm32}"
        f"{_hex_comment(result.imm32)}"
    )


# --- Data-processing register ---


def _fmt_dp_reg(result: Any, mnemonic: str, instr: int = 0, size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, mnemonic)
    sh = _shift(result.shift_t, result.shift_n)
    if _is_rdn_encoding(instr, size):
        return f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}{sh}"
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}{sh}"
    )


def _fmt_mov_reg(result: Any, size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, "mov")
    return f"mov{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_mvn_reg(result: Any, size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, "mvn")
    sh = _shift(result.shift_t, result.shift_n)
    return f"mvn{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}{sh}"


def _fmt_rrx(result: Any, size: int = 0, **_) -> str:
    # RRX is a ROR by one bit through carry, and 32-bit only, so it too is
    # spelled as the MOV that encodes it.
    return _fmt_mov_shifted(result, "rrx", size=size)


# --- Test/compare ---


def _fmt_test_imm(result: Any, mnemonic: str, size: int = 0, **_) -> str:
    w = _width(size, mnemonic)
    return (
        f"{mnemonic}{w}{_SEP}{_reg(result.n)},"
        f" #{result.imm32}{_hex_comment(result.imm32)}"
    )


def _fmt_test_reg(result: Any, mnemonic: str, size: int = 0, **_) -> str:
    sh = _shift(result.shift_t, result.shift_n)
    m = mnemonic + _width(size, mnemonic)
    return f"{m}{_SEP}{_reg(result.n)}, {_reg(result.m)}{sh}"


# --- Shift immediate ---


def _fmt_shift_imm(result: Any, mnemonic: str, size: int = 0, **_) -> str:
    # A wide shift by an immediate is MOV (register) T3 with its shift filled
    # in -- one encoding, and objdump spells it that way: "mov.w r7, r7,
    # lsl #2". Only the narrow encodings are shifts in their own right.
    if not _is_narrow(size):
        return _fmt_mov_shifted(result, f"{mnemonic} #{result.shift_n}", size=size)
    s = _flags(result.setflags) + _width(size, mnemonic)
    return f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}, #{result.shift_n}"


def _fmt_mov_shifted(result: Any, shift: str, size: int = 0) -> str:
    """MOV (register) T3 carrying `shift`, e.g. `lsl #2` or `rrx`."""
    s = _flags(result.setflags) + _width(size, "mov")
    return f"mov{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}, {shift}"


def _fmt_ror_imm(result: Any, size: int = 0, **_) -> str:
    return _fmt_shift_imm(result, "ror", size=size)


# --- Shift register ---


def _fmt_shift_reg(
    result: Any, mnemonic: str, instr: int = 0, size: int = 0, **_
) -> str:
    s = _flags(result.setflags) + _width(size, mnemonic)
    if _is_rdn_encoding(instr, size):
        return f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.m)}"
    return f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# --- SP arithmetic ---


# Only the 16-bit encodings of SP arithmetic leave SP out of the operands
# ("add sp, #4", "add r0, sp"); the wide forms spell it as the second operand
# even when that repeats the destination.


def _fmt_add_sp_imm(result: Any, size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, "add")
    if result.d == 13 and _is_narrow(size):
        return f"add{s}{_SEP}sp, #{result.imm32}{_hex_comment(result.imm32)}"
    return (
        f"add{s}{_SEP}{_reg(result.d)}, sp, #{result.imm32}{_hex_comment(result.imm32)}"
    )


def _fmt_add_sp_reg(result: Any, size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, "add")
    sh = _shift(result.shift_t, result.shift_n)
    if _is_narrow(size):
        if result.d == 13:
            return f"add{s}{_SEP}sp, {_reg(result.m)}{sh}"
        if result.d == result.m:
            return f"add{s}{_SEP}{_reg(result.d)}, sp{sh}"
    return f"add{s}{_SEP}{_reg(result.d)}, sp, {_reg(result.m)}{sh}"


def _fmt_sub_sp_imm(result: Any, size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, "sub")
    if result.d == 13 and _is_narrow(size):
        return f"sub{s}{_SEP}sp, #{result.imm32}{_hex_comment(result.imm32)}"
    return (
        f"sub{s}{_SEP}{_reg(result.d)}, sp, #{result.imm32}{_hex_comment(result.imm32)}"
    )


def _fmt_sub_sp_reg(result: Any, size: int = 0, **_) -> str:
    s = _flags(result.setflags) + _width(size, "sub")
    sh = _shift(result.shift_t, result.shift_n)
    if _is_narrow(size):
        if result.d == 13:
            return f"sub{s}{_SEP}sp, {_reg(result.m)}{sh}"
        if result.d == result.m:
            return f"sub{s}{_SEP}{_reg(result.d)}, sp{sh}"
    return f"sub{s}{_SEP}{_reg(result.d)}, sp, {_reg(result.m)}{sh}"


# --- ADR ---


def _fmt_adr(result: Any, offset: int = 0, size: int = 0, **_) -> str:
    """ADR renders as the PC-relative add/sub it encodes, never as `adr`.

    The wide encodings are the unshifted-immediate `addw`/`subw` forms; the
    narrow one is `add Rd, pc, #imm` and carries the resolved target as a
    comment, since the immediate is relative to Align(PC, 4).
    """
    d = _reg(result.d)
    if size == InstructionSize.SIZE_32BIT:
        mnemonic = "addw" if result.add else "subw"
        return f"{mnemonic}{_SEP}{d}, pc, #{result.imm32}{_hex_comment(result.imm32)}"
    base = ((offset + 4) & ~3) & 0xFFFFFFFF
    target = (base + (result.imm32 if result.add else -result.imm32)) & 0xFFFFFFFF
    return f"add{_SEP}{d}, pc, #{result.imm32}\t@ (adr {d}, 0x{target:x})"


# --- Load/store immediate ---


def _fmt_ldst_imm(result: Any, mnemonic: str, size: int = 0) -> str:
    addr = _addr_imm(result, size=size)
    return f"{mnemonic}{_width(size, mnemonic)}{_SEP}{addr}"


def _fmt_ldst_imm_t(
    result: Any, mnemonic: str, offset: int = 0, size: int = 0, **_
) -> str:
    addr = _addr_imm(result, size=size)
    asm = f"{mnemonic}{_width(size, mnemonic)}{_SEP}{_reg(result.t)}, {addr}"
    if result.n == 15:
        asm += _hex_target(offset, result.imm32, result.add, _is_narrow(size))
    return asm


def _fmt_ldst_imm_dual(result: Any, mnemonic: str, size: int = 0, **_) -> str:
    addr = _addr_imm_dual(result, size=size)
    return f"{mnemonic}{_width(size, mnemonic)}{_SEP}{addr}"


# --- Load/store register ---


def _fmt_ldst_reg(result: Any, mnemonic: str, size: int = 0, **_) -> str:
    addr = _addr_reg(result.t, result.n, result.m, result.shift_t, result.shift_n)
    return f"{mnemonic}{_width(size, mnemonic)}{_SEP}{addr}"


def _fmt_ldst_reg_rt(result: Any, mnemonic: str, size: int = 0) -> str:
    addr = _addr_reg(result.t, result.n, result.m, result.shift_t, result.shift_n)
    return f"{mnemonic}{_width(size, mnemonic)}{_SEP}{addr}"


# --- Load/store literal ---


def _fmt_ldst_lit(
    result: Any, mnemonic: str, offset: int = 0, size: int = 0, **_
) -> str:
    m = mnemonic + _width(size, mnemonic)
    asm = f"{m}{_SEP}{_addr_literal(result, size=size)}"
    return f"{asm}{_hex_target(offset, result.imm32, result.add, _is_narrow(size))}"


def _fmt_ldst_lit_dual(
    result: Any, mnemonic: str = "ldrd", offset: int = 0, **_
) -> str:
    offset_text = _offset_text(
        result.imm32, result.add, spell_zero=False, negative_zero=True
    )
    return (
        f"{mnemonic}{_SEP}{_reg(result.t)}, {_reg(result.t2)},"
        f" [pc{offset_text}]{_hex_target(offset, result.imm32, result.add)}"
    )


# --- Preload (PLD / PLI) ---


def _fmt_pld_imm(result: Any, **_) -> str:
    sign = "" if result.add else "-"
    return (
        f"pld{_SEP}[{_reg(result.n)},"
        f" #{sign}{result.imm32}]{_hex_comment(result.imm32)}"
    )


def _fmt_pld_lit(result: Any, offset: int = 0, **_) -> str:
    sign = "" if result.add else "-"
    return (
        f"pld{_SEP}[pc, #{sign}{result.imm32}]"
        f"{_hex_target(offset, result.imm32, result.add)}"
    )


def _fmt_pld_reg(result: Any, **_) -> str:
    if result.shift_n == 0:
        return f"pld{_SEP}[{_reg(result.n)}, {_reg(result.m)}]"
    return f"pld{_SEP}[{_reg(result.n)}, {_reg(result.m)}, lsl #{result.shift_n}]"


def _fmt_pli_imm_lit(result: Any, **_) -> str:
    sign = "" if result.add else "-"
    return (
        f"pli{_SEP}[{_reg(result.n)},"
        f" #{sign}{result.imm32}]{_hex_comment(result.imm32)}"
    )


def _fmt_pli_reg(result: Any, **_) -> str:
    if result.shift_n == 0:
        return f"pli{_SEP}[{_reg(result.n)}, {_reg(result.m)}]"
    return f"pli{_SEP}[{_reg(result.n)}, {_reg(result.m)}, lsl #{result.shift_n}]"


# --- Exclusive load/store ---


def _fmt_strex(result: Any, **_) -> str:
    return f"strex{_SEP}{_addr_excl(result.d, result.t, result.n, result.imm32)}"


def _fmt_ldrex(result: Any, **_) -> str:
    return f"ldrex{_SEP}{_addr_excl_single(result.t, result.n, result.imm32)}"


def _fmt_strexb(result: Any, **_) -> str:
    return f"strexb{_SEP}{_reg(result.d)}, {_reg(result.t)}, [{_reg(result.n)}]"


def _fmt_ldrexb(result: Any, **_) -> str:
    return f"ldrexb{_SEP}{_reg(result.t)}, [{_reg(result.n)}]"


def _fmt_strexh(result: Any, **_) -> str:
    return f"strexh{_SEP}{_reg(result.d)}, {_reg(result.t)}, [{_reg(result.n)}]"


def _fmt_ldrexh(result: Any, **_) -> str:
    return f"ldrexh{_SEP}{_reg(result.t)}, [{_reg(result.n)}]"


# --- Unprivileged load/store ---


def _fmt_unpriv_ldr(result: Any, mnemonic: str, **_) -> str:
    if result.register_form:
        return (
            f"{mnemonic}{_SEP}{_reg(result.t)},"
            f" [{_reg(result.n)}], {_reg(result.imm32)}"
        )
    if result.imm32 == 0:
        return f"{mnemonic}{_SEP}{_reg(result.t)}, [{_reg(result.n)}]"
    # An imm8 offset, so no gloss -- as with the other core imm8 forms.
    return f"{mnemonic}{_SEP}{_reg(result.t)}, [{_reg(result.n)}, #{result.imm32}]"


def _fmt_unpriv_str(result: Any, mnemonic: str, **_) -> str:
    return _fmt_unpriv_ldr(result, mnemonic)


# --- Multiple transfer ---


def _fmt_multi_xfer(result: Any, mnemonic: str, size: int = 0, **_) -> str:
    wb = "!" if result.wback else ""
    m = mnemonic + _width(size, mnemonic)
    return f"{m}{_SEP}{_reg(result.n)}{wb}, {_reg_list(result.registers)}"


# --- Stack ---


def _fmt_pop(result: Any, size: int = 0, **_) -> str:
    # Only the 16-bit T1 form is spelled "pop"; the wide forms render as the
    # LDM/LDR they encode. UnalignedAllowed marks the single-register T3.
    if _is_narrow(size):
        return f"pop{_SEP}{_reg_list(result.registers)}"
    if result.UnalignedAllowed:
        return f"ldr{_width(size, 'ldr')}{_SEP}{_reg(result.t)}, [sp], #4"
    return f"ldmia{_width(size, 'ldmia')}{_SEP}sp!, {_reg_list(result.registers)}"


def _fmt_push(result: Any, size: int = 0, **_) -> str:
    if _is_narrow(size):
        return f"push{_SEP}{_reg_list(result.registers)}"
    if result.UnalignedAllowed:
        return f"str{_width(size, 'str')}{_SEP}{_reg(result.t)}, [sp, #-4]!"
    return f"stmdb{_SEP}sp!, {_reg_list(result.registers)}"


# --- Branch ---


def _fmt_b(result: Any, offset: int = 0, instr: int = 0, size: int = 0, **_) -> str:
    c = _cond(result.cond)
    # .n and .w occupy the same slot; B is the only mnemonic that spells both.
    if _is_narrow(size):
        width = ".n"
    else:
        width = _width(size, "b")
    return f"b{c}{width}{_SEP}{_branch_target(offset, result.imm32)}"


def _fmt_bl(result: Any, offset: int = 0, **_) -> str:
    return f"bl{_SEP}{_branch_target(offset, result.imm32)}"


def _fmt_blx_reg(result: Any, **_) -> str:
    return f"blx{_SEP}{_reg(result.m)}"


def _fmt_bx(result: Any, **_) -> str:
    return f"bx{_SEP}{_reg(result.m)}"


def _fmt_cbnz_cbz(result: Any, offset: int = 0, **_) -> str:
    mnemonic = "cbnz" if result.nonzero else "cbz"
    return f"{mnemonic}{_SEP}{_reg(result.n)}, {_branch_target(offset, result.imm32)}"


def _fmt_tbb_tbh(result: Any, offset: int = 0, **_) -> str:
    # TBH indexes a table of halfwords, so its index register is doubled --
    # the shift is part of the syntax, not an operand of its own.
    mnemonic = "tbh" if result.is_tbh else "tbb"
    shift = ", lsl #1" if result.is_tbh else ""
    return f"{mnemonic}{_SEP}[{_reg(result.n)}, {_reg(result.m)}{shift}]"


# --- Barrier ---


def _fmt_dmb(result: Any, **_) -> str:
    return f"dmb{_SEP}{_barrier(result.option)}"


def _fmt_dsb(result: Any, **_) -> str:
    if result.option == 0xC:
        return "dfb"
    return f"dsb{_SEP}{_barrier(result.option)}"


def _fmt_isb(result: Any, **_) -> str:
    # ISB names only the full-system option; everything else stays numeric.
    option = "sy" if result.option == 0xF else f"#{result.option}"
    return f"isb{_SEP}{option}"


# --- Bitfield ---


def _fmt_bfi(result: Any, **_) -> str:
    width = result.msbit - result.lsbit + 1
    return f"bfi{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.lsbit}, #{width}"


def _fmt_bfc(result: Any, **_) -> str:
    width = result.msbit - result.lsbit + 1
    return f"bfc{_SEP}{_reg(result.d)}, #{result.lsbit}, #{width}"


def _fmt_ubfx(result: Any, **_) -> str:
    width = result.widthminus1 + 1
    return f"ubfx{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.lsbit}, #{width}"


def _fmt_sbfx(result: Any, **_) -> str:
    width = result.widthminus1 + 1
    return f"sbfx{_SEP}{_reg(result.d)}, {_reg(result.n)}, #{result.lsbit}, #{width}"


# --- Extend ---


def _fmt_extend(result: Any, mnemonic: str, size: int = 0, **_) -> str:
    m = mnemonic + _width(size, mnemonic)
    if result.rotation == 0:
        return f"{m}{_SEP}{_reg(result.d)}, {_reg(result.m)}"
    rot = f"ror #{result.rotation}"
    return f"{m}{_SEP}{_reg(result.d)}, {_reg(result.m)}, {rot}"


def _fmt_extab(result: Any, mnemonic: str, **_) -> str:
    if result.rotation == 0:
        return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"
    rot = f"ror #{result.rotation}"
    return (
        f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {rot}"
    )


# --- Saturate ---


def _fmt_ssat(result: Any, **_) -> str:
    sh = _shift(result.shift_t, result.shift_n)
    return f"ssat{_SEP}{_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}{sh}"


def _fmt_usat(result: Any, **_) -> str:
    sh = _shift(result.shift_t, result.shift_n)
    return f"usat{_SEP}{_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}{sh}"


def _fmt_ssat16(result: Any, **_) -> str:
    return f"ssat16{_SEP}{_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}"


def _fmt_usat16(result: Any, **_) -> str:
    return f"usat16{_SEP}{_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}"


# --- SIMD 3-register ---


def _fmt_simd3(result: Any, mnemonic: str, **_) -> str:
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# --- SIMD 3-register + accumulator ---


def _fmt_smla(result: Any, mnemonic: str, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"{mnemonic}{s}{_SEP}{_reg(result.d)}, {_reg(result.n)},"
        f" {_reg(result.m)}, {_reg(result.a)}"
    )


def _fmt_smlal(result: Any, setflags: bool = False, **_) -> str:
    s = _flags(setflags)
    return (
        f"smlal{s}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


# --- SIMD long multiply ---


def _fmt_smull(result: Any, setflags: bool = False, **_) -> str:
    s = _flags(setflags)
    return (
        f"smull{s}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


def _fmt_umull(result: Any, setflags: bool = False, **_) -> str:
    s = _flags(setflags)
    return (
        f"umull{s}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


def _fmt_umlal(result: Any, setflags: bool = False, **_) -> str:
    s = _flags(setflags)
    return (
        f"umlal{s}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


def _fmt_umaal(result: Any, **_) -> str:
    return (
        f"umaal{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


# --- Multiply ---


def _fmt_mul(
    result: Any, setflags: bool = False, instr: int = 0, size: int = 0, **_
) -> str:
    s = _flags(setflags) + _width(size, "mul")
    # T1 spells the destination as <Rdm>, so it is the multiplicand that goes
    # unwritten here, not the multiplier the other Rdn encodings drop.
    if _is_rdn_encoding(instr, size):
        return f"mul{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}"
    return f"mul{s}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_mla(result: Any, setflags: bool = False, **_) -> str:
    s = _flags(setflags)
    return (
        f"mla{s}{_SEP}{_reg(result.d)}, {_reg(result.n)},"
        f" {_reg(result.m)}, {_reg(result.a)}"
    )


def _fmt_mls(result: Any, **_) -> str:
    return (
        f"mls{_SEP}{_reg(result.d)}, {_reg(result.n)},"
        f" {_reg(result.m)}, {_reg(result.a)}"
    )


def _fmt_sdiv(result: Any, **_) -> str:
    return f"sdiv{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_udiv(result: Any, **_) -> str:
    return f"udiv{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# --- Miscellaneous ---


def _fmt_clz(result: Any, **_) -> str:
    return f"clz{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_rev(result: Any, size: int = 0, **_) -> str:
    w = _width(size, "rev")
    return f"rev{w}{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_rev16(result: Any, size: int = 0, **_) -> str:
    w = _width(size, "rev16")
    return f"rev16{w}{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_revsh(result: Any, size: int = 0, **_) -> str:
    w = _width(size, "revsh")
    return f"revsh{w}{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_rbit(result: Any, **_) -> str:
    return f"rbit{_SEP}{_reg(result.d)}, {_reg(result.m)}"


def _fmt_sel(result: Any, **_) -> str:
    return f"sel{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_pkhbt_pkhtb(result: Any, **_) -> str:
    mnemonic = "pkhtb" if result.tbform else "pkhbt"
    sh = _shift(result.shift_t, result.shift_n)
    # When tbform is False, shift type is LSL; when True, ASR
    if mnemonic == "pkhbt" and result.shift_t == 0 and result.shift_n == 0:
        return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}{sh}"


# --- MOVT ---


def _fmt_movt(result: Any, **_) -> str:
    return f"movt{_SEP}{_reg(result.d)}, #{result.imm16}{_hex_comment(result.imm16)}"


# --- USAD8 / USADA8 ---


def _fmt_usad8(result: Any, **_) -> str:
    return f"usad8{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_usada8(result: Any, **_) -> str:
    return (
        f"usada8{_SEP}{_reg(result.d)}, {_reg(result.n)},"
        f" {_reg(result.m)}, {_reg(result.a)}"
    )


# --- System registers ---


def _fmt_mrs(result: Any, **_) -> str:
    spec = _SPEC_REGS.get(result.SYSm, f"spec_reg_{result.SYSm:#x}")
    return f"mrs{_SEP}{_reg(result.d)}, {spec}"


def _fmt_msr(result: Any, **_) -> str:
    spec = _SPEC_REGS.get(result.SYSm, f"spec_reg_{result.SYSm:#x}")
    return f"msr{_SEP}{spec}, {_reg(result.n)}"


# --- CPS ---


def _fmt_cps(result: Any, **_) -> str:
    effect = "ie" if result.enable else "id"
    flags = ""
    if result.affectPRI:
        flags += "i"
    if result.affectFAULT:
        flags += "f"
    return f"cps{effect}{_SEP}{flags}"


# --- IT ---


def _fmt_it(result: Any, **_) -> str:
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


def _fmt_noargs(result: Any, mnemonic: str, size: int = 0, **_) -> str:
    return f"{mnemonic}{_width(size, mnemonic)}"


def _fmt_bkpt(result: Any, **_) -> str:
    return f"bkpt{_SEP}0x{result.imm32:04x}"


def _fmt_svc(result: Any, **_) -> str:
    return f"svc{_SEP}{result.imm32}{_hex_comment(result.imm32)}"


def _fmt_udf(result: Any, size: int = 0, **_) -> str:
    w = _width(size, "udf")
    return f"udf{w}{_SEP}#{result.imm32}{_hex_comment(result.imm32)}"


def _fmt_dbg(result: Any, **_) -> str:
    return f"dbg{_SEP}#{result.option}"


# --- VFP data-processing 3-reg ---


def _fmt_vfp_dp3(result: Any, mnemonic: str, **_) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


# --- VFP data-processing 2-reg ---


def _fmt_vfp_dp2(result: Any, mnemonic: str, **_) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {m}"


# --- VFP combined mnemonics ---


def _fmt_vfma_vfms(result: Any, **_) -> str:
    mnemonic = "vfms" if result.op1_neg else "vfma"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


def _fmt_vfnma_vfnms(result: Any, **_) -> str:
    mnemonic = "vfnms" if result.op1_neg else "vfnma"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


def _fmt_vmla_vmls(result: Any, **_) -> str:
    mnemonic = "vmls" if not result.add else "vmla"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


def _fmt_vnmla_vnmls_vnmul(result: Any, **_) -> str:
    types = {0: "vnmla", 1: "vnmls", 2: "vnmul"}
    mnemonic = types.get(result.type, "vnmla")
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


# --- VFP compare ---


def _fmt_vcmp_vcmpe(result: Any, **_) -> str:
    mnemonic = "vcmpe" if result.quiet_nan_exc else "vcmp"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    if result.with_zero:
        return f"{mnemonic}{precision}{_SEP}{d}, #0.0"
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {m}"


# --- VFP convert ---

_VCVTA_MODES = {0: "vcvta", 1: "vcvtn", 2: "vcvtp", 3: "vcvtm"}


def _fmt_vcvta_round(result: Any, **_) -> str:
    mnemonic = _VCVTA_MODES.get(result.round_mode, "vcvta")
    signed = "u" if result.unsigned else "s"
    src_prec = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}.{signed}32{src_prec}{_SEP}{d}, {m}"


def _fmt_vcvt_integer(result: Any, **_) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    unsigned = "u" if result.unsigned else ""
    rounding = "r" if (result.round_zero or result.round_nearest) else ""
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    if result.to_integer:
        return f"vcvt{rounding}{unsigned}.s32{precision}{_SEP}{d}, {m}"
    return f"vcvt{rounding}{unsigned}{precision}.s32{_SEP}{d}, {m}"


def _fmt_vcvt_fixed(result: Any, **_) -> str:
    unsigned = "u" if result.unsigned else ""
    d = _sreg(result.d)
    if result.to_fixed:
        return f"vcvt{unsigned}.s32.f32{_SEP}{d}, {d}, #{result.frac_bits}"
    return f"vcvt{unsigned}.f32.s32{_SEP}{d}, {d}, #{result.frac_bits}"


def _fmt_vcvt_double_single(result: Any, **_) -> str:
    if result.double_to_single:
        return f"vcvt.f32.f64{_SEP}{_sreg(result.d)}, {_dreg(result.m)}"
    return f"vcvt.f64.f32{_SEP}{_dreg(result.d)}, {_sreg(result.m)}"


def _fmt_vcvtb_vcvtt(result: Any, **_) -> str:
    mnemonic = "vcvtt" if result.lowbit else "vcvtb"
    return f"{mnemonic}.f32.f16{_SEP}{_sreg(result.d)}, {_sreg(result.m)}"


# --- VFP move ---


def _fmt_vmov_imm(result: Any, **_) -> str:
    return (
        f"vmov{_SEP}{_vfp_reg(result.dp_operation, result.d)}, #{result.imm32}"
        f"{_hex_comment(result.imm32)}"
    )


def _fmt_vmov_reg(result: Any, **_) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"vmov{precision}{_SEP}{d}, {m}"


def _fmt_vmov_core_to_scalar(result: Any, **_) -> str:
    return f"vmov{_SEP}{_sreg(result.d)}, {_reg(result.t)}"


def _fmt_vmov_scalar_to_core(result: Any, **_) -> str:
    return f"vmov{_SEP}{_reg(result.t)}, {_sreg(result.n)}"


def _fmt_vmov_core_and_single(result: Any, **_) -> str:
    if result.to_arm_register:
        return f"vmov{_SEP}{_reg(result.t)}, {_sreg(result.n)}"
    return f"vmov{_SEP}{_sreg(result.n)}, {_reg(result.t)}"


def _fmt_vmov_two_core_and_single(result: Any, **_) -> str:
    if result.to_arm_registers:
        return (
            f"vmov{_SEP}{_reg(result.t)}, {_reg(result.t2)},"
            f" {_sreg(result.m)}, {_sreg(result.m + 1)}"
        )
    return (
        f"vmov{_SEP}{_sreg(result.m)}, {_sreg(result.m + 1)},"
        f" {_reg(result.t)}, {_reg(result.t2)}"
    )


def _fmt_vmov_two_core_and_doubleword(result: Any, **_) -> str:
    if result.to_arm_registers:
        return f"vmov{_SEP}{_reg(result.t)}, {_reg(result.t2)}, {_dreg(result.m)}"
    return f"vmov{_SEP}{_dreg(result.m)}, {_reg(result.t)}, {_reg(result.t2)}"


# --- VFP system ---


def _fmt_vmrs(result: Any, **_) -> str:
    if result.t == 15:
        return f"vmrs{_SEP}apsr_nzcv, fpscr"
    return f"vmrs{_SEP}{_reg(result.t)}, fpscr"


def _fmt_vmsr(result: Any, **_) -> str:
    return f"vmsr{_SEP}fpscr, {_reg(result.t)}"


# --- VFP conditional select ---


def _fmt_vsel(result: Any, **_) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    c = _cond(result.cond)
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"vsel{c}{precision}{_SEP}{d}, {n}, {m}"


# --- VFP round ---

_VRINT_MODES = {0: "vrinta", 1: "vrintn", 2: "vrintp", 3: "vrintm"}
_VRINTZ_MODES = {0: "vrintz", 1: "vrintr"}


def _fmt_vrint_round(result: Any, **_) -> str:
    mnemonic = _VRINT_MODES.get(result.rmode, "vrinta")
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {m}"


def _fmt_vrint_zr(result: Any, **_) -> str:
    mnemonic = _VRINTZ_MODES.get(result.rmode, "vrintz")
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {m}"


def _fmt_vrintx(result: Any, **_) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"vrintx{precision}{_SEP}{d}, {m}"


# --- VFP load/store ---


def _fmt_vldr_vstr(result: Any, mnemonic: str, offset: int = 0, **_) -> str:
    single_reg = result.single_reg
    reg_name_fn = _sreg if single_reg else _dreg
    # 32-bit only, and an offset form throughout.
    offset_text = _offset_text(
        result.imm32, result.add, spell_zero=False, negative_zero=True
    )
    if result.n == 15:
        # PC-relative: glossed with the address it loads from, like LDR.
        hc = _hex_target(offset, result.imm32, result.add, narrow=False)
    elif result.add:
        hc = _hex_comment(result.imm32)
    else:
        hc = _hex_comment_signed(-result.imm32)
    return (
        f"{mnemonic}{_SEP}{reg_name_fn(result.d)}, [{_reg(result.n)}{offset_text}]{hc}"
    )


def _fmt_vldm_vstm(result: Any, mnemonic: str, **_) -> str:
    wb = "!" if result.wback else ""
    reg_str = _vfp_reg_list(result.single_regs, result.d, result.regs)
    return f"{mnemonic}{_SEP}{_reg(result.n)}{wb}, {reg_str}"


def _fmt_vpush_vpop(result: Any, mnemonic: str, **_) -> str:
    reg_str = _vfp_reg_list(result.single_regs, result.d, result.regs)
    return f"{mnemonic}{_SEP}{reg_str}"


# --- VFP max/min ---


def _fmt_vmaxnm_vminnm(result: Any, **_) -> str:
    mnemonic = "vmaxnm" if result.maximum else "vminnm"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision}{_SEP}{d}, {n}, {m}"


# --- SIMD combined mnemonics ---


def _fmt_smlabb_variants(result: Any, **_) -> str:
    mnemonic = _nhigh_mhigh_mnemonic("smla", result.n_high, result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlalbb_variants(result: Any, **_) -> str:
    mnemonic = _nhigh_mhigh_mnemonic("smlal", result.n_high, result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)}, {_reg(result.n)}, {_reg(result.m)}"  # noqa: E501


def _fmt_smulbb_variants(result: Any, **_) -> str:
    mnemonic = _nhigh_mhigh_mnemonic("smul", result.n_high, result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smlad_variants(result: Any, **_) -> str:
    mnemonic = _mswap_mnemonic("smlad", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlald_variants(result: Any, **_) -> str:
    mnemonic = _mswap_mnemonic("smlald", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)}, {_reg(result.n)}, {_reg(result.m)}"  # noqa: E501


def _fmt_smlaw_variants(result: Any, **_) -> str:
    mnemonic = _mhigh_mnemonic("smlaw", result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlsd_variants(result: Any, **_) -> str:
    mnemonic = _mswap_mnemonic("smlsd", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlsld_variants(result: Any, **_) -> str:
    mnemonic = _mswap_mnemonic("smlsld", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.dLo)}, {_reg(result.dHi)}, {_reg(result.n)}, {_reg(result.m)}"  # noqa: E501


def _fmt_smmla_variants(result: Any, **_) -> str:
    mnemonic = _round_mnemonic("smmla", result.round)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smmls_variants(result: Any, **_) -> str:
    mnemonic = _round_mnemonic("smmls", result.round)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smmul_variants(result: Any, **_) -> str:
    mnemonic = _round_mnemonic("smmul", result.round)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smuad_variants(result: Any, **_) -> str:
    mnemonic = _mswap_mnemonic("smuad", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smulw_variants(result: Any, **_) -> str:
    mnemonic = _mhigh_mnemonic("smulw", result.m_high)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smusd_variants(result: Any, **_) -> str:
    mnemonic = _mswap_mnemonic("smusd", result.m_swap)
    return f"{mnemonic}{_SEP}{_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# ---------------------------------------------------------------------------
# Coprocessor formatters
# ---------------------------------------------------------------------------


# The coprocessor instructions spell their operands their own way: the
# coprocessor and its opcodes as bare numbers, its registers as crN, and the
# trailing opc2 in braces.


def _fmt_cdp_cdp2(result: Any, instr: int = 0, **_) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return (
        f"cdp{suffix}{_SEP}{result.cp}, {result.opc1}, cr{result.CRd},"
        f" cr{result.CRn}, cr{result.CRm}, {{{result.opc2}}}"
    )


def _fmt_mcr_mcr2(result: Any, instr: int = 0, **_) -> str:
    return _fmt_mcr_mrc(result, "mcr", instr)


def _fmt_mrc_mrc2(result: Any, instr: int = 0, **_) -> str:
    return _fmt_mcr_mrc(result, "mrc", instr)


def _fmt_mcr_mrc(result: Any, base: str, instr: int) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return (
        f"{base}{suffix}{_SEP}{result.cp}, {result.opc1}, {_reg(result.t)},"
        f" cr{result.CRn}, cr{result.CRm}, {{{result.opc2}}}"
    )


def _fmt_mcrr_mcrr2(result: Any, instr: int = 0, **_) -> str:
    return _fmt_mcrr_mrrc(result, "mcrr", instr)


def _fmt_mrrc_mrrc2(result: Any, instr: int = 0, **_) -> str:
    return _fmt_mcrr_mrrc(result, "mrrc", instr)


def _fmt_mcrr_mrrc(result: Any, base: str, instr: int) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return (
        f"{base}{suffix}{_SEP}{result.cp}, {result.opc1},"
        f" {_reg(result.t)}, {_reg(result.t2)}, cr{result.CRm}"
    )


def _coproc_mnemonic(base: str, result: Any, instr: int) -> str:
    """`ldc`/`stc` with the variant and the long bit it carries: `ldc2l`."""
    return f"{base}{'2' if _is_coproc2(instr) else ''}{'l' if result.D else ''}"


def _addr_coproc(result: Any, size: int = 0) -> str:
    # The unindexed form has no offset at all: its imm8 is an option code for
    # the coprocessor, written in braces and unscaled.
    if not result.index and not result.wback:
        return f"[{_reg(result.n)}], {{{result.imm32 >> 2}}}"
    return _addr_imm(result, _XFER_COPROC, size=size)


def _fmt_stc_stc2(result: Any, instr: int = 0, size: int = 0, **_) -> str:
    m = _coproc_mnemonic("stc", result, instr)
    return f"{m}{_SEP}{result.cp}, cr{result.CRd}, {_addr_coproc(result, size=size)}"


def _fmt_ldc_ldc2_imm(result: Any, instr: int = 0, size: int = 0, **_) -> str:
    m = _coproc_mnemonic("ldc", result, instr)
    return f"{m}{_SEP}{result.cp}, cr{result.CRd}, {_addr_coproc(result, size=size)}"


def _fmt_ldc_ldc2_lit(result: Any, instr: int = 0, offset: int = 0, **_) -> str:
    m = _coproc_mnemonic("ldc", result, instr)
    sign = "" if result.add else "-"
    target = _hex_target(offset, result.imm32, result.add, narrow=False)
    return f"{m}{_SEP}{result.cp}, cr{result.CRd}, [pc, #{sign}{result.imm32}]{target}"


# ---------------------------------------------------------------------------
# Dispatch table — maps an opcode to the formatter that spells it
# ---------------------------------------------------------------------------
#
# Every entry is called as `entry(result, instr=..., size=..., offset=...)`,
# so a formatter names the ones it uses and lets `**_` swallow the rest. The
# lambdas are there to bind a mnemonic where one formatter serves several
# opcodes; they pass the rest of the call straight through.

_DISPATCH: dict[int, Any] = {
    Opcode.OP_ADC_IMMEDIATE: lambda r, **kw: _fmt_dp_imm(r, "adc", **kw),
    Opcode.OP_ADC_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "adc", **kw),
    Opcode.OP_ADD_IMMEDIATE: lambda r, **kw: _fmt_dp_imm(r, "add", **kw),
    Opcode.OP_ADD_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "add", **kw),
    Opcode.OP_ADD_SP_PLUS_IMMEDIATE: _fmt_add_sp_imm,
    Opcode.OP_ADD_SP_PLUS_REGISTER: _fmt_add_sp_reg,
    Opcode.OP_ADR: _fmt_adr,
    Opcode.OP_AND_IMMEDIATE: _fmt_and_imm,
    Opcode.OP_AND_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "and", **kw),
    Opcode.OP_ASR_IMMEDIATE: lambda r, **kw: _fmt_shift_imm(r, "asr", **kw),
    Opcode.OP_ASR_REGISTER: lambda r, **kw: _fmt_shift_reg(r, "asr", **kw),
    Opcode.OP_B: _fmt_b,
    Opcode.OP_BFC: _fmt_bfc,
    Opcode.OP_BFI: _fmt_bfi,
    Opcode.OP_BIC_IMMEDIATE: lambda r, **kw: _fmt_and_imm(r, "bic", **kw),
    Opcode.OP_BIC_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "bic", **kw),
    Opcode.OP_BKPT: _fmt_bkpt,
    Opcode.OP_BL: _fmt_bl,
    Opcode.OP_BLX_REGISTER: _fmt_blx_reg,
    Opcode.OP_BX: _fmt_bx,
    Opcode.OP_CBNZ_CBZ: _fmt_cbnz_cbz,
    Opcode.OP_CDP_CDP2: _fmt_cdp_cdp2,
    Opcode.OP_CLREX: lambda r, **kw: _fmt_noargs(r, "clrex", **kw),
    Opcode.OP_CLZ: _fmt_clz,
    Opcode.OP_CMN_IMMEDIATE: lambda r, **kw: _fmt_test_imm(r, "cmn", **kw),
    Opcode.OP_CMN_REGISTER: lambda r, **kw: _fmt_test_reg(r, "cmn", **kw),
    Opcode.OP_CMP_IMMEDIATE: lambda r, **kw: _fmt_test_imm(r, "cmp", **kw),
    Opcode.OP_CMP_REGISTER: lambda r, **kw: _fmt_test_reg(r, "cmp", **kw),
    Opcode.OP_CPS: _fmt_cps,
    Opcode.OP_CSDB: lambda r, **kw: _fmt_noargs(r, "csdb", **kw),
    Opcode.OP_DBG: _fmt_dbg,
    Opcode.OP_DMB: _fmt_dmb,
    Opcode.OP_DSB: _fmt_dsb,
    Opcode.OP_EOR_IMMEDIATE: lambda r, **kw: _fmt_and_imm(r, "eor", **kw),
    Opcode.OP_EOR_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "eor", **kw),
    Opcode.OP_ISB: _fmt_isb,
    Opcode.OP_IT: _fmt_it,
    Opcode.OP_LDC_LDC2_IMMEDIATE: _fmt_ldc_ldc2_imm,
    Opcode.OP_LDC_LDC2_LITERAL: _fmt_ldc_ldc2_lit,
    Opcode.OP_LDM: lambda r, **kw: _fmt_multi_xfer(r, "ldmia", **kw),
    Opcode.OP_LDMDB: lambda r, **kw: _fmt_multi_xfer(r, "ldmdb", **kw),
    Opcode.OP_LDR_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_t(r, "ldr", **kw),
    Opcode.OP_LDR_LITERAL: lambda r, **kw: _fmt_ldst_lit(r, "ldr", **kw),
    Opcode.OP_LDR_REGISTER: lambda r, **kw: _fmt_ldst_reg(r, "ldr", **kw),
    Opcode.OP_LDRB_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_t(r, "ldrb", **kw),
    Opcode.OP_LDRB_LITERAL: lambda r, **kw: _fmt_ldst_lit(r, "ldrb", **kw),
    Opcode.OP_LDRB_REGISTER: lambda r, **kw: _fmt_ldst_reg(r, "ldrb", **kw),
    Opcode.OP_LDRBT: lambda r, **kw: _fmt_unpriv_ldr(r, "ldrbt", **kw),
    Opcode.OP_LDRD_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_dual(r, "ldrd", **kw),
    Opcode.OP_LDRD_LITERAL: _fmt_ldst_lit_dual,
    Opcode.OP_LDREX: _fmt_ldrex,
    Opcode.OP_LDREXB: _fmt_ldrexb,
    Opcode.OP_LDREXH: _fmt_ldrexh,
    Opcode.OP_LDRH_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_t(r, "ldrh", **kw),
    Opcode.OP_LDRH_LITERAL: lambda r, **kw: _fmt_ldst_lit(r, "ldrh", **kw),
    Opcode.OP_LDRH_REGISTER: lambda r, **kw: _fmt_ldst_reg(r, "ldrh", **kw),
    Opcode.OP_LDRHT: lambda r, **kw: _fmt_unpriv_ldr(r, "ldrht", **kw),
    Opcode.OP_LDRSB_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_t(r, "ldrsb", **kw),
    Opcode.OP_LDRSB_LITERAL: lambda r, **kw: _fmt_ldst_lit(r, "ldrsb", **kw),
    Opcode.OP_LDRSB_REGISTER: lambda r, **kw: _fmt_ldst_reg(r, "ldrsb", **kw),
    Opcode.OP_LDRSBT: lambda r, **kw: _fmt_unpriv_ldr(r, "ldrsbt", **kw),
    Opcode.OP_LDRSH_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_t(r, "ldrsh", **kw),
    Opcode.OP_LDRSH_LITERAL: lambda r, **kw: _fmt_ldst_lit(r, "ldrsh", **kw),
    Opcode.OP_LDRSH_REGISTER: lambda r, **kw: _fmt_ldst_reg(r, "ldrsh", **kw),
    Opcode.OP_LDRSHT: lambda r, **kw: _fmt_unpriv_ldr(r, "ldrsht", **kw),
    Opcode.OP_LDRT: lambda r, **kw: _fmt_unpriv_ldr(r, "ldrt", **kw),
    Opcode.OP_LSL_IMMEDIATE: lambda r, **kw: _fmt_shift_imm(r, "lsl", **kw),
    Opcode.OP_LSL_REGISTER: lambda r, **kw: _fmt_shift_reg(r, "lsl", **kw),
    Opcode.OP_LSR_IMMEDIATE: lambda r, **kw: _fmt_shift_imm(r, "lsr", **kw),
    Opcode.OP_LSR_REGISTER: lambda r, **kw: _fmt_shift_reg(r, "lsr", **kw),
    Opcode.OP_MCR_MCR2: _fmt_mcr_mcr2,
    Opcode.OP_MCRR_MCRR2: _fmt_mcrr_mcrr2,
    Opcode.OP_MLA: lambda r, **kw: _fmt_mla(r, getattr(r, "setflags", False), **kw),
    Opcode.OP_MLS: _fmt_mls,
    Opcode.OP_MOV_IMMEDIATE: _fmt_mov_imm,
    Opcode.OP_MOV_REGISTER: _fmt_mov_reg,
    Opcode.OP_MOVT: _fmt_movt,
    Opcode.OP_MRC_MRC2: _fmt_mrc_mrc2,
    Opcode.OP_MRRC_MRRC2: _fmt_mrrc_mrrc2,
    Opcode.OP_MRS: _fmt_mrs,
    Opcode.OP_MSR: _fmt_msr,
    Opcode.OP_MUL: lambda r, **kw: _fmt_mul(r, getattr(r, "setflags", False), **kw),
    Opcode.OP_MVN_IMMEDIATE: _fmt_mvn_imm,
    Opcode.OP_MVN_REGISTER: _fmt_mvn_reg,
    Opcode.OP_NOP: lambda r, **kw: _fmt_noargs(r, "nop", **kw),
    Opcode.OP_ORN_IMMEDIATE: lambda r, **kw: _fmt_and_imm(r, "orn", **kw),
    Opcode.OP_ORN_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "orn", **kw),
    Opcode.OP_ORR_IMMEDIATE: lambda r, **kw: _fmt_and_imm(r, "orr", **kw),
    Opcode.OP_ORR_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "orr", **kw),
    Opcode.OP_PKHBT_PKHTB: _fmt_pkhbt_pkhtb,
    Opcode.OP_PLD_IMMEDIATE: _fmt_pld_imm,
    Opcode.OP_PLD_LITERAL: _fmt_pld_lit,
    Opcode.OP_PLD_REGISTER: _fmt_pld_reg,
    Opcode.OP_PLI_IMMEDIATE_LITERAL: _fmt_pli_imm_lit,
    Opcode.OP_PLI_REGISTER: _fmt_pli_reg,
    Opcode.OP_POP: _fmt_pop,
    Opcode.OP_PSSBB: lambda r, **kw: _fmt_noargs(r, "pssbb", **kw),
    Opcode.OP_PUSH: _fmt_push,
    Opcode.OP_QADD: lambda r, **kw: _fmt_simd3(r, "qadd", **kw),
    Opcode.OP_QADD16: lambda r, **kw: _fmt_simd3(r, "qadd16", **kw),
    Opcode.OP_QADD8: lambda r, **kw: _fmt_simd3(r, "qadd8", **kw),
    Opcode.OP_QASX: lambda r, **kw: _fmt_simd3(r, "qasx", **kw),
    Opcode.OP_QDADD: lambda r, **kw: _fmt_simd3(r, "qdadd", **kw),
    Opcode.OP_QDSUB: lambda r, **kw: _fmt_simd3(r, "qdsub", **kw),
    Opcode.OP_QSAX: lambda r, **kw: _fmt_simd3(r, "qsax", **kw),
    Opcode.OP_QSUB: lambda r, **kw: _fmt_simd3(r, "qsub", **kw),
    Opcode.OP_QSUB16: lambda r, **kw: _fmt_simd3(r, "qsub16", **kw),
    Opcode.OP_QSUB8: lambda r, **kw: _fmt_simd3(r, "qsub8", **kw),
    Opcode.OP_RBIT: _fmt_rbit,
    Opcode.OP_REV: _fmt_rev,
    Opcode.OP_REV16: _fmt_rev16,
    Opcode.OP_REVSH: _fmt_revsh,
    Opcode.OP_ROR_IMMEDIATE: _fmt_ror_imm,
    Opcode.OP_ROR_REGISTER: lambda r, **kw: _fmt_shift_reg(r, "ror", **kw),
    Opcode.OP_RRX: _fmt_rrx,
    Opcode.OP_RSB_IMMEDIATE: _fmt_rsb_imm,
    Opcode.OP_RSB_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "rsb", **kw),
    Opcode.OP_SADD16: lambda r, **kw: _fmt_simd3(r, "sadd16", **kw),
    Opcode.OP_SADD8: lambda r, **kw: _fmt_simd3(r, "sadd8", **kw),
    Opcode.OP_SASX: lambda r, **kw: _fmt_simd3(r, "sasx", **kw),
    Opcode.OP_SBC_IMMEDIATE: lambda r, **kw: _fmt_dp_imm(r, "sbc", **kw),
    Opcode.OP_SBC_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "sbc", **kw),
    Opcode.OP_SBFX: _fmt_sbfx,
    Opcode.OP_SDIV: _fmt_sdiv,
    Opcode.OP_SEL: _fmt_sel,
    Opcode.OP_SEV: lambda r, **kw: _fmt_noargs(r, "sev", **kw),
    Opcode.OP_SHADD16: lambda r, **kw: _fmt_simd3(r, "shadd16", **kw),
    Opcode.OP_SHADD8: lambda r, **kw: _fmt_simd3(r, "shadd8", **kw),
    Opcode.OP_SHASX: lambda r, **kw: _fmt_simd3(r, "shasx", **kw),
    Opcode.OP_SHSAX: lambda r, **kw: _fmt_simd3(r, "shsax", **kw),
    Opcode.OP_SHSUB16: lambda r, **kw: _fmt_simd3(r, "shsub16", **kw),
    Opcode.OP_SHSUB8: lambda r, **kw: _fmt_simd3(r, "shsub8", **kw),
    Opcode.OP_SMLABB_SMLABT_SMLATB_SMLATT: _fmt_smlabb_variants,
    Opcode.OP_SMLAD_SMLADX: _fmt_smlad_variants,
    Opcode.OP_SMLAL: lambda r, **kw: _fmt_smlal(r, getattr(r, "setflags", False), **kw),
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
    Opcode.OP_SMULL: lambda r, **kw: _fmt_smull(r, getattr(r, "setflags", False), **kw),
    Opcode.OP_SMULWB_SMULWT: _fmt_smulw_variants,
    Opcode.OP_SMUSD_SMUSDX: _fmt_smusd_variants,
    Opcode.OP_SSAT: _fmt_ssat,
    Opcode.OP_SSAT16: _fmt_ssat16,
    Opcode.OP_SSAX: lambda r, **kw: _fmt_simd3(r, "ssax", **kw),
    Opcode.OP_SSBB: lambda r, **kw: _fmt_noargs(r, "ssbb", **kw),
    Opcode.OP_SSUB16: lambda r, **kw: _fmt_simd3(r, "ssub16", **kw),
    Opcode.OP_SSUB8: lambda r, **kw: _fmt_simd3(r, "ssub8", **kw),
    Opcode.OP_STC_STC2: _fmt_stc_stc2,
    Opcode.OP_STM: lambda r, **kw: _fmt_multi_xfer(r, "stmia", **kw),
    Opcode.OP_STMDB: lambda r, **kw: _fmt_multi_xfer(r, "stmdb", **kw),
    Opcode.OP_STR_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_t(r, "str", **kw),
    Opcode.OP_STR_REGISTER: lambda r, **kw: _fmt_ldst_reg(r, "str", **kw),
    Opcode.OP_STRB_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_t(r, "strb", **kw),
    Opcode.OP_STRB_REGISTER: lambda r, **kw: _fmt_ldst_reg(r, "strb", **kw),
    Opcode.OP_STRBT: lambda r, **kw: _fmt_unpriv_str(r, "strbt", **kw),
    Opcode.OP_STRD_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_dual(r, "strd", **kw),
    Opcode.OP_STREX: _fmt_strex,
    Opcode.OP_STREXB: _fmt_strexb,
    Opcode.OP_STREXH: _fmt_strexh,
    Opcode.OP_STRH_IMMEDIATE: lambda r, **kw: _fmt_ldst_imm_t(r, "strh", **kw),
    Opcode.OP_STRH_REGISTER: lambda r, **kw: _fmt_ldst_reg(r, "strh", **kw),
    Opcode.OP_STRHT: lambda r, **kw: _fmt_unpriv_str(r, "strht", **kw),
    Opcode.OP_STRT: lambda r, **kw: _fmt_unpriv_str(r, "strt", **kw),
    Opcode.OP_SUB_IMMEDIATE: lambda r, **kw: _fmt_dp_imm(r, "sub", **kw),
    Opcode.OP_SUB_REGISTER: lambda r, **kw: _fmt_dp_reg(r, "sub", **kw),
    Opcode.OP_SUB_SP_MINUS_IMMEDIATE: _fmt_sub_sp_imm,
    Opcode.OP_SUB_SP_MINUS_REGISTER: _fmt_sub_sp_reg,
    Opcode.OP_SVC: _fmt_svc,
    Opcode.OP_SXTAB: lambda r, **kw: _fmt_extab(r, "sxtab", **kw),
    Opcode.OP_SXTAB16: lambda r, **kw: _fmt_extab(r, "sxtab16", **kw),
    Opcode.OP_SXTAH: lambda r, **kw: _fmt_extab(r, "sxtah", **kw),
    Opcode.OP_SXTB: lambda r, **kw: _fmt_extend(r, "sxtb", **kw),
    Opcode.OP_SXTB16: lambda r, **kw: _fmt_extend(r, "sxtb16", **kw),
    Opcode.OP_SXTH: lambda r, **kw: _fmt_extend(r, "sxth", **kw),
    Opcode.OP_TBB_TBH: _fmt_tbb_tbh,
    Opcode.OP_TEQ_IMMEDIATE: lambda r, **kw: _fmt_test_imm(r, "teq", **kw),
    Opcode.OP_TEQ_REGISTER: lambda r, **kw: _fmt_test_reg(r, "teq", **kw),
    Opcode.OP_TST_IMMEDIATE: lambda r, **kw: _fmt_test_imm(r, "tst", **kw),
    Opcode.OP_TST_REGISTER: lambda r, **kw: _fmt_test_reg(r, "tst", **kw),
    Opcode.OP_UADD16: lambda r, **kw: _fmt_simd3(r, "uadd16", **kw),
    Opcode.OP_UADD8: lambda r, **kw: _fmt_simd3(r, "uadd8", **kw),
    Opcode.OP_UASX: lambda r, **kw: _fmt_simd3(r, "uasx", **kw),
    Opcode.OP_UBFX: _fmt_ubfx,
    Opcode.OP_UDF: _fmt_udf,
    Opcode.OP_UDIV: _fmt_udiv,
    Opcode.OP_UHADD16: lambda r, **kw: _fmt_simd3(r, "uhadd16", **kw),
    Opcode.OP_UHADD8: lambda r, **kw: _fmt_simd3(r, "uhadd8", **kw),
    Opcode.OP_UHASX: lambda r, **kw: _fmt_simd3(r, "uhasx", **kw),
    Opcode.OP_UHSAX: lambda r, **kw: _fmt_simd3(r, "uhsax", **kw),
    Opcode.OP_UHSUB16: lambda r, **kw: _fmt_simd3(r, "uhsub16", **kw),
    Opcode.OP_UHSUB8: lambda r, **kw: _fmt_simd3(r, "uhsub8", **kw),
    Opcode.OP_UMAAL: _fmt_umaal,
    Opcode.OP_UMLAL: lambda r, **kw: _fmt_umlal(r, getattr(r, "setflags", False), **kw),
    Opcode.OP_UMULL: lambda r, **kw: _fmt_umull(r, getattr(r, "setflags", False), **kw),
    Opcode.OP_UQADD16: lambda r, **kw: _fmt_simd3(r, "uqadd16", **kw),
    Opcode.OP_UQADD8: lambda r, **kw: _fmt_simd3(r, "uqadd8", **kw),
    Opcode.OP_UQASX: lambda r, **kw: _fmt_simd3(r, "uqasx", **kw),
    Opcode.OP_UQSAX: lambda r, **kw: _fmt_simd3(r, "uqsax", **kw),
    Opcode.OP_UQSUB16: lambda r, **kw: _fmt_simd3(r, "uqsub16", **kw),
    Opcode.OP_UQSUB8: lambda r, **kw: _fmt_simd3(r, "uqsub8", **kw),
    Opcode.OP_USAD8: _fmt_usad8,
    Opcode.OP_USADA8: _fmt_usada8,
    Opcode.OP_USAT: _fmt_usat,
    Opcode.OP_USAT16: _fmt_usat16,
    Opcode.OP_USAX: lambda r, **kw: _fmt_simd3(r, "usax", **kw),
    Opcode.OP_USUB16: lambda r, **kw: _fmt_simd3(r, "usub16", **kw),
    Opcode.OP_USUB8: lambda r, **kw: _fmt_simd3(r, "usub8", **kw),
    Opcode.OP_UXTAB: lambda r, **kw: _fmt_extab(r, "uxtab", **kw),
    Opcode.OP_UXTAB16: lambda r, **kw: _fmt_extab(r, "uxtab16", **kw),
    Opcode.OP_UXTAH: lambda r, **kw: _fmt_extab(r, "uxtah", **kw),
    Opcode.OP_UXTB: lambda r, **kw: _fmt_extend(r, "uxtb", **kw),
    Opcode.OP_UXTB16: lambda r, **kw: _fmt_extend(r, "uxtb16", **kw),
    Opcode.OP_UXTH: lambda r, **kw: _fmt_extend(r, "uxth", **kw),
    Opcode.OP_VABS: lambda r, **kw: _fmt_vfp_dp2(r, "vabs", **kw),
    Opcode.OP_VADD: lambda r, **kw: _fmt_vfp_dp3(r, "vadd", **kw),
    Opcode.OP_VCMP_VCMPE: _fmt_vcmp_vcmpe,
    Opcode.OP_VCVTA_VCVTN_VCVTP_VCVTM: _fmt_vcvta_round,
    Opcode.OP_VCVT_VCVTR_INTEGER: _fmt_vcvt_integer,
    Opcode.OP_VCVT_FIXED_POINT: _fmt_vcvt_fixed,
    Opcode.OP_VCVT_DOUBLE_SINGLE: _fmt_vcvt_double_single,
    Opcode.OP_VCVTB_VCVTT: _fmt_vcvtb_vcvtt,
    Opcode.OP_VDIV: lambda r, **kw: _fmt_vfp_dp3(r, "vdiv", **kw),
    Opcode.OP_VFMA_VFMS: _fmt_vfma_vfms,
    Opcode.OP_VFNMA_VFNMS: _fmt_vfnma_vfnms,
    Opcode.OP_VLDM: lambda r, **kw: _fmt_vldm_vstm(r, "vldm", **kw),
    Opcode.OP_VLDR: lambda r, **kw: _fmt_vldr_vstr(r, "vldr", **kw),
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
    Opcode.OP_VMUL: lambda r, **kw: _fmt_vfp_dp3(r, "vmul", **kw),
    Opcode.OP_VNEG: lambda r, **kw: _fmt_vfp_dp2(r, "vneg", **kw),
    Opcode.OP_VNMLA_VNMLS_VNMUL: _fmt_vnmla_vnmls_vnmul,
    Opcode.OP_VPOP: lambda r, **kw: _fmt_vpush_vpop(r, "vpop", **kw),
    Opcode.OP_VPUSH: lambda r, **kw: _fmt_vpush_vpop(r, "vpush", **kw),
    Opcode.OP_VRINTA_VRINTN_VRINTP_VRINTM: _fmt_vrint_round,
    Opcode.OP_VRINTX: _fmt_vrintx,
    Opcode.OP_VRINTZ_VRINTR: _fmt_vrint_zr,
    Opcode.OP_VSEL: _fmt_vsel,
    Opcode.OP_VSQRT: lambda r, **kw: _fmt_vfp_dp2(r, "vsqrt", **kw),
    Opcode.OP_VSTM: lambda r, **kw: _fmt_vldm_vstm(r, "vstm", **kw),
    Opcode.OP_VSTR: lambda r, **kw: _fmt_vldr_vstr(r, "vstr", **kw),
    Opcode.OP_VSUB: lambda r, **kw: _fmt_vfp_dp3(r, "vsub", **kw),
    Opcode.OP_WFE: lambda r, **kw: _fmt_noargs(r, "wfe", **kw),
    Opcode.OP_WFI: lambda r, **kw: _fmt_noargs(r, "wfi", **kw),
    Opcode.OP_YIELD: lambda r, **kw: _fmt_noargs(r, "yield", **kw),
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
    result: object,
    instr: int,
    size: InstructionSize,
    offset: int = 0,
    istate: int = 0,
) -> str:
    """Convert a decoded instruction dataclass to an assembler mnemonic string.

    Args:
        result: Decoded instruction dataclass instance (or pseudo-instruction).
        instr: The instruction word, exactly `size` bits wide -- the same word
               that was handed to :func:`decode`. Which of an instruction's
               encodings was matched is not in `result`, so the spelling of a
               few of them (MCR vs MCR2, `add r0, r1` vs `add r0, r0, r1`)
               has to be read back off the word.
        size: Width of that word, as determined by the caller before decoding.
              It decides the `.w`/`.n` qualifiers and the operand forms that
              only one width spells.
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
        # A word no encoding matches has no mnemonic to spell, so the whole
        # field is a comment carrying the word itself, as objdump writes it.
        return f"\t\t@ <UNDEFINED> instruction: 0x{instr:0{size // 4}x}"
    if opc == Opcode.OP_UNDEFINED:
        return "<undefined>"
    if opc == Opcode.OP_UNPREDICTABLE:
        return "<unpredictable>"
    if opc == Opcode.OP_SEE:
        return "<see>"

    fmt_func = _DISPATCH.get(opc)

    if fmt_func is None:
        return repr(result)

    # Every formatter is handed the whole word; the ones that spell nothing
    # but their own operands absorb what they have no use for through **_.
    asm = fmt_func(result, instr=instr, size=size, offset=offset)

    # B (T1/T3) and VSEL carry their own condition, which is what the
    # architecture reports as CurrentCond -- they never take the block's. (Both
    # are UNPREDICTABLE in a block, so the decoder rarely lets one through.)
    if getattr(result, "cond", COND_AL) == COND_AL:
        asm = _in_cond_block(asm, istate)
    return asm

"""ARMv7-M instruction disassembler.

Converts decoded instruction dataclass instances to UAL (Unified Assembler
Language) syntax strings, suitable for display as disassembled output.
"""

from __future__ import annotations

from typing import Any

from armv7m_decoder._decoder import NoMatch, See, Undefined, Unpredictable

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
    "",
]
_BARRIER_OPTIONS: dict[int, str] = {
    0x2: "osh",
    0x3: "nsh",
    0x4: "ish",
    0x5: "un",
    0x6: "sy",
    0x7: "st",
    0x8: "ld",
}
_SPEC_REGS: dict[int, str] = {
    0x00: "apsr",
    0x01: "iapsr",
    0x02: "eapsr",
    0x03: "xpsr",
    0x05: "ipsr",
    0x06: "epsr",
    0x07: "iepsr",
    0x08: "msp",
    0x09: "psp",
    0x10: "primask",
    0x11: "basepri",
    0x12: "basepri_max",
    0x13: "faultmask",
    0x14: "control",
}
_REG_NUMBERS: dict[str, int] = {
    **{f"r{i}": i for i in range(13)},
    "sp": 13,
    "lr": 14,
    "pc": 15,
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _reg(r: int) -> str:
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
    if cond >= 14:
        return ""
    return _COND_CODES[cond]


def _flags(setflags: bool) -> str:
    return "s" if setflags else ""


def _barrier(opt: int) -> str:
    return _BARRIER_OPTIONS.get(opt, f"#{opt}")


def _reg_list(registers: int) -> str:
    names = [_reg(i) for i in range(16) if registers & (1 << i)]
    if not names:
        return "{}"
    parts: list[str] = []
    i = 0
    while i < len(names):
        j = i
        while (
            j + 1 < len(names)
            and _REG_NUMBERS.get(names[j + 1], -1) == _REG_NUMBERS.get(names[j], -2) + 1
        ):
            j += 1
        if i == j:
            parts.append(names[i])
        elif j == i + 1:
            parts.append(names[i])
            parts.append(names[j])
        else:
            parts.append(f"{names[i]}-{names[j]}")
        i = j + 1
    return "{" + ", ".join(parts) + "}"


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


def _addr_imm(n: int, imm32: int, index: bool, add: bool, wback: bool) -> str:
    sign = "" if add else "-"
    offset_text = "" if imm32 == 0 and add else f", #{sign}{imm32}"
    if index and wback:
        return f"[{_reg(n)}{offset_text}]!"
    if index and not wback:
        return f"[{_reg(n)}{offset_text}]"
    return f"[{_reg(n)}], #{sign}{imm32}"


def _addr_imm_dual(
    t: int, t2: int, n: int, imm32: int, index: bool, add: bool, wback: bool
) -> str:
    return f"{_reg(t)}, {_reg(t2)}, {_addr_imm(n, imm32, index, add, wback)}"


def _addr_reg(t: int, n: int, m: int, shift_t: int, shift_n: int) -> str:
    if shift_t == 0 and shift_n == 0:
        return f"{_reg(t)}, [{_reg(n)}, {_reg(m)}]"
    return f"{_reg(t)}, [{_reg(n)}, {_reg(m)}, lsl #{shift_n}]"


def _addr_excl(d: int, t: int, n: int, imm32: int) -> str:
    if imm32 == 0:
        return f"{_reg(d)}, {_reg(t)}, [{_reg(n)}]"
    return f"{_reg(d)}, {_reg(t)}, [{_reg(n)}, #{imm32}]"


def _addr_excl_single(t: int, n: int, imm32: int) -> str:
    if imm32 == 0:
        return f"{_reg(t)}, [{_reg(n)}]"
    return f"{_reg(t)}, [{_reg(n)}, #{imm32}]"


def _addr_literal(t: int, imm32: int, add: bool) -> str:
    sign = "" if add else "-"
    return f"{_reg(t)}, [pc, #{sign}{imm32}]"


def _addr_unpriv(t: int, n: int, imm32: int, register_form: bool) -> str:
    if register_form:
        return f"{_reg(t)}, [{_reg(n)}], {_reg(imm32)}"
    if imm32 == 0:
        return f"{_reg(t)}, [{_reg(n)}]"
    return f"{_reg(t)}, [{_reg(n)}, #{imm32}]"


def _addr_unpriv_ldr(t: int, n: int, imm32: int) -> str:
    if imm32 == 0:
        return f"{_reg(t)}, [{_reg(n)}]"
    return f"{_reg(t)}, [{_reg(n)}, #{imm32}]"


def _branch_target(offset: int, imm32: int) -> str:
    return f"#{hex((offset + 4 + imm32) & 0xFFFFFFFF)}"


# ---------------------------------------------------------------------------
# Combined-mnemonic helpers
# ---------------------------------------------------------------------------


def _is_coproc2(instr: int) -> bool:
    return bool(instr & 0x10000000)


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


def _fmt_dp_imm(result: Any, mnemonic: str) -> str:
    s = _flags(result.setflags)
    return f"{mnemonic}{s} {_reg(result.d)}, {_reg(result.n)}, #{result.imm32}"


def _fmt_mov_imm(result: Any) -> str:
    s = _flags(result.setflags)
    return f"mov{s} {_reg(result.d)}, #{result.imm32}"


def _fmt_mvn_imm(result: Any, mnemonic: str = "mvn") -> str:
    s = _flags(result.setflags)
    return f"{mnemonic}{s} {_reg(result.d)}, #{result.imm32}"


def _fmt_and_imm(result: Any, mnemonic: str = "and") -> str:
    s = _flags(getattr(result, "setflags", False))
    return f"{mnemonic}{s} {_reg(result.d)}, {_reg(result.n)}, #{result.imm32}"


# --- Data-processing register ---


def _fmt_dp_reg(result: Any, mnemonic: str) -> str:
    s = _flags(result.setflags)
    sh = _shift(result.shift_t, result.shift_n)
    return f"{mnemonic}{s} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}{sh}"


def _fmt_mov_reg(result: Any) -> str:
    s = _flags(result.setflags)
    return f"mov{s} {_reg(result.d)}, {_reg(result.m)}"


def _fmt_mvn_reg(result: Any) -> str:
    s = _flags(result.setflags)
    sh = _shift(result.shift_t, result.shift_n)
    return f"mvn{s} {_reg(result.d)}, {_reg(result.m)}{sh}"


def _fmt_rrx(result: Any) -> str:
    s = _flags(result.setflags)
    return f"rrx{s} {_reg(result.d)}, {_reg(result.m)}"


# --- Test/compare ---


def _fmt_test_imm(result: Any, mnemonic: str) -> str:
    return f"{mnemonic} {_reg(result.n)}, #{result.imm32}"


def _fmt_test_reg(result: Any, mnemonic: str) -> str:
    sh = _shift(result.shift_t, result.shift_n)
    return f"{mnemonic} {_reg(result.n)}, {_reg(result.m)}{sh}"


# --- Shift immediate ---


def _fmt_shift_imm(result: Any, mnemonic: str) -> str:
    s = _flags(result.setflags)
    return f"{mnemonic}{s} {_reg(result.d)}, {_reg(result.m)}, #{result.shift_n}"


def _fmt_ror_imm(result: Any) -> str:
    s = _flags(result.setflags)
    return f"ror{s} {_reg(result.d)}, {_reg(result.m)}, #{result.shift_n}"


# --- Shift register ---


def _fmt_shift_reg(result: Any, mnemonic: str) -> str:
    s = _flags(result.setflags)
    return f"{mnemonic}{s} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# --- SP arithmetic ---


def _fmt_add_sp_imm(result: Any) -> str:
    s = _flags(result.setflags)
    return f"add{s} {_reg(result.d)}, sp, #{result.imm32}"


def _fmt_add_sp_reg(result: Any) -> str:
    s = _flags(result.setflags)
    sh = _shift(result.shift_t, result.shift_n)
    return f"add{s} {_reg(result.d)}, sp, {_reg(result.m)}{sh}"


def _fmt_sub_sp_imm(result: Any) -> str:
    s = _flags(result.setflags)
    return f"sub{s} {_reg(result.d)}, sp, #{result.imm32}"


def _fmt_sub_sp_reg(result: Any) -> str:
    s = _flags(result.setflags)
    sh = _shift(result.shift_t, result.shift_n)
    return f"sub{s} {_reg(result.d)}, sp, {_reg(result.m)}{sh}"


# --- ADR ---


def _fmt_adr(result: Any) -> str:
    sign = "" if result.add else "-"
    return f"adr {_reg(result.d)}, #{sign}{result.imm32}"


# --- Load/store immediate ---


def _fmt_ldst_imm(result: Any, mnemonic: str) -> str:
    addr = _addr_imm(result.n, result.imm32, result.index, result.add, result.wback)
    return f"{mnemonic} {addr}"


def _fmt_ldst_imm_t(result: Any, mnemonic: str) -> str:
    addr = _addr_imm(result.n, result.imm32, result.index, result.add, result.wback)
    return f"{mnemonic} {_reg(result.t)}, {addr}"


def _fmt_ldst_imm_dual(result: Any, mnemonic: str) -> str:
    addr = _addr_imm_dual(
        result.t,
        result.t2,
        result.n,
        result.imm32,
        result.index,
        result.add,
        result.wback,
    )
    return f"{mnemonic} {addr}"


# --- Load/store register ---


def _fmt_ldst_reg(result: Any, mnemonic: str) -> str:
    addr = _addr_reg(result.t, result.n, result.m, result.shift_t, result.shift_n)
    return f"{mnemonic} {addr}"


def _fmt_ldst_reg_rt(result: Any, mnemonic: str) -> str:
    addr = _addr_reg(result.t, result.n, result.m, result.shift_t, result.shift_n)
    return f"{mnemonic} {addr}"


# --- Load/store literal ---


def _fmt_ldst_lit(result: Any, mnemonic: str) -> str:
    return f"{mnemonic} {_addr_literal(result.t, result.imm32, result.add)}"


def _fmt_ldst_lit_dual(result: Any, mnemonic: str = "ldrd") -> str:
    sign = "" if result.add else "-"
    return (
        f"{mnemonic} {_reg(result.t)}, {_reg(result.t2)}, [pc, #{sign}{result.imm32}]"
    )


# --- Preload (PLD / PLI) ---


def _fmt_pld_imm(result: Any) -> str:
    sign = "" if result.add else "-"
    return f"pld [{_reg(result.n)}, #{sign}{result.imm32}]"


def _fmt_pld_lit(result: Any) -> str:
    sign = "" if result.add else "-"
    return f"pld [pc, #{sign}{result.imm32}]"


def _fmt_pld_reg(result: Any) -> str:
    if result.shift_t == 0 and result.shift_n == 0:
        return f"pld [{_reg(result.n)}, {_reg(result.m)}]"
    return f"pld [{_reg(result.n)}, {_reg(result.m)}, lsl #{result.shift_n}]"


def _fmt_pli_imm_lit(result: Any) -> str:
    sign = "" if result.add else "-"
    return f"pli [{_reg(result.n)}, #{sign}{result.imm32}]"


def _fmt_pli_reg(result: Any) -> str:
    if result.shift_t == 0 and result.shift_n == 0:
        return f"pli [{_reg(result.n)}, {_reg(result.m)}]"
    return f"pli [{_reg(result.n)}, {_reg(result.m)}, lsl #{result.shift_n}]"


# --- Exclusive load/store ---


def _fmt_strex(result: Any) -> str:
    return f"strex {_addr_excl(result.d, result.t, result.n, result.imm32)}"


def _fmt_ldrex(result: Any) -> str:
    return f"ldrex {_addr_excl_single(result.t, result.n, result.imm32)}"


def _fmt_strexb(result: Any) -> str:
    return f"strexb {_reg(result.d)}, {_reg(result.t)}, [{_reg(result.n)}]"


def _fmt_ldrexb(result: Any) -> str:
    return f"ldrexb {_reg(result.t)}, [{_reg(result.n)}]"


def _fmt_strexh(result: Any) -> str:
    return f"strexh {_reg(result.d)}, {_reg(result.t)}, [{_reg(result.n)}]"


def _fmt_ldrexh(result: Any) -> str:
    return f"ldrexh {_reg(result.t)}, [{_reg(result.n)}]"


# --- Unprivileged load/store ---


def _fmt_unpriv_ldr(result: Any, mnemonic: str) -> str:
    if result.register_form:
        return f"{mnemonic} {_reg(result.t)}, [{_reg(result.n)}], {_reg(result.imm32)}"
    if result.imm32 == 0:
        return f"{mnemonic} {_reg(result.t)}, [{_reg(result.n)}]"
    return f"{mnemonic} {_reg(result.t)}, [{_reg(result.n)}, #{result.imm32}]"


def _fmt_unpriv_str(result: Any, mnemonic: str) -> str:
    return _fmt_unpriv_ldr(result, mnemonic)


# --- Multiple transfer ---


def _fmt_multi_xfer(result: Any, mnemonic: str) -> str:
    wb = "!" if result.wback else ""
    return f"{mnemonic} {_reg(result.n)}{wb}, {_reg_list(result.registers)}"


# --- Stack ---


def _fmt_pop(result: Any) -> str:
    return f"pop {_reg_list(result.registers)}"


def _fmt_push(result: Any) -> str:
    return f"push {_reg_list(result.registers)}"


# --- Branch ---


def _fmt_b(result: Any, offset: int = 0) -> str:
    c = "" if result.cond == 0 else _cond(result.cond)
    return f"b{c} {_branch_target(offset, result.imm32)}"


def _fmt_bl(result: Any, offset: int = 0) -> str:
    return f"bl {_branch_target(offset, result.imm32)}"


def _fmt_blx_reg(result: Any) -> str:
    return f"blx {_reg(result.m)}"


def _fmt_bx(result: Any) -> str:
    return f"bx {_reg(result.m)}"


def _fmt_cbnz_cbz(result: Any, offset: int = 0) -> str:
    mnemonic = "cbnz" if result.nonzero else "cbz"
    return f"{mnemonic} {_reg(result.n)}, {_branch_target(offset, result.imm32)}"


def _fmt_tbb_tbh(result: Any, offset: int = 0) -> str:
    mnemonic = "tbh" if result.is_tbh else "tbb"
    return f"{mnemonic} [{_reg(result.n)}, {_reg(result.m)}]"


# --- Barrier ---


def _fmt_barrier(result: Any, mnemonic: str) -> str:
    return f"{mnemonic} {_barrier(result.option)}"


# --- Bitfield ---


def _fmt_bfi(result: Any) -> str:
    width = result.msbit - result.lsbit + 1
    return f"bfi {_reg(result.d)}, {_reg(result.n)}, #{result.lsbit}, #{width}"


def _fmt_bfc(result: Any) -> str:
    width = result.msbit - result.lsbit + 1
    return f"bfc {_reg(result.d)}, #{result.lsbit}, #{width}"


def _fmt_ubfx(result: Any) -> str:
    width = result.widthminus1 + 1
    return f"ubfx {_reg(result.d)}, {_reg(result.n)}, #{result.lsbit}, #{width}"


def _fmt_sbfx(result: Any) -> str:
    width = result.widthminus1 + 1
    return f"sbfx {_reg(result.d)}, {_reg(result.n)}, #{result.lsbit}, #{width}"


# --- Extend ---


def _fmt_extend(result: Any, mnemonic: str) -> str:
    if result.rotation == 0:
        return f"{mnemonic} {_reg(result.d)}, {_reg(result.m)}"
    rot = f"ror #{result.rotation}"
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.m)}, {rot}"


def _fmt_extab(result: Any, mnemonic: str) -> str:
    if result.rotation == 0:
        return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"
    rot = f"ror #{result.rotation}"
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {rot}"


# --- Saturate ---


def _fmt_ssat(result: Any) -> str:
    sh = _shift(result.shift_t, result.shift_n)
    return f"ssat {_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}{sh}"


def _fmt_usat(result: Any) -> str:
    sh = _shift(result.shift_t, result.shift_n)
    return f"usat {_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}{sh}"


def _fmt_ssat16(result: Any) -> str:
    return f"ssat16 {_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}"


def _fmt_usat16(result: Any) -> str:
    return f"usat16 {_reg(result.d)}, #{result.saturate_to}, {_reg(result.n)}"


# --- SIMD 3-register ---


def _fmt_simd3(result: Any, mnemonic: str) -> str:
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# --- SIMD 3-register + accumulator ---


def _fmt_smla(result: Any, mnemonic: str, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"{mnemonic}{s} {_reg(result.d)}, {_reg(result.n)},"
        f" {_reg(result.m)}, {_reg(result.a)}"
    )


def _fmt_smlal(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"smlal{s} {_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


# --- SIMD long multiply ---


def _fmt_smull(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"smull{s} {_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


def _fmt_umull(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"umull{s} {_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


def _fmt_umlal(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"umlal{s} {_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


def _fmt_umaal(result: Any) -> str:
    return (
        f"umaal {_reg(result.dLo)}, {_reg(result.dHi)},"
        f" {_reg(result.n)}, {_reg(result.m)}"
    )


# --- Multiply ---


def _fmt_mul(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return f"mul{s} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_mla(result: Any, setflags: bool = False) -> str:
    s = _flags(setflags)
    return (
        f"mla{s} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"
    )


def _fmt_mls(result: Any) -> str:
    return f"mls {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"


def _fmt_sdiv(result: Any) -> str:
    return f"sdiv {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_udiv(result: Any) -> str:
    return f"udiv {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# --- Miscellaneous ---


def _fmt_clz(result: Any) -> str:
    return f"clz {_reg(result.d)}, {_reg(result.m)}"


def _fmt_rev(result: Any) -> str:
    return f"rev {_reg(result.d)}, {_reg(result.m)}"


def _fmt_rev16(result: Any) -> str:
    return f"rev16 {_reg(result.d)}, {_reg(result.m)}"


def _fmt_revsh(result: Any) -> str:
    return f"revsh {_reg(result.d)}, {_reg(result.m)}"


def _fmt_rbit(result: Any) -> str:
    return f"rbit {_reg(result.d)}, {_reg(result.m)}"


def _fmt_sel(result: Any) -> str:
    return f"sel {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_pkhbt_pkhtb(result: Any) -> str:
    mnemonic = "pkhtb" if result.tbform else "pkhbt"
    sh = _shift(result.shift_t, result.shift_n)
    # When tbform is False, shift type is LSL; when True, ASR
    if mnemonic == "pkhbt" and result.shift_t == 0 and result.shift_n == 0:
        return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}{sh}"


# --- MOVT ---


def _fmt_movt(result: Any) -> str:
    return f"movt {_reg(result.d)}, #{result.imm16}"


# --- USAD8 / USADA8 ---


def _fmt_usad8(result: Any) -> str:
    return f"usad8 {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_usada8(result: Any) -> str:
    return (
        f"usada8 {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"
    )


# --- System registers ---


def _fmt_mrs(result: Any) -> str:
    spec = _SPEC_REGS.get(result.SYSm, f"spec_reg_{result.SYSm:#x}")
    return f"mrs {_reg(result.d)}, {spec}"


def _fmt_msr(result: Any) -> str:
    spec = _SPEC_REGS.get(result.SYSm, f"spec_reg_{result.SYSm:#x}")
    return f"msr {spec}, {_reg(result.n)}"


# --- CPS ---


def _fmt_cps(result: Any) -> str:
    effect = "ie" if result.enable else "id"
    flags = ""
    if result.affectPRI:
        flags += "i"
    if result.affectFAULT:
        flags += "f"
    return f"cps{effect} {flags}"


# --- IT ---


def _fmt_it(result: Any) -> str:
    c = _COND_CODES[result.firstcond]
    mask = result.mask
    suffix = ""
    for i in range(3, 0, -1):
        if mask & (1 << i):
            x = "t" if ((mask >> i) & 1) == ((result.firstcond >> 0) & 1) else "e"
            suffix += x
    return f"it{suffix} {c}"


# --- Misc zero-operand ---


def _fmt_noargs(mnemonic: str) -> str:
    return mnemonic


def _fmt_bkpt(result: Any) -> str:
    return f"bkpt #{result.imm32}"


def _fmt_svc(result: Any) -> str:
    return f"svc #{result.imm32}"


def _fmt_udf(result: Any) -> str:
    return f"udf #{result.imm32}"


def _fmt_db(result: Any) -> str:
    return f"db {_barrier(result.option)}"


# --- VFP data-processing 3-reg ---


def _fmt_vfp_dp3(result: Any, mnemonic: str) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {n}, {m}"


# --- VFP data-processing 2-reg ---


def _fmt_vfp_dp2(result: Any, mnemonic: str) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {m}"


# --- VFP combined mnemonics ---


def _fmt_vfma_vfms(result: Any) -> str:
    mnemonic = "vfms" if result.op1_neg else "vfma"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {n}, {m}"


def _fmt_vfnma_vfnms(result: Any) -> str:
    mnemonic = "vfnms" if result.op1_neg else "vfnma"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {n}, {m}"


def _fmt_vmla_vmls(result: Any) -> str:
    mnemonic = "vmls" if not result.add else "vmla"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {n}, {m}"


def _fmt_vnmla_vnmls_vnmul(result: Any) -> str:
    types = {0: "vnmla", 1: "vnmls", 2: "vnmul"}
    mnemonic = types.get(result.type, "vnmla")
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {n}, {m}"


# --- VFP compare ---


def _fmt_vcmp_vcmpe(result: Any) -> str:
    mnemonic = "vcmpe" if result.quiet_nan_exc else "vcmp"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    if result.with_zero:
        return f"{mnemonic}{precision} {d}, #0.0"
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {m}"


# --- VFP convert ---

_VCVTA_MODES = {0: "vcvta", 1: "vcvtn", 2: "vcvtp", 3: "vcvtm"}


def _fmt_vcvta_round(result: Any) -> str:
    mnemonic = _VCVTA_MODES.get(result.round_mode, "vcvta")
    signed = "u" if result.unsigned else "s"
    src_prec = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}.{signed}32{src_prec} {d}, {m}"


def _fmt_vcvt_integer(result: Any) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    unsigned = "u" if result.unsigned else ""
    rounding = "r" if (result.round_zero or result.round_nearest) else ""
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    if result.to_integer:
        return f"vcvt{rounding}{unsigned}.s32{precision} {d}, {m}"
    return f"vcvt{rounding}{unsigned}{precision}.s32 {d}, {m}"


def _fmt_vcvt_fixed(result: Any) -> str:
    unsigned = "u" if result.unsigned else ""
    d = _sreg(result.d)
    if result.to_fixed:
        return f"vcvt{unsigned}.s32.f32 {d}, {d}, #{result.frac_bits}"
    return f"vcvt{unsigned}.f32.s32 {d}, {d}, #{result.frac_bits}"


def _fmt_vcvt_double_single(result: Any) -> str:
    if result.double_to_single:
        return f"vcvt.f32.f64 {_sreg(result.d)}, {_dreg(result.m)}"
    return f"vcvt.f64.f32 {_dreg(result.d)}, {_sreg(result.m)}"


def _fmt_vcvtb_vcvtt(result: Any) -> str:
    mnemonic = "vcvtt" if result.lowbit else "vcvtb"
    return f"{mnemonic}.f32.f16 {_sreg(result.d)}, {_sreg(result.m)}"


# --- VFP move ---


def _fmt_vmov_imm(result: Any) -> str:
    return f"vmov {_vfp_reg(result.dp_operation, result.d)}, #{result.imm32}"


def _fmt_vmov_reg(result: Any) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"vmov{precision} {d}, {m}"


def _fmt_vmov_core_to_scalar(result: Any) -> str:
    return f"vmov {_sreg(result.d)}, {_reg(result.t)}"


def _fmt_vmov_scalar_to_core(result: Any) -> str:
    return f"vmov {_reg(result.t)}, {_sreg(result.n)}"


def _fmt_vmov_core_and_single(result: Any) -> str:
    if result.to_arm_register:
        return f"vmov {_reg(result.t)}, {_sreg(result.n)}"
    return f"vmov {_sreg(result.n)}, {_reg(result.t)}"


def _fmt_vmov_two_core_and_single(result: Any) -> str:
    if result.to_arm_registers:
        return (
            f"vmov {_reg(result.t)}, {_reg(result.t2)},"
            f" {_sreg(result.m)}, {_sreg(result.m + 1)}"
        )
    return (
        f"vmov {_sreg(result.m)}, {_sreg(result.m + 1)},"
        f" {_reg(result.t)}, {_reg(result.t2)}"
    )


def _fmt_vmov_two_core_and_doubleword(result: Any) -> str:
    if result.to_arm_registers:
        return f"vmov {_reg(result.t)}, {_reg(result.t2)}, {_dreg(result.m)}"
    return f"vmov {_dreg(result.m)}, {_reg(result.t)}, {_reg(result.t2)}"


# --- VFP system ---


def _fmt_vmrs(result: Any) -> str:
    if result.t == 15:
        return "vmrs apsr_nzcv, fpscr"
    return f"vmrs {_reg(result.t)}, fpscr"


def _fmt_vmsr(result: Any) -> str:
    return f"vmsr fpscr, {_reg(result.t)}"


# --- VFP conditional select ---


def _fmt_vsel(result: Any) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    c = _cond(result.cond)
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"vsel{c}{precision} {d}, {n}, {m}"


# --- VFP round ---

_VRINT_MODES = {0: "vrinta", 1: "vrintn", 2: "vrintp", 3: "vrintm"}
_VRINTZ_MODES = {0: "vrintz", 1: "vrintr"}


def _fmt_vrint_round(result: Any) -> str:
    mnemonic = _VRINT_MODES.get(result.rmode, "vrinta")
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {m}"


def _fmt_vrint_zr(result: Any) -> str:
    mnemonic = _VRINTZ_MODES.get(result.rmode, "vrintz")
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {m}"


def _fmt_vrintx(result: Any) -> str:
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"vrintx{precision} {d}, {m}"


# --- VFP load/store ---


def _fmt_vldr_vstr(result: Any, mnemonic: str) -> str:
    sign = "" if result.add else "-"
    single_reg = result.single_reg
    reg_name_fn = _sreg if single_reg else _dreg
    return (
        f"{mnemonic} {reg_name_fn(result.d)}, [{_reg(result.n)}, #{sign}{result.imm32}]"
    )


def _fmt_vldm_vstm(result: Any, mnemonic: str) -> str:
    wb = "!" if result.wback else ""
    reg_str = _vfp_reg_list(result.single_regs, result.d, result.regs)
    return f"{mnemonic} {_reg(result.n)}{wb}, {reg_str}"


def _fmt_vpush_vpop(result: Any, mnemonic: str) -> str:
    reg_str = _vfp_reg_list(result.single_regs, result.d, result.regs)
    return f"{mnemonic} {reg_str}"


# --- VFP max/min ---


def _fmt_vmaxnm_vminnm(result: Any) -> str:
    mnemonic = "vmaxnm" if result.maximum else "vminnm"
    precision = ".f64" if result.dp_operation else ".f32"
    d = _vfp_reg(result.dp_operation, result.d)
    n = _vfp_reg(result.dp_operation, result.n)
    m = _vfp_reg(result.dp_operation, result.m)
    return f"{mnemonic}{precision} {d}, {n}, {m}"


# --- SIMD combined mnemonics ---


def _fmt_smlabb_variants(result: Any) -> str:
    mnemonic = _nhigh_mhigh_mnemonic("smla", result.n_high, result.m_high)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlalbb_variants(result: Any) -> str:
    mnemonic = _nhigh_mhigh_mnemonic("smlal", result.n_high, result.m_high)
    return f"{mnemonic} {_reg(result.dLo)}, {_reg(result.dHi)}, {_reg(result.n)}, {_reg(result.m)}"  # noqa: E501


def _fmt_smulbb_variants(result: Any) -> str:
    mnemonic = _nhigh_mhigh_mnemonic("smul", result.n_high, result.m_high)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smlad_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smlad", result.m_swap)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlald_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smlald", result.m_swap)
    return f"{mnemonic} {_reg(result.dLo)}, {_reg(result.dHi)}, {_reg(result.n)}, {_reg(result.m)}"  # noqa: E501


def _fmt_smlaw_variants(result: Any) -> str:
    mnemonic = _mhigh_mnemonic("smlaw", result.m_high)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlsd_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smlsd", result.m_swap)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smlsld_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smlsld", result.m_swap)
    return f"{mnemonic} {_reg(result.dLo)}, {_reg(result.dHi)}, {_reg(result.n)}, {_reg(result.m)}"  # noqa: E501


def _fmt_smmla_variants(result: Any) -> str:
    mnemonic = _round_mnemonic("smmla", result.round)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smmls_variants(result: Any) -> str:
    mnemonic = _round_mnemonic("smmls", result.round)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}, {_reg(result.a)}"  # noqa: E501


def _fmt_smmul_variants(result: Any) -> str:
    mnemonic = _round_mnemonic("smmul", result.round)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smuad_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smuad", result.m_swap)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smulw_variants(result: Any) -> str:
    mnemonic = _mhigh_mnemonic("smulw", result.m_high)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


def _fmt_smusd_variants(result: Any) -> str:
    mnemonic = _mswap_mnemonic("smusd", result.m_swap)
    return f"{mnemonic} {_reg(result.d)}, {_reg(result.n)}, {_reg(result.m)}"


# ---------------------------------------------------------------------------
# Coprocessor formatters
# ---------------------------------------------------------------------------


def _fmt_cdp_cdp2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"cdp{suffix} p{result.cp}, #{result.opc1}, {_reg(result.CRd)}, c{result.CRn}, c{result.CRm}, #{result.opc2}"  # noqa: E501


def _fmt_mcr_mcr2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"mcr{suffix} p{result.cp}, #{result.opc1}, {_reg(result.t)}, c{result.CRn}, c{result.CRm}, #{result.opc2}"  # noqa: E501


def _fmt_mrc_mrc2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"mrc{suffix} p{result.cp}, #{result.opc1}, {_reg(result.t)}, c{result.CRn}, c{result.CRm}, #{result.opc2}"  # noqa: E501


def _fmt_mcrr_mcrr2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"mcrr{suffix} p{result.cp}, #{result.opc1}, {_reg(result.t)}, {_reg(result.t2)}, c{result.CRm}"  # noqa: E501


def _fmt_mrrc_mrrc2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    return f"mrrc{suffix} p{result.cp}, #{result.opc1}, {_reg(result.t)}, {_reg(result.t2)}, c{result.CRm}"  # noqa: E501


def _fmt_stc_stc2(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    addr = _addr_imm(result.n, result.imm32, result.index, result.add, result.wback)
    return f"stc{suffix} p{result.cp}, c{result.CRd}, {addr}"


def _fmt_ldc_ldc2_imm(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    addr = _addr_imm(result.n, result.imm32, result.index, result.add, result.wback)
    return f"ldc{suffix} p{result.cp}, c{result.CRd}, {addr}"


def _fmt_ldc_ldc2_lit(result: Any, instr: int = 0) -> str:
    suffix = "2" if _is_coproc2(instr) else ""
    sign = "" if result.add else "-"
    return f"ldc{suffix} p{result.cp}, c{result.CRd}, [pc, #{sign}{result.imm32}]"


# ---------------------------------------------------------------------------
# Shadow aliases (CPY, NEG, MOV_shifted_register)
# ---------------------------------------------------------------------------


def _fmt_cpy(result: Any) -> str:
    return f"cpy {_reg(result.d)}, {_reg(result.m)}"


def _fmt_neg(result: Any) -> str:
    return f"neg {_reg(result.d)}, {_reg(result.m)}"


def _fmt_mov_shifted(result: Any) -> str:
    s = _flags(result.setflags)
    sh = _shift(result.shift_t, result.shift_n)
    return f"mov{s} {_reg(result.d)}, {_reg(result.m)}{sh}"


# ---------------------------------------------------------------------------
# Dispatch table — maps class name → (formatter_fn, extra_args)
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Any] = {
    "ADC_immediate": lambda r: _fmt_dp_imm(r, "adc"),
    "ADC_register": lambda r: _fmt_dp_reg(r, "adc"),
    "ADD_immediate": lambda r: _fmt_dp_imm(r, "add"),
    "ADD_register": lambda r: _fmt_dp_reg(r, "add"),
    "ADD_SP_plus_immediate": _fmt_add_sp_imm,
    "ADD_SP_plus_register": _fmt_add_sp_reg,
    "ADR": _fmt_adr,
    "AND_immediate": _fmt_and_imm,
    "AND_register": lambda r: _fmt_dp_reg(r, "and"),
    "ASR_immediate": lambda r: _fmt_shift_imm(r, "asr"),
    "ASR_register": lambda r: _fmt_shift_reg(r, "asr"),
    "B": _fmt_b,
    "BFC": _fmt_bfc,
    "BFI": _fmt_bfi,
    "BIC_immediate": lambda r: _fmt_and_imm(r, "bic"),
    "BIC_register": lambda r: _fmt_dp_reg(r, "bic"),
    "BKPT": _fmt_bkpt,
    "BL": _fmt_bl,
    "BLX_register": _fmt_blx_reg,
    "BX": _fmt_bx,
    "CBNZ_CBZ": _fmt_cbnz_cbz,
    "CDP_CDP2": _fmt_cdp_cdp2,
    "CLREX": lambda r: "clrex",
    "CLZ": _fmt_clz,
    "CMN_immediate": lambda r: _fmt_test_imm(r, "cmn"),
    "CMN_register": lambda r: _fmt_test_reg(r, "cmn"),
    "CMP_immediate": lambda r: _fmt_test_imm(r, "cmp"),
    "CMP_register": lambda r: _fmt_test_reg(r, "cmp"),
    "CPS": _fmt_cps,
    "CPY": _fmt_cpy,
    "CSDB": lambda r: "csdb",
    "DBG": _fmt_db,
    "DMB": lambda r: _fmt_barrier(r, "dmb"),
    "DSB": lambda r: _fmt_barrier(r, "dsb"),
    "EOR_immediate": lambda r: _fmt_and_imm(r, "eor"),
    "EOR_register": lambda r: _fmt_dp_reg(r, "eor"),
    "ISB": lambda r: _fmt_barrier(r, "isb"),
    "IT": _fmt_it,
    "LDC_LDC2_immediate": _fmt_ldc_ldc2_imm,
    "LDC_LDC2_literal": _fmt_ldc_ldc2_lit,
    "LDM": lambda r: _fmt_multi_xfer(r, "ldmia"),
    "LDMDB": lambda r: _fmt_multi_xfer(r, "ldmdb"),
    "LDR_immediate": lambda r: _fmt_ldst_imm_t(r, "ldr"),
    "LDR_literal": lambda r: _fmt_ldst_lit(r, "ldr"),
    "LDR_register": lambda r: _fmt_ldst_reg(r, "ldr"),
    "LDRB_immediate": lambda r: _fmt_ldst_imm_t(r, "ldrb"),
    "LDRB_literal": lambda r: _fmt_ldst_lit(r, "ldrb"),
    "LDRB_register": lambda r: _fmt_ldst_reg(r, "ldrb"),
    "LDRBT": lambda r: _fmt_unpriv_ldr(r, "ldrbt"),
    "LDRD_immediate": lambda r: _fmt_ldst_imm_dual(r, "ldrd"),
    "LDRD_literal": _fmt_ldst_lit_dual,
    "LDREX": _fmt_ldrex,
    "LDREXB": _fmt_ldrexb,
    "LDREXH": _fmt_ldrexh,
    "LDRH_immediate": lambda r: _fmt_ldst_imm_t(r, "ldrh"),
    "LDRH_literal": lambda r: _fmt_ldst_lit(r, "ldrh"),
    "LDRH_register": lambda r: _fmt_ldst_reg(r, "ldrh"),
    "LDRHT": lambda r: _fmt_unpriv_ldr(r, "ldrht"),
    "LDRSB_immediate": lambda r: _fmt_ldst_imm_t(r, "ldrsb"),
    "LDRSB_literal": lambda r: _fmt_ldst_lit(r, "ldrsb"),
    "LDRSB_register": lambda r: _fmt_ldst_reg(r, "ldrsb"),
    "LDRSBT": lambda r: _fmt_unpriv_ldr(r, "ldrsbt"),
    "LDRSH_immediate": lambda r: _fmt_ldst_imm_t(r, "ldrsh"),
    "LDRSH_literal": lambda r: _fmt_ldst_lit(r, "ldrsh"),
    "LDRSH_register": lambda r: _fmt_ldst_reg(r, "ldrsh"),
    "LDRSHT": lambda r: _fmt_unpriv_ldr(r, "ldrsht"),
    "LDRT": lambda r: _fmt_unpriv_ldr(r, "ldrt"),
    "LSL_immediate": lambda r: _fmt_shift_imm(r, "lsl"),
    "LSL_register": lambda r: _fmt_shift_reg(r, "lsl"),
    "LSR_immediate": lambda r: _fmt_shift_imm(r, "lsr"),
    "LSR_register": lambda r: _fmt_shift_reg(r, "lsr"),
    "MCR_MCR2": _fmt_mcr_mcr2,
    "MCRR_MCRR2": _fmt_mcrr_mcrr2,
    "MLA": lambda r: _fmt_mla(r, getattr(r, "setflags", False)),
    "MLS": _fmt_mls,
    "MOV_immediate": _fmt_mov_imm,
    "MOV_register": _fmt_mov_reg,
    "MOV_shifted_register": _fmt_mov_shifted,
    "MOVT": _fmt_movt,
    "MRC_MRC2": _fmt_mrc_mrc2,
    "MRRC_MRRC2": _fmt_mrrc_mrrc2,
    "MRS": _fmt_mrs,
    "MSR": _fmt_msr,
    "MUL": lambda r: _fmt_mul(r, getattr(r, "setflags", False)),
    "MVN_immediate": _fmt_mvn_imm,
    "MVN_register": _fmt_mvn_reg,
    "NEG": _fmt_neg,
    "NOP": lambda r: "nop",
    "ORN_immediate": lambda r: _fmt_and_imm(r, "orn"),
    "ORN_register": lambda r: _fmt_dp_reg(r, "orn"),
    "ORR_immediate": lambda r: _fmt_and_imm(r, "orr"),
    "ORR_register": lambda r: _fmt_dp_reg(r, "orr"),
    "PKHBT_PKHTB": _fmt_pkhbt_pkhtb,
    "PLD_immediate": _fmt_pld_imm,
    "PLD_literal": _fmt_pld_lit,
    "PLD_register": _fmt_pld_reg,
    "PLI_immediate_literal": _fmt_pli_imm_lit,
    "PLI_register": _fmt_pli_reg,
    "POP": _fmt_pop,
    "PSSBB": lambda r: "pssbb",
    "PUSH": _fmt_push,
    "QADD": lambda r: _fmt_simd3(r, "qadd"),
    "QADD16": lambda r: _fmt_simd3(r, "qadd16"),
    "QADD8": lambda r: _fmt_simd3(r, "qadd8"),
    "QASX": lambda r: _fmt_simd3(r, "qasx"),
    "QDADD": lambda r: _fmt_simd3(r, "qdadd"),
    "QDSUB": lambda r: _fmt_simd3(r, "qdsub"),
    "QSAX": lambda r: _fmt_simd3(r, "qsax"),
    "QSUB": lambda r: _fmt_simd3(r, "qsub"),
    "QSUB16": lambda r: _fmt_simd3(r, "qsub16"),
    "QSUB8": lambda r: _fmt_simd3(r, "qsub8"),
    "RBIT": _fmt_rbit,
    "REV": _fmt_rev,
    "REV16": _fmt_rev16,
    "REVSH": _fmt_revsh,
    "ROR_immediate": _fmt_ror_imm,
    "ROR_register": lambda r: _fmt_shift_reg(r, "ror"),
    "RRX": _fmt_rrx,
    "RSB_immediate": lambda r: _fmt_dp_imm(r, "rsb"),
    "RSB_register": lambda r: _fmt_dp_reg(r, "rsb"),
    "SADD16": lambda r: _fmt_simd3(r, "sadd16"),
    "SADD8": lambda r: _fmt_simd3(r, "sadd8"),
    "SASX": lambda r: _fmt_simd3(r, "sasx"),
    "SBC_immediate": lambda r: _fmt_dp_imm(r, "sbc"),
    "SBC_register": lambda r: _fmt_dp_reg(r, "sbc"),
    "SBFX": _fmt_sbfx,
    "SDIV": _fmt_sdiv,
    "SEL": _fmt_sel,
    "SEV": lambda r: "sev",
    "SHADD16": lambda r: _fmt_simd3(r, "shadd16"),
    "SHADD8": lambda r: _fmt_simd3(r, "shadd8"),
    "SHASX": lambda r: _fmt_simd3(r, "shasx"),
    "SHSAX": lambda r: _fmt_simd3(r, "shsax"),
    "SHSUB16": lambda r: _fmt_simd3(r, "shsub16"),
    "SHSUB8": lambda r: _fmt_simd3(r, "shsub8"),
    "SMLABB_SMLABT_SMLATB_SMLATT": _fmt_smlabb_variants,
    "SMLAD_SMLADX": _fmt_smlad_variants,
    "SMLAL": lambda r: _fmt_smlal(r, getattr(r, "setflags", False)),
    "SMLALBB_SMLALBT_SMLALTB_SMLALTT": _fmt_smlalbb_variants,
    "SMLALD_SMLALDX": _fmt_smlald_variants,
    "SMLAWB_SMLAWT": _fmt_smlaw_variants,
    "SMLSD_SMLSDX": _fmt_smlsd_variants,
    "SMLSLD_SMLSLDX": _fmt_smlsld_variants,
    "SMMLA_SMMLAR": _fmt_smmla_variants,
    "SMMLS_SMMLSR": _fmt_smmls_variants,
    "SMMUL_SMMULR": _fmt_smmul_variants,
    "SMUAD_SMUADX": _fmt_smuad_variants,
    "SMULBB_SMULBT_SMULTB_SMULTT": _fmt_smulbb_variants,
    "SMULL": lambda r: _fmt_smull(r, getattr(r, "setflags", False)),
    "SMULWB_SMULWT": _fmt_smulw_variants,
    "SMUSD_SMUSDX": _fmt_smusd_variants,
    "SSAT": _fmt_ssat,
    "SSAT16": _fmt_ssat16,
    "SSAX": lambda r: _fmt_simd3(r, "ssax"),
    "SSBB": lambda r: "ssbb",
    "SSUB16": lambda r: _fmt_simd3(r, "ssub16"),
    "SSUB8": lambda r: _fmt_simd3(r, "ssub8"),
    "STC_STC2": _fmt_stc_stc2,
    "STM": lambda r: _fmt_multi_xfer(r, "stmia"),
    "STMDB": lambda r: _fmt_multi_xfer(r, "stmdb"),
    "STR_immediate": lambda r: _fmt_ldst_imm_t(r, "str"),
    "STR_register": lambda r: _fmt_ldst_reg(r, "str"),
    "STRB_immediate": lambda r: _fmt_ldst_imm_t(r, "strb"),
    "STRB_register": lambda r: _fmt_ldst_reg(r, "strb"),
    "STRBT": lambda r: _fmt_unpriv_str(r, "strbt"),
    "STRD_immediate": lambda r: _fmt_ldst_imm_dual(r, "strd"),
    "STREX": _fmt_strex,
    "STREXB": _fmt_strexb,
    "STREXH": _fmt_strexh,
    "STRH_immediate": lambda r: _fmt_ldst_imm_t(r, "strh"),
    "STRH_register": lambda r: _fmt_ldst_reg(r, "strh"),
    "STRHT": lambda r: _fmt_unpriv_str(r, "strht"),
    "STRT": lambda r: _fmt_unpriv_str(r, "strt"),
    "SUB_immediate": lambda r: _fmt_dp_imm(r, "sub"),
    "SUB_register": lambda r: _fmt_dp_reg(r, "sub"),
    "SUB_SP_minus_immediate": _fmt_sub_sp_imm,
    "SUB_SP_minus_register": _fmt_sub_sp_reg,
    "SVC": _fmt_svc,
    "SXTAB": lambda r: _fmt_extab(r, "sxtab"),
    "SXTAB16": lambda r: _fmt_extab(r, "sxtab16"),
    "SXTAH": lambda r: _fmt_extab(r, "sxtah"),
    "SXTB": lambda r: _fmt_extend(r, "sxtb"),
    "SXTB16": lambda r: _fmt_extend(r, "sxtb16"),
    "SXTH": lambda r: _fmt_extend(r, "sxth"),
    "TBB_TBH": _fmt_tbb_tbh,
    "TEQ_immediate": lambda r: _fmt_test_imm(r, "teq"),
    "TEQ_register": lambda r: _fmt_test_reg(r, "teq"),
    "TST_immediate": lambda r: _fmt_test_imm(r, "tst"),
    "TST_register": lambda r: _fmt_test_reg(r, "tst"),
    "UADD16": lambda r: _fmt_simd3(r, "uadd16"),
    "UADD8": lambda r: _fmt_simd3(r, "uadd8"),
    "UASX": lambda r: _fmt_simd3(r, "uasx"),
    "UBFX": _fmt_ubfx,
    "UDF": _fmt_udf,
    "UDIV": _fmt_udiv,
    "UHADD16": lambda r: _fmt_simd3(r, "uhadd16"),
    "UHADD8": lambda r: _fmt_simd3(r, "uhadd8"),
    "UHASX": lambda r: _fmt_simd3(r, "uhasx"),
    "UHSAX": lambda r: _fmt_simd3(r, "uhsax"),
    "UHSUB16": lambda r: _fmt_simd3(r, "uhsub16"),
    "UHSUB8": lambda r: _fmt_simd3(r, "uhsub8"),
    "UMAAL": _fmt_umaal,
    "UMLAL": lambda r: _fmt_umlal(r, getattr(r, "setflags", False)),
    "UMULL": lambda r: _fmt_umull(r, getattr(r, "setflags", False)),
    "UQADD16": lambda r: _fmt_simd3(r, "uqadd16"),
    "UQADD8": lambda r: _fmt_simd3(r, "uqadd8"),
    "UQASX": lambda r: _fmt_simd3(r, "uqasx"),
    "UQSAX": lambda r: _fmt_simd3(r, "uqsax"),
    "UQSUB16": lambda r: _fmt_simd3(r, "uqsub16"),
    "UQSUB8": lambda r: _fmt_simd3(r, "uqsub8"),
    "USAD8": _fmt_usad8,
    "USADA8": _fmt_usada8,
    "USAT": _fmt_usat,
    "USAT16": _fmt_usat16,
    "USAX": lambda r: _fmt_simd3(r, "usax"),
    "USUB16": lambda r: _fmt_simd3(r, "usub16"),
    "USUB8": lambda r: _fmt_simd3(r, "usub8"),
    "UXTAB": lambda r: _fmt_extab(r, "uxtab"),
    "UXTAB16": lambda r: _fmt_extab(r, "uxtab16"),
    "UXTAH": lambda r: _fmt_extab(r, "uxtah"),
    "UXTB": lambda r: _fmt_extend(r, "uxtb"),
    "UXTB16": lambda r: _fmt_extend(r, "uxtb16"),
    "UXTH": lambda r: _fmt_extend(r, "uxth"),
    "VABS": lambda r: _fmt_vfp_dp2(r, "vabs"),
    "VADD": lambda r: _fmt_vfp_dp3(r, "vadd"),
    "VCMP_VCMPE": _fmt_vcmp_vcmpe,
    "VCVTA_VCVTN_VCVTP_VCVTM": _fmt_vcvta_round,
    "VCVT_VCVTR_integer": _fmt_vcvt_integer,
    "VCVT_fixed_point": _fmt_vcvt_fixed,
    "VCVT_double_single": _fmt_vcvt_double_single,
    "VCVTB_VCVTT": _fmt_vcvtb_vcvtt,
    "VDIV": lambda r: _fmt_vfp_dp3(r, "vdiv"),
    "VFMA_VFMS": _fmt_vfma_vfms,
    "VFNMA_VFNMS": _fmt_vfnma_vfnms,
    "VLDM": lambda r: _fmt_vldm_vstm(r, "vldm"),
    "VLDR": lambda r: _fmt_vldr_vstr(r, "vldr"),
    "VMAXNM_VMINNM": _fmt_vmaxnm_vminnm,
    "VMLA_VMLS": _fmt_vmla_vmls,
    "VMOV_immediate": _fmt_vmov_imm,
    "VMOV_register": _fmt_vmov_reg,
    "VMOV_core_to_scalar": _fmt_vmov_core_to_scalar,
    "VMOV_scalar_to_core": _fmt_vmov_scalar_to_core,
    "VMOV_core_and_single": _fmt_vmov_core_and_single,
    "VMOV_two_core_and_two_single": _fmt_vmov_two_core_and_single,
    "VMOV_two_core_and_doubleword": _fmt_vmov_two_core_and_doubleword,
    "VMRS": _fmt_vmrs,
    "VMSR": _fmt_vmsr,
    "VMUL": lambda r: _fmt_vfp_dp3(r, "vmul"),
    "VNEG": lambda r: _fmt_vfp_dp2(r, "vneg"),
    "VNMLA_VNMLS_VNMUL": _fmt_vnmla_vnmls_vnmul,
    "VPOP": lambda r: _fmt_vpush_vpop(r, "vpop"),
    "VPUSH": lambda r: _fmt_vpush_vpop(r, "vpush"),
    "VRINTA_VRINTN_VRINTP_VRINTM": _fmt_vrint_round,
    "VRINTX": _fmt_vrintx,
    "VRINTZ_VRINTR": _fmt_vrint_zr,
    "VSEL": _fmt_vsel,
    "VSQRT": lambda r: _fmt_vfp_dp2(r, "vsqrt"),
    "VSTM": lambda r: _fmt_vldm_vstm(r, "vstm"),
    "VSTR": lambda r: _fmt_vldr_vstr(r, "vstr"),
    "VSUB": lambda r: _fmt_vfp_dp3(r, "vsub"),
    "WFE": lambda r: "wfe",
    "WFI": lambda r: "wfi",
    "YIELD": lambda r: "yield",
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def disassemble(result: object, instr: int = 0, offset: int = 0) -> str:
    """Convert a decoded instruction dataclass to an assembler mnemonic string.

    Args:
        result: Decoded instruction dataclass instance (or pseudo-instruction).
        instr: Raw 32-bit instruction word (needed to disambiguate coprocessor
               variants like MCR vs MCR2).
        offset: Address of the instruction in memory (for computing absolute
                branch targets).

    Returns:
        UAL assembler syntax string, e.g. ``"adds r0, r1, #42"``.
    """
    if isinstance(result, NoMatch):
        return f"<nomatch {hex(result.code)}>"
    if isinstance(result, Undefined):
        return f"<undefined {hex(result.code)}>"
    if isinstance(result, Unpredictable):
        return f"<unpredictable {hex(result.code)}>"
    if isinstance(result, See):
        return f"<see {hex(result.code)}>"

    cls = type(result).__name__
    fmt_func = _DISPATCH.get(cls)

    if fmt_func is None:
        return repr(result)

    return fmt_func(result)

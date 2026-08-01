"""ITSTATE tracking for the ARMv7-M IT (If-Then) instruction.

An ``IT`` instruction makes up to four following instructions conditional. The
condition of each of those instructions is not in its own encoding -- it lives
in ITSTATE, which ``IT`` loads and every instruction afterwards shifts along.
Decoding a stream therefore needs the same running state a core keeps:
ITSTATE<7:0> (``firstcond:mask``), held in ``Context.istate`` so the generated
decoder can read it through ``InITBlock`` / ``LastInITBlock``, and needed by
:func:`disassemble` to spell ``moveq`` rather than ``mov``.

Callers walking a stream keep that state up to date with::

    result, n_bytes = decode(instr, ctx)
    asm = disassemble(result, instr, offset, ctx.istate)
    ctx.istate = next_itstate(ctx.istate, result)

See the Armv7-M ARM (ARM DDI 0403E.e) B1.4.2 for ITSTATE, and A7.7.38 for IT.
"""

from __future__ import annotations

from armv7m_decoder._decoder import IT

# Condition code "always" -- what CurrentCond() reports outside an IT block.
COND_AL = 0xE


def in_it_block(istate: int) -> bool:
    """Whether ``istate`` places the next instruction inside an IT block."""
    return istate & 0b1111 != 0


def current_cond(istate: int) -> int:
    """Condition code the next instruction executes under.

    ``CurrentCond()`` in the ARM ARM: ITSTATE<7:4> inside an IT block, and
    ``COND_AL`` outside one. The architecture leaves an ITSTATE with a zero
    mask but a non-zero condition UNPREDICTABLE; a disassembler has nothing
    useful to say about it, so it reads as outside a block too.
    """
    if in_it_block(istate):
        return (istate >> 4) & 0b1111
    return COND_AL


def next_itstate(istate: int, result: object) -> int:
    """ITSTATE after ``result``, the instruction just decoded, has run.

    ``IT`` loads ITSTATE with its own ``firstcond:mask``; every other
    instruction consumes one slot of the block via ``ITAdvance()``.
    """
    if isinstance(result, IT):
        return ((result.firstcond & 0xF) << 4) | (result.mask & 0xF)
    if istate & 0b111 == 0:
        return 0
    return (istate & ~0b11111) | ((istate << 1) & 0b11111)

"""ITSTATE tracking tests: the state an IT block leaves for the instructions
that follow it."""

import pytest

from armv7m_decoder import COND_AL, current_cond, in_it_block, next_itstate

from .helpers import decode_word


@pytest.fixture
def it_ittee_gt(ctx):
    """Decoded ``ittee gt`` -- a block with all four slots in use."""
    result = decode_word(ctx, 0xBFC7)
    return result


class TestITState:
    def test_no_block_is_unconditional(self) -> None:
        assert not in_it_block(0)
        assert current_cond(0) == COND_AL

    def test_it_loads_firstcond_and_mask(self, it_ittee_gt) -> None:
        assert next_itstate(0, it_ittee_gt) == 0xC7

    def test_block_walks_its_conditions(self, ctx, it_ittee_gt) -> None:
        istate = next_itstate(0, it_ittee_gt)
        conds = []
        while in_it_block(istate):
            conds.append(current_cond(istate))
            nop = decode_word(ctx, 0xBF00)
            istate = next_itstate(istate, nop)
        # gt, gt, le, le -- "then, then, else, else".
        assert conds == [0xC, 0xC, 0xD, 0xD]
        assert istate == 0

    def test_single_slot_block_ends_at_once(self, ctx) -> None:
        it = decode_word(ctx, 0xBF08)  # it eq
        istate = next_itstate(0, it)
        assert current_cond(istate) == 0x0
        nop = decode_word(ctx, 0xBF00)
        assert next_itstate(istate, nop) == 0

    def test_advance_outside_a_block_is_a_no_op(self, ctx) -> None:
        nop = decode_word(ctx, 0xBF00)
        assert next_itstate(0, nop) == 0

    def test_zero_mask_reads_as_outside_a_block(self) -> None:
        # UNPREDICTABLE per the architecture; the disassembler treats the
        # instruction as unconditional rather than inventing a condition.
        assert not in_it_block(0xD0)
        assert current_cond(0xD0) == COND_AL

    def test_flagged_it_opens_no_block(self, ctx) -> None:
        """An IT the disassembler will not spell must not condition what follows.

        Before decoder-forge v5 a flagged side effect replaced the instruction,
        so a bad IT never reached here. It arrives intact now, and acting on
        the fields of a word spelled `<unpredictable>` would let one bad
        halfword condition the several after it.
        """
        # firstcond 0b1111 is UNPREDICTABLE, and the mask is non-zero, so this
        # would otherwise open a four-slot block.
        bad_it = decode_word(ctx, 0xBFF8)
        assert bad_it.mask != 0
        assert next_itstate(0, bad_it) == 0

        # mask 0000 is the hint space rather than an IT at all: SEE NOP.
        see_it = decode_word(ctx, 0xBF50)
        assert next_itstate(0, see_it) == 0

        # A clean IT still loads ITSTATE.
        assert next_itstate(0, decode_word(ctx, 0xBF08)) == 0x08

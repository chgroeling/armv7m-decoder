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

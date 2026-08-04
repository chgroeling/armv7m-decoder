"""Tests for the CLI, which is what writes the disassembly listing.

The disassembler spells an instruction; the CLI spells a line. Everything the
line holds beyond the mnemonic -- the address, the bytes, and what stands in
for a mnemonic where no encoding matched -- is settled here.
"""

from __future__ import annotations

import struct

from click.testing import CliRunner

from armv7m_decoder.cli import main


def run(tmp_path, halfwords: list[int]) -> list[str]:
    """Disassemble a Thumb stream through the CLI and return its lines."""
    bin_path = tmp_path / "firmware.bin"
    bin_path.write_bytes(b"".join(struct.pack("<H", hw) for hw in halfwords))
    result = CliRunner().invoke(main, [str(bin_path)])
    assert result.exit_code == 0, result.output
    return result.output.splitlines()


class TestNoMatch:
    def test_word_stands_in_for_the_mnemonic(self, tmp_path) -> None:
        # No encoding matches, so there is no mnemonic to spell -- objdump
        # writes the word itself as a comment, and so do we. A line is
        # mnemonic, tab, operands, tab, comment; with the first two empty the
        # comment is left holding both tabs, which is how objdump lands it in
        # the same column as any other comment.
        assert run(tmp_path, [0xF2E5, 0x3EFF]) == [
            "       0:\tf2e5 3eff \t\t\t@ <UNDEFINED> instruction: 0xf2e53eff",
        ]

    def test_narrow_word_is_a_halfword_wide(self, tmp_path) -> None:
        assert run(tmp_path, [0xB717]) == [
            "       0:\tb717      \t\t\t@ <UNDEFINED> instruction: 0xb717",
        ]

    def test_a_matched_word_lands_its_comment_in_the_same_column(
        self, tmp_path
    ) -> None:
        assert run(tmp_path, [0xF20D, 0x154F]) == [
            "       0:\tf20d 154f \taddw\tr5, sp, #335\t@ 0x14f",
        ]

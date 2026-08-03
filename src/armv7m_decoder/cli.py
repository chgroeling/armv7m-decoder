"""CLI entry point for armv7m-decoder.

Usage::

    armv7m-decoder decode firmware.bin --start-address 0xD4
"""

import sys
from pathlib import Path
from typing import Optional

import click

from armv7m_decoder import (
    Context,
    Opcode,
    disassemble,
    fetch_and_decode,
    next_itstate,
)


def parse_address(_ctx, _param, value: str) -> int:
    try:
        return int(value, 0)
    except ValueError:
        raise click.BadParameter(f"{value!r} is not a valid address")


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("BIN_PATH", type=str)
@click.option(
    "--start-address",
    default="0x0",
    callback=parse_address,
    help="Offset into BIN_PATH at which decoding starts (default: 0x0)",
)
@click.option(
    "--out-file",
    default=None,
    type=str,
    help="Output file (defaults to stdout)",
)
@click.option(
    "--max-instructions",
    default=None,
    type=int,
    help="Maximum number of instructions to decode",
)
def decode_cmd(
    bin_path: str,
    start_address: int,
    out_file: Optional[str],
    max_instructions: Optional[int],
) -> None:
    """Decode a binary file using the ARMv7-M instruction decoder."""
    data = Path(bin_path).read_bytes()

    ctx = Context()

    out = open(out_file, "w", encoding="utf-8") if out_file else sys.stdout

    offset = start_address
    total = 0
    while True:
        if max_instructions is not None and total >= max_instructions:
            break

        # One word out of the buffer, decoded at the size its first halfword
        # settles -- ``None`` once what is left is not a whole instruction.
        word = fetch_and_decode(data, offset, ctx)
        if word is None:
            break

        # A decode only reads the context, so this is still the ITSTATE the word
        # decoded under: it tells the disassembler to spell ``moveq`` rather than
        # ``mov``, and the next state advances from it.
        istate = ctx.istate
        result = word.instruction
        hex_bytes = " ".join(f"{hw:04x}" for hw in word.halfwords)
        asm = disassemble(result, word.size, offset, istate)
        if result.opcode == Opcode.OP_NO_MATCH:
            # A word no encoding matches has no mnemonic to spell, so the whole
            # field is a comment carrying the word itself, as objdump writes
            # it. A line is mnemonic, tab, operands, tab, comment; with the
            # first two empty the comment is left holding both tabs, which is
            # how objdump lands it in the same column as any other comment:
            #
            #     2:\tf20d 154f \taddw\tr5, sp, #335\t@ 0x14f
            #     6:\tf2e5 3eff \t\t\t@ <UNDEFINED> instruction: 0xf2e53eff
            asm = f"\t\t@ <UNDEFINED> instruction: 0x{word.word:0{word.size // 4}x}"
        out.write(f"{offset:8x}:\t{hex_bytes:<10}\t{asm}\n")
        ctx.istate = next_itstate(istate, result)

        offset += word.n_bytes
        total += 1

    if out_file:
        out.close()


if __name__ == "__main__":
    main()

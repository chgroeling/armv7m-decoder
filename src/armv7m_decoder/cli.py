"""CLI entry point for armv7m-decoder.

Usage::

    armv7m-decoder decode firmware.bin --start-address 0xD4
"""

import struct
import sys
from pathlib import Path
from typing import Optional

import click

from armv7m_decoder import (
    Context,
    InstructionSize,
    decode,
    disassemble,
    instr_size,
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
    while offset + 2 <= len(data):
        if max_instructions is not None and total >= max_instructions:
            break

        # The first halfword settles the size, whether or not an encoding
        # matches -- skipping only half of a 32-bit word would leave the
        # second halfword to be disassembled as an instruction of its own.
        hw1 = struct.unpack_from("<H", data, offset)[0]
        size = instr_size(hw1)
        n_bytes = size // 8
        if offset + n_bytes > len(data):
            break

        if size == InstructionSize.SIZE_32BIT:
            hw2 = struct.unpack_from("<H", data, offset + 2)[0]
            instr = (hw1 << 16) | hw2
            hex_bytes = f"{hw1:04x} {hw2:04x}"
        else:
            instr = hw1
            hex_bytes = f"{hw1:04x}"

        # ITSTATE reaches the decoder through ``ctx`` -- 16-bit data processing
        # inside an IT block decodes without the S bit -- and tells the
        # disassembler to spell ``moveq`` rather than ``mov``.
        istate = ctx.istate
        result = decode(instr, ctx, size)
        asm = disassemble(result, instr, size, offset, istate)
        out.write(f"{offset:8x}:\t{hex_bytes:<10}\t{asm}\n")
        ctx.istate = next_itstate(istate, result)

        offset += n_bytes
        total += 1

    if out_file:
        out.close()


if __name__ == "__main__":
    main()

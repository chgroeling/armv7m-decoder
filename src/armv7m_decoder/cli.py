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
    decode,
    decoded_bytes,
    disassemble,
    get_decoder_eval_bytes,
    get_min_instr_bytes,
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

    eval_bytes = get_decoder_eval_bytes()
    min_bytes = get_min_instr_bytes()

    ctx = Context()

    out = open(out_file, "w", encoding="utf-8") if out_file else sys.stdout

    offset = start_address
    total = 0
    while offset < len(data):
        if max_instructions is not None and total >= max_instructions:
            break

        remaining = data[offset:]
        if len(remaining) < min_bytes:
            break

        buf = remaining[:eval_bytes]
        if len(buf) < eval_bytes:
            buf = buf + b"\x00" * (eval_bytes - len(buf))

        hw1 = struct.unpack("<H", buf[0:2])[0]
        hw2 = struct.unpack("<H", buf[2:4])[0]
        instr = (hw1 << 16) | hw2

        # ITSTATE reaches the decoder through ``ctx`` -- 16-bit data processing
        # inside an IT block decodes without the S bit -- and tells the
        # disassembler to spell ``moveq`` rather than ``mov``.
        istate = ctx.istate
        result, n_bytes = decode(instr, ctx)
        # A word the decoder matched nothing for still has the length Thumb
        # gives it -- skipping only half of a 32-bit one would disassemble its
        # second halfword as an instruction.
        n_bytes = decoded_bytes(instr, result, n_bytes)

        if result is not None:
            if n_bytes == 2:
                hex_bytes = f"{hw1:04x}"
                instr_clean = hw1 << 16
            else:
                hex_bytes = f"{hw1:04x} {hw2:04x}"
                instr_clean = instr
            asm = disassemble(result, instr_clean, offset, istate)
            out.write(f"{offset:8x}:\t{hex_bytes:<10}\t{asm}\n")
            ctx.istate = next_itstate(istate, result)

        offset += n_bytes
        total += 1

    if out_file:
        out.close()


if __name__ == "__main__":
    main()

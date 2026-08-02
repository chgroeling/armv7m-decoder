"""Public API for the ARMv7-M instruction decoder.

Provides the :func:`decode` entry point, the :class:`Context` for threading runtime
state, the :func:`disassemble` function for assembler formatting, the ITSTATE
helpers that carry an IT block's condition across a stream
(:func:`next_itstate`, :func:`current_cond`, :func:`in_it_block`), the
instruction-size helpers that settle how wide a word is before it is decoded
(:func:`instr_size`, :func:`instr_bytes`), the pseudo-instruction classes
(:class:`NoMatch`, :class:`Undefined`, :class:`Unpredictable`, :class:`See`),
and all instruction dataclasses.
"""

from armv7m_decoder._decoder import (  # noqa: F401
    Context,
    Encoding,
    InstructionSize,
    NoMatch,
    Opcode,
    See,
    Undefined,
    Unpredictable,
    decode,
    decode_16bit,
    decode_32bit,
    get_supported_sizes,
)
from armv7m_decoder._disasm import disassemble  # noqa: F401
from armv7m_decoder._itstate import (  # noqa: F401
    COND_AL,
    current_cond,
    in_it_block,
    next_itstate,
)
from armv7m_decoder._size import instr_bytes, instr_size  # noqa: F401

__all__ = [
    "COND_AL",
    "Context",
    "Encoding",
    "InstructionSize",
    "NoMatch",
    "Opcode",
    "See",
    "Undefined",
    "Unpredictable",
    "current_cond",
    "decode",
    "decode_16bit",
    "decode_32bit",
    "disassemble",
    "get_supported_sizes",
    "in_it_block",
    "instr_bytes",
    "instr_size",
    "next_itstate",
]

# Re-export all instruction dataclasses from the generated decoder.  An instruction
# dataclass is identified by a non-negative ``opcode`` ClassVar.
from armv7m_decoder import _decoder  # noqa: E402

_opcode_to_name: dict[int, str] = {}
for _name, _obj in vars(_decoder).items():
    if _name.startswith("_"):
        continue
    _opc = getattr(_obj, "opcode", None)
    if isinstance(_opc, int) and _opc >= 0:
        _opcode_to_name[_opc] = _name
        globals()[_name] = _obj
        __all__.append(_name)

# Export a lookup table for reverse-mapping opcodes to class names.
__all__.append("_opcode_to_name")

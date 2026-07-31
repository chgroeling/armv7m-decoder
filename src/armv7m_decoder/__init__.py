"""Public API for the ARMv7-M instruction decoder.

Provides the :func:`decode` entry point, the :class:`Context` for threading runtime
state, the :func:`disassemble` function for assembler formatting, the
pseudo-instruction classes (:class:`NoMatch`, :class:`Undefined`,
:class:`Unpredictable`, :class:`See`), and all instruction dataclasses.
"""

from armv7m_decoder._decoder import (  # noqa: F401
    Context,
    DecoderState,
    NoMatch,
    Opcode,
    See,
    Undefined,
    Unpredictable,
    decode,
    get_decoder_eval_bytes,
    get_min_instr_bytes,
)
from armv7m_decoder._disasm import disassemble  # noqa: F401

__all__ = [
    "Context",
    "DecoderState",
    "NoMatch",
    "Opcode",
    "See",
    "Undefined",
    "Unpredictable",
    "decode",
    "disassemble",
    "get_decoder_eval_bytes",
    "get_min_instr_bytes",
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

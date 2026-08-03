"""Public API for the ARMv7-M instruction decoder.

Decoding starts at :func:`fetch_and_decode` for a word still in a byte buffer, or
:func:`decode_word` for one the caller already has in hand; both settle the size
themselves and report a :class:`DecodedWord`. Also provides the :class:`Context`
for threading runtime state, the :func:`disassemble` function for assembler
formatting, :func:`current_cond` and ``COND_AL`` for the condition an ITSTATE
puts the next instruction under, :class:`NoMatch` for a word no encoding
matches, the ``SIDEFFECT_*`` flags an instruction reports on its ``sideeffects``
member, and all instruction dataclasses.
"""

from armv7m_decoder._decoder import (  # noqa: F401
    SIDEFFECT_NONE,
    SIDEFFECT_SEE,
    SIDEFFECT_UNDEFINED,
    SIDEFFECT_UNPREDICTABLE,
    Context,
    Encoding,
    InstructionSize,
    NoMatch,
    Opcode,
    get_supported_sizes,
)
from armv7m_decoder._disasm import disassemble  # noqa: F401
from armv7m_decoder._itstate import COND_AL, current_cond  # noqa: F401
from armv7m_decoder._word import (  # noqa: F401
    DecodedWord,
    decode_word,
    fetch_and_decode,
)

__all__ = [
    "COND_AL",
    "SIDEFFECT_NONE",
    "SIDEFFECT_SEE",
    "SIDEFFECT_UNDEFINED",
    "SIDEFFECT_UNPREDICTABLE",
    "Context",
    "DecodedWord",
    "Encoding",
    "InstructionSize",
    "NoMatch",
    "Opcode",
    "current_cond",
    "decode_word",
    "disassemble",
    "fetch_and_decode",
    "get_supported_sizes",
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

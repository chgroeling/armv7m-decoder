"""Public API for the ARMv7-M instruction decoder.

Provides the :func:`decode` entry point, the :class:`Context` for threading runtime
state, the pseudo-instruction classes (:class:`NoMatch`, :class:`Undefined`,
:class:`Unpredictable`, :class:`See`), and all instruction dataclasses.
"""

from armv7m_decoder._decoder import (  # noqa: F401
    Context,
    NoMatch,
    See,
    Undefined,
    Unpredictable,
    decode,
    get_decoder_eval_bytes,
    get_min_instr_bytes,
)

__all__ = [
    "Context",
    "NoMatch",
    "See",
    "Undefined",
    "Unpredictable",
    "decode",
    "get_decoder_eval_bytes",
    "get_min_instr_bytes",
]

# Re-export all instruction dataclasses from the generated decoder.  An instruction
# dataclass is identified by a non-negative ``_id`` ClassVar.
from armv7m_decoder import _decoder  # noqa: E402

_id_to_name: dict[int, str] = {}
for _name, _obj in vars(_decoder).items():
    if _name.startswith("_"):
        continue
    _id = getattr(_obj, "_id", None)
    if isinstance(_id, int) and _id >= 0:
        _id_to_name[_id] = _name
        globals()[_name] = _obj
        __all__.append(_name)

# Export a lookup table for reverse-mapping instruction ids to class names.
__all__.append("_id_to_name")

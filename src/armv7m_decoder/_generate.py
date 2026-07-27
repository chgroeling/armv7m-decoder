"""Regeneration script: reads the ARMv7-M format YAML and calls decoder-forge
to produce an up-to-date ``_decoder.py``.

Usage::

    uv run python -m armv7m_decoder._generate
"""

import sys
from pathlib import Path

from decoder_forge.generate_code import generate_code
from decoder_forge.template_engine import TemplateEngine

FORMAT_YAML = Path(__file__).resolve().parent.parent.parent / "formats" / "armv7-m.yaml"
OUTPUT_FILE = Path(__file__).resolve().parent / "_decoder.py"
DECODER_WIDTH = 32


def main() -> None:
    yaml_text = FORMAT_YAML.read_text(encoding="utf-8")
    tengine = TemplateEngine()
    tengine.load("python")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        generate_code(yaml_text, DECODER_WIDTH, tengine, f)

    print(f"Generated {OUTPUT_FILE} from {FORMAT_YAML}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Split authored SOPs into retrievable chunks.

YOU WRITE THIS ONE. Chunking strategy is the single biggest lever on retrieval
quality, which means it is the thing your config sweep exists to measure.

Input:  data/documents/*.md, each with a front-matter header and numbered sections
Output: Chunk objects ready to embed and insert into doc_chunks
"""

from dataclasses import dataclass
from pathlib import Path

from src.config import cfg

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section: str | None
    content: str


def chunk_document(path: Path, chunk_size: int | None = None,
                   overlap: int | None = None) -> list[Chunk]:
    """Split one markdown SOP into chunks.

    TODO - recommended approach (section-aware, not blind character splitting):
      1. Parse the header for doc_id and doc_title.
      2. Split on section headings (## 3.2 Calibration Interval -> section "3.2").
      3. If a section exceeds chunk_size, split it further WITH overlap.
      4. Prepend the doc title and section number to each chunk's content so an
         isolated chunk still carries its context into the prompt.

    Why section-aware beats fixed-size splitting: your golden set cites specific
    sections (sop_calibration#3.2). If chunks align to sections, recall@k is
    measurable against a real ground truth. If they do not, your metric is mush.

    IMPORTANT - the 256-token ceiling:
      all-MiniLM-L6-v2 truncates its input at 256 tokens. A 512-token chunk does
      not produce a "richer" embedding; the second half is silently discarded.
      The chunk text is still handed to Claude in full, so the failure is
      invisible in the output and only shows up as unexplained bad recall.

      So: measure chunk_size in TOKENS with the model's own tokenizer, and sweep
      128 / 192 / 256. If you want to test larger chunks, switch to a
      longer-context embedding model and change vector(384) in the schema.

    Measure chunk_size 128 / 192 / 256 and overlap 0 / 32 in the sweep. Do not
    guess - report the numbers.
    """
    raise NotImplementedError("Week 3: implement chunking")


def chunk_all(chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Chunk every markdown file in data/documents/."""
    size = chunk_size or cfg.chunk_size
    lap = overlap if overlap is not None else cfg.chunk_overlap
    chunks: list[Chunk] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        chunks.extend(chunk_document(path, size, lap))
    return chunks

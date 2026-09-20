"""Split authored SOPs into retrievable chunks.

Chunking strategy is the single biggest lever on retrieval quality, which is
why the config sweep exists to measure it.

Design decisions, in the order they matter:

1. SECTION-AWARE, not fixed-size. The golden set cites `sop_calibration#3.2`.
   If chunks align to document sections, recall@k is measurable against real
   ground truth. If a chunk straddles two sections, the citation is ambiguous
   and the headline metric becomes meaningless.

2. TOKENS, not characters. The 4-chars-per-token rule of thumb is off by enough
   to push a chunk past the embedding model's ceiling without you noticing.

3. THE 256-TOKEN CEILING. all-MiniLM-L6-v2 truncates input at 256 tokens. A
   512-token chunk does not embed more text - the tail is silently discarded,
   while the full chunk still reaches Claude. Invisible in the output; shows up
   only as unexplained bad recall.

4. CONTEXT HEADER on every chunk. A retrieved chunk arrives stripped of its
   surroundings. Without a breadcrumb, "12 months" is an orphan number.

5. SPLIT ON PARAGRAPH, THEN SENTENCE. A chunk cut mid-sentence embeds badly -
   the vector encodes a fragment meaning something other than either neighbour.

Run `python -m src.ingest.chunk` to chunk, embed and load doc_chunks.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import cfg

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# ## 3. Calibration Requirements   /   ### 3.2 Calibration Interval
HEADING = re.compile(r"^(#{2,3})\s+(\d+(?:\.\d+)*)?\.?\s*(.+?)\s*$", re.MULTILINE)
PARAGRAPH = re.compile(r"\n\s*\n")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*\-])")


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section: str | None
    content: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id}#{self.section}" if self.section else self.doc_id


@dataclass
class Section:
    number: str | None
    title: str
    body: str


def _default_counter(text: str) -> int:
    """Token count via the embedding model's own tokenizer."""
    from src.ingest.embed import count_tokens

    return count_tokens(text)


def parse_document(path: Path) -> tuple[dict, list[Section]]:
    """Split one markdown SOP into frontmatter metadata plus flat sections."""
    raw = path.read_text(encoding="utf-8")

    match = FRONTMATTER.match(raw)
    if match is None:
        raise ValueError(f"{path.name}: missing YAML frontmatter (--- ... ---)")
    meta = yaml.safe_load(match.group(1)) or {}
    for key in ("doc_id", "doc_title"):
        if key not in meta:
            raise ValueError(f"{path.name}: frontmatter missing '{key}'")

    body = raw[match.end():]
    headings = list(HEADING.finditer(body))

    sections: list[Section] = []

    # Anything before the first ## heading (usually a preamble) is kept, so a
    # document with content there does not silently lose it.
    preamble_end = headings[0].start() if headings else len(body)
    preamble = _strip_h1(body[:preamble_end]).strip()
    if preamble:
        sections.append(Section(number=None, title=meta["doc_title"], body=preamble))

    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        text = body[heading.end():end].strip()
        if not text:
            continue  # a parent heading whose child heading follows immediately
        sections.append(
            Section(number=heading.group(2), title=heading.group(3), body=text)
        )

    return meta, sections


def _strip_h1(text: str) -> str:
    return re.sub(r"^#\s+.+?$", "", text, count=1, flags=re.MULTILINE)


def _units(body: str) -> list[str]:
    """Split a section body into paragraph units, then sentences within
    paragraphs that are themselves oversized. Markdown list blocks stay whole."""
    return [p.strip() for p in PARAGRAPH.split(body) if p.strip()]


def _hard_split(text: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """Last resort for a single unit larger than the whole budget: split on
    words. Only fires on pathological input, but without it such a unit would
    be emitted oversized and silently truncated at embedding time."""
    words = text.split()
    out: list[str] = []
    cur: list[str] = []
    for word in words:
        cur.append(word)
        if count(" ".join(cur)) > budget:
            cur.pop()
            if cur:
                out.append(" ".join(cur))
            cur = [word]
    if cur:
        out.append(" ".join(cur))
    return out


def _pack(body: str, budget: int, overlap: int, count: Callable[[str], int]) -> list[str]:
    """Pack units into chunks of at most `budget` tokens, with `overlap` carry."""
    if count(body) <= budget:
        return [body]

    units: list[str] = []
    for para in _units(body):
        if count(para) <= budget:
            units.append(para)
            continue
        # Oversized paragraph: fall back to sentences, then to words.
        for sentence in SENTENCE.split(para):
            if count(sentence) <= budget:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, budget, count))

    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    for unit in units:
        unit_tokens = count(unit)
        if cur and cur_tokens + unit_tokens > budget:
            chunks.append("\n\n".join(cur))
            # Carry the trailing units worth roughly `overlap` tokens.
            #
            # The allowance is capped so that tail + the incoming unit still fit
            # the budget. Without that cap, the flush check (which ran against
            # the OLD cur_tokens) lets tail_tokens + unit_tokens overflow, and
            # the oversized chunk is then silently truncated at embedding time.
            allowance = min(overlap, max(0, budget - unit_tokens))
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(cur):
                prev_tokens = count(prev)
                if tail_tokens + prev_tokens > allowance:
                    break
                tail.insert(0, prev)
                tail_tokens += prev_tokens
            cur, cur_tokens = tail, tail_tokens
        cur.append(unit)
        cur_tokens += unit_tokens

    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_document(
    path: Path,
    chunk_size: int | None = None,
    overlap: int | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Split one markdown SOP into chunks ready to embed."""
    budget = chunk_size or cfg.chunk_size
    lap = overlap if overlap is not None else cfg.chunk_overlap
    count = token_counter or _default_counter

    meta, sections = parse_document(path)
    doc_id, doc_title = meta["doc_id"], meta["doc_title"]

    chunks: list[Chunk] = []
    for section in sections:
        if section.number:
            header = f"{doc_title} - {section.number} {section.title}"
        elif section.title == doc_title:
            header = doc_title  # preamble; do not repeat the title twice
        else:
            header = f"{doc_title} - {section.title}"
        # The header lives inside the chunk, so it spends part of the budget.
        body_budget = budget - count(header) - 2
        if body_budget < 1:
            raise ValueError(f"{path.name}: chunk_size {budget} too small for header {header!r}")

        for piece in _pack(section.body, body_budget, lap, count):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section=section.number,
                    content=f"{header}\n\n{piece}",
                )
            )
    return chunks


def chunk_all(
    chunk_size: int | None = None,
    overlap: int | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Chunk every markdown file in data/documents/."""
    chunks: list[Chunk] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        chunks.extend(chunk_document(path, chunk_size, overlap, token_counter))
    return chunks


def build_index() -> None:
    """Chunk, embed and load doc_chunks. Destructive: replaces existing rows."""
    from src.db import connect
    from src.ingest.embed import embed_batch, max_input_tokens

    chunks = chunk_all()
    if not chunks:
        raise RuntimeError(f"No documents found in {DOCS_DIR}")

    ceiling = max_input_tokens()
    if cfg.chunk_size > ceiling:
        raise ValueError(
            f"CHUNK_SIZE={cfg.chunk_size} exceeds the embedding model's {ceiling}-token "
            f"limit. Chunks would be silently truncated before embedding."
        )

    print(f"Embedding {len(chunks)} chunks from {len(list(DOCS_DIR.glob('*.md')))} documents...")
    vectors = embed_batch([c.content for c in chunks])

    with connect() as conn:
        conn.execute("TRUNCATE doc_chunks RESTART IDENTITY")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO doc_chunks (doc_id, doc_title, section, content, token_count,"
                " embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (c.doc_id, c.doc_title, c.section, c.content, None, v)
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        conn.commit()
    print(f"Loaded {len(chunks)} chunks into doc_chunks.")


if __name__ == "__main__":
    build_index()

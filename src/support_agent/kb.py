"""A tiny knowledge base over the Markdown articles in ``support_agent/kb``.

Articles are split into ``##`` sections and ranked with TF-IDF over lowercased word tokens.
That is enough for a handful of help-centre pages; a real deployment would plug a proper
retriever in behind the same ``search`` signature.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or the this to "  # noqa: SIM905
    "what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]  # crude plural folding: "refunds" -> "refund"
        tokens.append(tok)
    return tokens


@dataclass(frozen=True)
class Section:
    article: str
    title: str
    heading: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    article: str
    title: str
    heading: str
    snippet: str
    score: float


def _split_sections(article: str, markdown: str) -> list[Section]:
    title = article
    sections: list[Section] = []
    heading = "Overview"
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append(Section(article, title, heading, body))

    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    flush()
    return sections


class KnowledgeBase:
    def __init__(self, sections: list[Section]) -> None:
        self.sections = sections
        self._tf = [Counter(tokenize(f"{s.title} {s.heading} {s.text}")) for s in sections]
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(sections)
        self._idf = {term: math.log((n + 1) / (count + 0.5)) for term, count in df.items()}

    @classmethod
    def from_directory(cls, directory: Path) -> KnowledgeBase:
        sections: list[Section] = []
        for path in sorted(directory.glob("*.md")):
            sections.extend(_split_sections(path.stem, path.read_text(encoding="utf-8")))
        return cls(sections)

    @classmethod
    def bundled(cls) -> KnowledgeBase:
        root = resources.files("support_agent") / "kb"
        sections: list[Section] = []
        for entry in sorted(root.iterdir(), key=lambda e: e.name):
            if entry.name.endswith(".md"):
                sections.extend(_split_sections(entry.name[:-3], entry.read_text("utf-8")))
        return cls(sections)

    def article(self, name: str) -> list[Section]:
        return [s for s in self.sections if s.article == name]

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            length_norm = math.sqrt(sum(tf.values())) or 1.0
            score = sum(tf[t] * self._idf.get(t, 0.0) for t in terms) / length_norm
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        hits = []
        for score, i in scored[:top_k]:
            s = self.sections[i]
            hits.append(SearchHit(s.article, s.title, s.heading, s.text, round(score, 4)))
        return hits

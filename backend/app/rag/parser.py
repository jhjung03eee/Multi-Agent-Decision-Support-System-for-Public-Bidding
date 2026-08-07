import io
import re

from app.schemas import BidFacts

KRW_UNITS = {"조": 10**12, "억": 10**8, "만": 10**4}

_HEADING_PATTERNS = [
    re.compile(r"^\s*(?:[0-9]{1,2}[.)]|[□■◇◆○●▶▷Ⅰ-Ⅹ])\s*(.{2,40})$"),
    re.compile(r"^\s*<\s*(.{2,40})\s*>\s*$"),
    re.compile(r"^\s*\[\s*(.{2,40})\s*\]\s*$"),
]

_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)]|[가-힣][.)])\s+(.*\S)\s*$")

_LABELS = {
    "title": ("사업명", "용역명", "공고명", "件名", "과업명"),
    "agency": ("발주기관", "수요기관", "공고기관", "계약담당", "기관명"),
    "budget": ("사업예산", "추정가격", "배정예산", "사업금액", "예산", "기초금액", "총사업비"),
    "deadline": ("제안서 제출마감", "입찰마감", "제출마감", "접수마감", "마감일시", "마감"),
    "duration": ("사업기간", "계약기간", "수행기간", "용역기간"),
    "region": ("사업지역", "이행지역", "지역제한", "소재지", "지역"),
}

BID_SIGNAL_KEYWORDS = (
    "입찰",
    "공고",
    "용역",
    "사업",
    "발주",
    "계약",
    "제안서",
    "낙찰",
    "과업",
    "구축",
    "제출",
    "심사",
    "평가",
    "예산",
    "나라장터",
)


def to_markdown(raw: bytes | str, filename: str) -> str:
    """Normalize an uploaded notice into markdown with `##` section headings."""
    if filename.lower().endswith(".pdf"):
        text = _pdf_to_text(raw if isinstance(raw, bytes) else raw.encode())
    else:
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

    if re.search(r"^#{1,6}\s+\S", text, re.MULTILINE):
        return text.strip()
    return _promote_headings(text).strip()


def _pdf_to_text(payload: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(payload))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _promote_headings(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        heading = _as_heading(line)
        lines.append(f"## {heading}" if heading else line)
    return "\n".join(lines)


def _as_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 60 or ":" in stripped:
        return None
    for pattern in _HEADING_PATTERNS:
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip()
    return None


def parse_krw(text: str) -> int | None:
    compact = text.replace(",", "").replace(" ", "")
    unit_matches = list(re.finditer(r"(\d+(?:\.\d+)?)(조|억|만)", compact))
    if unit_matches:
        total = sum(float(m.group(1)) * KRW_UNITS[m.group(2)] for m in unit_matches)
        tail = re.search(r"[조억만](\d+)원", compact)
        if tail:
            total += float(tail.group(1))
        # round, not truncate: 9.2 * 10**8 is 919999999.99… in binary floating point
        return round(total)
    plain = re.search(r"(\d{5,})원", compact)
    return int(plain.group(1)) if plain else None


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (section title, body) pairs, preserving document order."""
    sections: list[tuple[str, list[str]]] = [("서문", [])]
    for line in markdown.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*\S)\s*$", line)
        if heading:
            sections.append((heading.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(title, "\n".join(body).strip()) for title, body in sections if "\n".join(body).strip()]


def extract_facts(markdown: str) -> BidFacts:
    facts = BidFacts()
    labeled = _labeled_values(markdown)

    facts.title = labeled.get("title") or _first_heading(markdown) or "미상"
    facts.agency = labeled.get("agency")
    facts.deadline = labeled.get("deadline")
    facts.duration = labeled.get("duration")
    facts.region = labeled.get("region")

    budget_text = labeled.get("budget")
    if budget_text:
        facts.budget_text = budget_text
        facts.budget_krw = parse_krw(budget_text)

    facts.qualifications = _section_items(markdown, ("자격", "참가자격", "입찰참가", "제한사항"))
    facts.evaluation_criteria = _section_items(markdown, ("평가", "심사", "배점", "낙찰자 결정"))
    return facts


def bid_document_issue(markdown: str, facts: BidFacts) -> str | None:
    """Cheap, deterministic check that the input is actually a bid announcement.

    Runs before any agent so a document with no bid content in it (or a
    guardrail test) never burns four LLM calls just to report "no evidence
    found". Returns a user-facing reason when it isn't a bid document,
    otherwise None.
    """
    has_core_fact = any(
        [facts.agency, facts.budget_krw, facts.deadline, facts.duration, facts.region]
    )
    has_requirements = bool(facts.qualifications or facts.evaluation_criteria)
    has_keyword = any(keyword in markdown for keyword in BID_SIGNAL_KEYWORDS)
    if has_core_fact or has_requirements or has_keyword:
        return None
    return "입력한 문서에서 공고 내용을 확인할 수 없습니다. 실제 입찰 공고문을 첨부해 주세요."


def _first_heading(markdown: str) -> str | None:
    match = re.search(r"^#{1,6}\s+(.*\S)\s*$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else None


def _labeled_values(markdown: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in markdown.splitlines():
        cleaned = re.sub(r"^[\s\-*•·|#]+", "", line).strip().strip("|").strip()
        if not cleaned:
            continue
        separator = re.search(r"[:：]|\s{2,}|\|", cleaned)
        if not separator:
            continue
        label = cleaned[: separator.start()].strip()
        value = cleaned[separator.end() :].strip().strip("|").strip()
        if not value:
            continue
        for key, keywords in _LABELS.items():
            if key in found:
                continue
            if any(keyword in label for keyword in keywords):
                found[key] = value
                break
    return found


def _section_items(markdown: str, keywords: tuple[str, ...]) -> list[str]:
    items: list[str] = []
    for title, body in split_sections(markdown):
        if not any(keyword in title for keyword in keywords):
            continue
        for line in body.splitlines():
            bullet = _BULLET.match(line)
            if bullet:
                items.append(bullet.group(1).strip())
    return items[:15]

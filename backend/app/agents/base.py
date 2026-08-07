import json
import time
from typing import Callable

from app.agents import heuristics
from app.agents.profiles import AGENT_PROFILES, AgentProfile
from app.config import Settings
from app.guardrails import apply_guardrails
from app.llm.base import LLMClient
from app.rag.retriever import Retriever
from app.schemas import (
    AgentOpinion,
    AgentRole,
    BidFacts,
    Citation,
    CompanyProfile,
    Decision,
    EvidenceItem,
    RetrievedChunk,
)

HeuristicFn = Callable[[BidFacts, CompanyProfile, list[RetrievedChunk]], dict]

HEURISTICS: dict[AgentRole, HeuristicFn] = {
    AgentRole.SALES: heuristics.sales_opinion,
    AgentRole.TECHNICAL: heuristics.technical_opinion,
    AgentRole.FINANCE: heuristics.finance_opinion,
    AgentRole.LEGAL: heuristics.legal_opinion,
}


class DomainAgent:
    """One committee member: own retrieval scope, own prompt, own criteria."""

    def __init__(self, profile: AgentProfile, llm: LLMClient, settings: Settings) -> None:
        self.profile = profile
        self._llm = llm
        self._settings = settings

    async def review(
        self, retriever: Retriever, facts: BidFacts, company: CompanyProfile
    ) -> AgentOpinion:
        started = time.perf_counter()
        retrieved = await retriever.retrieve(
            list(self.profile.queries), self.profile.section_boost
        )
        heuristic = HEURISTICS[self.profile.role]

        raw = await self._llm.structured(
            self.profile.system_prompt,
            self._user_prompt(facts, company, retrieved),
            lambda: heuristic(facts, company, retrieved),
        )
        opinion = self._to_opinion(raw, retrieved)
        opinion.latency_ms = int((time.perf_counter() - started) * 1000)
        return apply_guardrails(opinion, retrieved, self._settings)

    def _user_prompt(
        self, facts: BidFacts, company: CompanyProfile, retrieved: list[RetrievedChunk]
    ) -> str:
        evidence = "\n\n".join(
            f"[{chunk.chunk_id}] (section: {chunk.section})\n{chunk.text}" for chunk in retrieved
        )
        return (
            "## COMPANY PROFILE (참고용 컨텍스트 — chunk_id로 인용 금지)\n"
            f"{json.dumps(company.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            "## BID FACTS (참고용 컨텍스트 — chunk_id로 인용 금지)\n"
            f"{json.dumps(facts.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            "## EVIDENCE (citations의 chunk_id는 여기 [cXXX] 값만 사용 가능)\n"
            f"{evidence or '(관련 발췌문 없음)'}\n\n"
            f"위 근거만으로 {self.profile.display_name}의 관점에서 판단하고 JSON으로 답하라."
        )

    def _to_opinion(self, raw: dict, retrieved: list[RetrievedChunk]) -> AgentOpinion:
        flags = ["llm_fallback"] if raw.get("degraded") else []
        return AgentOpinion(
            role=self.profile.role,
            display_name=self.profile.display_name,
            perspective=self.profile.perspective,
            decision=_coerce_decision(raw.get("decision")),
            confidence=_clamp(raw.get("confidence")),
            summary=str(raw.get("summary", "")).strip() or "판단 요약이 생성되지 않았습니다.",
            strengths=_evidence_items(raw.get("strengths")),
            risks=_evidence_items(raw.get("risks")),
            citations=_citations(raw.get("citations")),
            criteria_scores=_scores(raw.get("criteria_scores"), self.profile.criteria),
            guardrail_flags=flags,
            retrieved=retrieved,
        )


def build_agents(llm: LLMClient, settings: Settings) -> list[DomainAgent]:
    return [DomainAgent(profile, llm, settings) for profile in AGENT_PROFILES]


def _coerce_decision(value: object) -> Decision:
    text = str(value or "").strip().upper().replace("_", "-").replace(" ", "")
    if text in {"NO-GO", "NOGO"}:
        return Decision.NO_GO
    if text == "GO":
        return Decision.GO
    return Decision.REVIEW


def _clamp(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.4
    if number > 1:
        number = number / 100 if number <= 100 else 1.0
    return round(max(0.0, min(number, 1.0)), 2)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:6]


def _evidence_items(value: object) -> list[EvidenceItem]:
    if not isinstance(value, list):
        return []
    items: list[EvidenceItem] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            grounded = bool(item.get("grounded", False))
        else:
            text = str(item).strip()
            grounded = False
        if text:
            items.append(EvidenceItem(text=text, grounded=grounded))
    return items[:6]


def _citations(value: object) -> list[Citation]:
    if not isinstance(value, list):
        return []
    citations: list[Citation] = []
    for item in value:
        if not isinstance(item, dict) or not item.get("chunk_id"):
            continue
        citations.append(
            Citation(
                chunk_id=str(item["chunk_id"]).strip(),
                section=str(item.get("section", "")).strip(),
                quote=str(item.get("quote", "")).strip()[:400],
            )
        )
    return citations[:6]


def _scores(value: object, criteria: tuple[str, ...]) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    scores: dict[str, float] = {}
    for name in criteria:
        if name in value:
            scores[name] = _clamp(value[name])
    for key, raw in value.items():
        if key not in scores and len(scores) < len(criteria) + 2:
            scores[str(key)] = _clamp(raw)
    return scores

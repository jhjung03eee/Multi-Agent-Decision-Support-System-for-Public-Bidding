import json
import statistics

from app.agents.profiles import get_profile
from app.config import Settings
from app.llm.base import LLMClient
from app.schemas import (
    DECISION_SCORE,
    AgentOpinion,
    AgentRole,
    BidFacts,
    CommitteeResult,
    Decision,
    Vote,
)

CHAIR_SYSTEM = (
    "너는 공공입찰 Go/No-Go 심의위원회의 위원장이다.\n"
    "네 역할은 새로운 판단을 만드는 것이 아니라, 각 위원의 의견을 종합하여 최종 결론을 정리하는 것이다.\n\n"
    "규칙:\n"
    "1. 이미 계산된 가중 투표 결과(decision)를 뒤집지 마라. 그 결론을 설명하라.\n"
    "2. 위원 간 이견이 있으면 반드시 명시하라.\n"
    "3. 법무 위원이 NO-GO인 경우 그 사유를 최우선으로 다뤄라.\n"
    "4. 근거 없는 새로운 사실을 만들어내지 마라.\n\n"
    "JSON 스키마만 출력하라:\n"
    "{\n"
    '  "executive_summary": "경영진 보고용 3~5문장 요약",\n'
    '  "key_strengths": ["..."],\n'
    '  "key_risks": ["..."],\n'
    '  "conditions": ["참여 시 반드시 확보해야 할 조건"]\n'
    "}"
)


class CommitteeChair:
    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    async def decide(
        self, opinions: list[AgentOpinion], facts: BidFacts, narrative: bool = True
    ) -> CommitteeResult:
        votes = [self._to_vote(opinion) for opinion in opinions]
        total_weight = sum(vote.weight for vote in votes) or 1.0
        committee_score = sum(vote.weighted_score for vote in votes) / total_weight

        decision = _score_to_decision(committee_score)
        veto_reason = self._legal_veto(opinions)
        if veto_reason:
            decision = Decision.NO_GO

        confidence = self._committee_confidence(opinions, committee_score)
        human_reasons = self._human_review_reasons(opinions, confidence, decision)

        narrative_text = (
            await self._narrative(opinions, facts, decision, committee_score, veto_reason)
            if narrative
            else _fallback_narrative(opinions, decision, committee_score, veto_reason)
        )

        return CommitteeResult(
            decision=decision,
            confidence=confidence,
            committee_score=round(committee_score, 3),
            priority=_priority(committee_score, decision),
            votes=votes,
            executive_summary=narrative_text["executive_summary"],
            key_strengths=narrative_text["key_strengths"],
            key_risks=narrative_text["key_risks"],
            conditions=narrative_text["conditions"],
            dissenting_roles=_dissenters(opinions, decision),
            human_review_required=bool(human_reasons),
            human_review_reasons=human_reasons,
        )

    def _to_vote(self, opinion: AgentOpinion) -> Vote:
        weight = get_profile(opinion.role).weight
        # 확신이 낮은 표는 의결에서 영향력을 잃는다.
        effective_weight = weight * (0.5 + 0.5 * opinion.confidence)
        return Vote(
            role=opinion.role,
            display_name=opinion.display_name,
            decision=opinion.decision,
            confidence=opinion.confidence,
            weight=round(effective_weight, 4),
            weighted_score=round(effective_weight * DECISION_SCORE[opinion.decision], 4),
        )

    def _legal_veto(self, opinions: list[AgentOpinion]) -> str | None:
        for opinion in opinions:
            if (
                opinion.role is AgentRole.LEGAL
                and opinion.decision is Decision.NO_GO
                and opinion.confidence >= 0.7
            ):
                return opinion.summary
        return None

    def _committee_confidence(self, opinions: list[AgentOpinion], score: float) -> float:
        confidences = [o.confidence for o in opinions]
        spread = statistics.pstdev([DECISION_SCORE[o.decision] for o in opinions])
        consensus = 1 - min(spread * 2, 1.0)
        base = statistics.fmean(confidences)
        return round(min(0.97, 0.6 * base + 0.4 * consensus), 2)

    def _human_review_reasons(
        self, opinions: list[AgentOpinion], confidence: float, decision: Decision
    ) -> list[str]:
        reasons: list[str] = []
        if confidence < self._settings.confidence_threshold:
            reasons.append(
                f"위원회 신뢰도 {confidence:.0%}가 임계값 "
                f"{self._settings.confidence_threshold:.0%} 미만"
            )
        for opinion in opinions:
            if "no_evidence_retrieved" in opinion.guardrail_flags:
                reasons.append(f"{opinion.display_name}: 관련 근거 검색 실패")
            if "insufficient_citations" in opinion.guardrail_flags:
                reasons.append(f"{opinion.display_name}: 인용 근거 부족으로 REVIEW 강등")
        decisions = {o.decision for o in opinions}
        if Decision.GO in decisions and Decision.NO_GO in decisions:
            reasons.append("위원 간 GO/NO-GO 정면 충돌")
        if decision is Decision.REVIEW:
            reasons.append("최종 판정이 REVIEW이므로 담당자 확인 필요")
        return list(dict.fromkeys(reasons))

    async def _narrative(
        self,
        opinions: list[AgentOpinion],
        facts: BidFacts,
        decision: Decision,
        score: float,
        veto_reason: str | None,
    ) -> dict:
        digest = [
            {
                "role": o.display_name,
                "decision": o.decision.value,
                "confidence": o.confidence,
                "summary": o.summary,
                "strengths": o.strengths,
                "risks": o.risks,
            }
            for o in opinions
        ]
        user = (
            f"## 최종 판정 (변경 불가)\n{decision.value} (가중 점수 {score:.2f})\n\n"
            + (f"## 법무 거부권 발동 사유\n{veto_reason}\n\n" if veto_reason else "")
            + f"## 공고 정보\n{json.dumps(facts.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"## 위원 의견\n{json.dumps(digest, ensure_ascii=False, indent=2)}\n\n"
            "위 의견을 종합하여 경영진 보고서를 JSON으로 작성하라."
        )
        raw = await self._llm.structured(
            CHAIR_SYSTEM, user, lambda: _fallback_narrative(opinions, decision, score, veto_reason)
        )
        return {
            "executive_summary": str(raw.get("executive_summary", "")).strip()
            or _fallback_narrative(opinions, decision, score, veto_reason)["executive_summary"],
            "key_strengths": _top(raw.get("key_strengths"), [s for o in opinions for s in o.strengths]),
            "key_risks": _top(raw.get("key_risks"), [r for o in opinions for r in o.risks]),
            "conditions": _top(raw.get("conditions"), []),
        }


def _fallback_narrative(
    opinions: list[AgentOpinion], decision: Decision, score: float, veto_reason: str | None
) -> dict:
    tally = ", ".join(f"{o.display_name} {o.decision.value}" for o in opinions)
    lead = {
        Decision.GO: "참여를 권고한다.",
        Decision.REVIEW: "추가 검토 후 결정할 것을 권고한다.",
        Decision.NO_GO: "참여하지 않을 것을 권고한다.",
    }[decision]
    summary = f"위원회 가중 점수는 {score:.2f}이며 {lead} 위원별 판단은 {tally}이다."
    if veto_reason:
        summary += f" 법무 위원의 거부권이 발동되었다: {veto_reason}"

    conditions = [f"{o.display_name} 지적사항 해소: {o.risks[0]}" for o in opinions if o.risks][:4]
    return {
        "executive_summary": summary,
        "key_strengths": [s for o in opinions for s in o.strengths][:5],
        "key_risks": [r for o in opinions for r in o.risks][:5],
        "conditions": conditions,
    }


def _top(value: object, fallback: list[str], limit: int = 5) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return items[:limit]
    return fallback[:limit]


def _score_to_decision(score: float) -> Decision:
    if score >= 0.7:
        return Decision.GO
    if score >= 0.45:
        return Decision.REVIEW
    return Decision.NO_GO


def _priority(score: float, decision: Decision) -> int:
    if decision is Decision.NO_GO:
        return 1
    return max(1, min(5, round(score * 5)))


def _dissenters(opinions: list[AgentOpinion], decision: Decision) -> list[AgentRole]:
    return [o.role for o in opinions if o.decision is not decision]

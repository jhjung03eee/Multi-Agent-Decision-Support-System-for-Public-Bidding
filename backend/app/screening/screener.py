import asyncio
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

from app.company import load_company_profile
from app.config import Settings, get_settings
from app.schemas import (
    CommitteeResult,
    CompanyProfile,
    Decision,
    Recommendation,
    ScreeningItem,
    ScreeningReport,
    ScreenOutcome,
)
from app.screening.dataset import BidRecord, corpus_as_of, load_company, load_corpus
from app.screening.filters import prefilter
from app.supervisor import Supervisor

logger = logging.getLogger(__name__)

DECISION_TO_RECOMMENDATION = {
    Decision.GO: Recommendation.STRONG,
    Decision.REVIEW: Recommendation.REVIEW,
    Decision.NO_GO: Recommendation.PASS,
}


class BatchScreener:
    """Screens a whole corpus: prefilter first, committee only on survivors."""

    def __init__(self, settings: Settings | None = None, concurrency: int = 4) -> None:
        self._settings = settings or get_settings()
        self._supervisor = Supervisor(self._settings)
        self._semaphore = asyncio.Semaphore(concurrency)

    def resolve_as_of(self, records: list[BidRecord]) -> tuple[date, str]:
        """Reference date for deadline checks, and where it came from.

        Priority: explicit BIDCOM_AS_OF, then the corpus's own announcement
        window, then the wall clock. The source is reported so the UI can say
        which one is in effect rather than silently showing odd D-days.
        """
        pinned = self._settings.as_of_date
        if pinned:
            return pinned, "configured"
        derived = corpus_as_of(records)
        if derived:
            return derived, "corpus"
        return date.today(), "today"

    async def screen_corpus(self, root: Path, today: date | None = None) -> ScreeningReport:
        records = load_corpus(root)
        company = load_company(root) or load_company_profile()
        return await self.screen(records, company, corpus=str(root), today=today)

    async def screen(
        self,
        records: list[BidRecord],
        company: CompanyProfile,
        corpus: str = "",
        today: date | None = None,
    ) -> ScreeningReport:
        started = time.perf_counter()
        as_of, as_of_source = (today, "explicit") if today else self.resolve_as_of(records)

        items = await asyncio.gather(
            *(self._screen_one(record, company, as_of) for record in records)
        )
        items = sorted(items, key=_rank_key)

        counts = {tier.value: 0 for tier in Recommendation}
        for item in items:
            counts[item.recommendation.value] += 1

        return ScreeningReport(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            as_of=as_of.isoformat(),
            as_of_source=as_of_source,
            corpus=corpus,
            company=company.name,
            total=len(items),
            screened_by_committee=sum(1 for item in items if item.committee),
            filtered_out=sum(1 for item in items if item.screen.blocked),
            counts=counts,
            items=items,
            total_latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _screen_one(
        self, record: BidRecord, company: CompanyProfile, today: date | None
    ) -> ScreeningItem:
        facts = record.facts()
        outcome = prefilter(record, company, today)
        base = dict(
            bid_id=record.bid_id,
            title=facts.title,
            agency=facts.agency,
            category=record.category,
            budget_krw=facts.budget_krw,
            deadline=facts.deadline,
            region=facts.region,
            days_left=outcome.days_left,
            urgent=outcome.urgent,
            screen=outcome,
        )

        if outcome.blocked:
            return ScreeningItem(
                **base,
                recommendation=Recommendation.PASS,
                score=0.0,
                reason=" · ".join(outcome.block_reasons),
            )

        try:
            async with self._semaphore:
                # Skip the chair's LLM narrative pass in batch mode: it's a second
                # sequential LLM round-trip per bid that batch reports don't render,
                # and cutting it roughly halves per-bid latency against the
                # serverless function's hard wall-clock budget.
                result = await self._supervisor.run(
                    record.markdown, record.bid_id, company, narrative=False
                )
        except Exception as exc:
            logger.exception("committee failed for %s", record.bid_id)
            return ScreeningItem(
                **base,
                recommendation=Recommendation.REVIEW,
                score=0.5,
                reason="심의 중 오류가 발생하여 담당자 확인이 필요합니다.",
                human_review_required=True,
                error=str(exc),
            )

        committee = result.committee
        return ScreeningItem(
            **base,
            recommendation=DECISION_TO_RECOMMENDATION[committee.decision],
            score=round(committee.committee_score, 3),
            reason=_reason(committee, outcome),
            committee=committee,
            human_review_required=committee.human_review_required,
            agent_decisions={o.role.value: o.decision for o in result.opinions},
        )


def _reason(committee: CommitteeResult, outcome: ScreenOutcome) -> str:
    parts: list[str] = []
    if outcome.urgent:
        parts.append(f"D-{outcome.days_left} 마감임박")
    if committee.decision is Decision.NO_GO and committee.key_risks:
        parts.append(committee.key_risks[0])
    elif committee.decision is Decision.GO and committee.key_strengths:
        parts.append(committee.key_strengths[0])
    elif committee.human_review_reasons:
        parts.append(committee.human_review_reasons[0])
    return " · ".join(parts) or committee.executive_summary[:120]


TIER_ORDER = {Recommendation.STRONG: 0, Recommendation.REVIEW: 1, Recommendation.PASS: 2}


def _rank_key(item: ScreeningItem) -> tuple:
    """적극추천 먼저, 같은 등급이면 마감 임박 순, 그 다음 점수 순."""
    return (
        TIER_ORDER[item.recommendation],
        item.days_left if item.days_left is not None else 999,
        -item.score,
    )

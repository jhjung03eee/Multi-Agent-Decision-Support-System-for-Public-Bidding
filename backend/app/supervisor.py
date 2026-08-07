import asyncio
import hashlib
import time
from typing import AsyncIterator

from app.agents.base import build_agents
from app.agents.chair import CommitteeChair
from app.company import load_company_profile
from app.config import Settings, get_settings
from app.guardrails import citation_stats, grounding_rate
from app.llm.factory import get_llm_client
from app.rag.chunker import chunk_markdown
from app.rag.embeddings import get_embeddings
from app.rag.parser import bid_document_issue, extract_facts
from app.rag.retriever import Retriever
from app.rag.store import VectorStore
from app.schemas import (
    AgentOpinion,
    CompanyProfile,
    ReviewResult,
    WorkflowEvent,
    WorkflowMetrics,
)


class Supervisor:
    """Owns the workflow: parse → index → agents → committee → validation → report."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = get_llm_client(self._settings)
        self._agents = build_agents(self._llm, self._settings)
        self._chair = CommitteeChair(self._llm, self._settings)

    async def run(
        self,
        markdown: str,
        document_name: str,
        company: CompanyProfile | None = None,
        narrative: bool = True,
    ) -> ReviewResult:
        result: ReviewResult | None = None
        stream = self.stream(markdown, document_name, company, narrative=narrative)
        try:
            async for event in stream:
                if event.stage == "completed":
                    result = ReviewResult(**event.payload)
                elif event.stage == "error":
                    raise ValueError(event.message)
        finally:
            await stream.aclose()
        if result is None:
            raise RuntimeError("workflow finished without a result")
        return result

    async def stream(
        self,
        markdown: str,
        document_name: str,
        company: CompanyProfile | None = None,
        narrative: bool = True,
    ) -> AsyncIterator[WorkflowEvent]:
        started = time.perf_counter()
        company = company or load_company_profile()
        document_id = hashlib.sha256(markdown.encode()).hexdigest()[:12]

        facts = extract_facts(markdown)
        yield WorkflowEvent(
            stage="parsing",
            message="공고문 파싱 및 핵심 정보 추출 완료",
            payload={"facts": facts.model_dump(), "document_id": document_id},
        )

        issue = bid_document_issue(markdown, facts)
        if issue:
            yield WorkflowEvent(
                stage="error",
                message=issue,
                payload={"reason": "not_a_bid_document"},
            )
            return

        chunks = chunk_markdown(markdown, self._settings.chunk_size, self._settings.chunk_overlap)
        store = VectorStore(get_embeddings(self._settings))
        await store.index(chunks)
        retriever = Retriever(store, self._settings.retrieval_top_k)
        yield WorkflowEvent(
            stage="indexing",
            message=f"{len(chunks)}개 청크 인덱싱 완료",
            payload={"chunk_count": len(chunks)},
        )

        for agent in self._agents:
            yield WorkflowEvent(
                stage="agent_started",
                message=f"{agent.profile.display_name} 검토 시작",
                payload={"role": agent.profile.role.value},
            )

        pending = {
            asyncio.create_task(agent.review(retriever, facts, company)): agent
            for agent in self._agents
        }
        opinions: list[AgentOpinion] = []
        for task in asyncio.as_completed(list(pending)):
            opinion = await task
            opinions.append(opinion)
            yield WorkflowEvent(
                stage="agent_completed",
                message=f"{opinion.display_name} 판단 완료: {opinion.decision.value}",
                payload={"opinion": opinion.model_dump(mode="json")},
            )

        order = {agent.profile.role: i for i, agent in enumerate(self._agents)}
        opinions.sort(key=lambda o: order[o.role])

        yield WorkflowEvent(stage="committee_started", message="위원회 가중 투표 진행")
        committee = await self._chair.decide(opinions, facts, narrative=narrative)
        yield WorkflowEvent(
            stage="committee_completed",
            message=f"위원장 최종 판정: {committee.decision.value}",
            payload={"committee": committee.model_dump(mode="json")},
        )

        total_citations, valid_citations = citation_stats(opinions)
        metrics = WorkflowMetrics(
            total_latency_ms=int((time.perf_counter() - started) * 1000),
            chunk_count=len(chunks),
            citation_count=total_citations,
            valid_citation_count=valid_citations,
            citation_validity_rate=round(valid_citations / total_citations, 3)
            if total_citations
            else 0.0,
            grounding_rate=grounding_rate(opinions),
            mean_confidence=round(sum(o.confidence for o in opinions) / len(opinions), 2),
            llm_provider=self._llm.name,
        )
        yield WorkflowEvent(
            stage="validation",
            message="가드레일 검증 및 지표 산출 완료",
            payload={"metrics": metrics.model_dump()},
        )

        result = ReviewResult(
            document_id=document_id,
            document_name=document_name,
            facts=facts,
            opinions=opinions,
            committee=committee,
            metrics=metrics,
        )
        yield WorkflowEvent(
            stage="completed",
            message="심의 완료",
            payload=result.model_dump(mode="json"),
        )

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Decision(str, Enum):
    GO = "GO"
    REVIEW = "REVIEW"
    NO_GO = "NO-GO"


class AgentRole(str, Enum):
    SALES = "sales"
    TECHNICAL = "technical"
    FINANCE = "finance"
    LEGAL = "legal"


DECISION_SCORE: dict[Decision, float] = {
    Decision.GO: 1.0,
    Decision.REVIEW: 0.5,
    Decision.NO_GO: 0.0,
}


class Citation(BaseModel):
    chunk_id: str
    section: str
    quote: str


class BidFacts(BaseModel):
    """Structured fields parsed out of the notice before any agent runs."""

    title: str = "미상"
    agency: str | None = None
    budget_krw: int | None = None
    budget_text: str | None = None
    deadline: str | None = None
    duration: str | None = None
    region: str | None = None
    qualifications: list[str] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk_id: str
    section: str
    text: str
    score: float


class EvidenceItem(BaseModel):
    """A strength/risk claim, tagged with whether it's backed by real EVIDENCE.

    `grounded=False` means the claim rests only on COMPANY PROFILE / BID FACTS
    context (generic company capability, not something confirmed by the
    announcement's own text) — the UI must not render it as if it were an
    evidence-backed finding.
    """

    text: str
    grounded: bool = False

    @model_validator(mode="before")
    @classmethod
    def _coerce_plain_string(cls, data: object) -> object:
        if isinstance(data, str):
            return {"text": data, "grounded": False}
        return data


class AgentOpinion(BaseModel):
    role: AgentRole
    display_name: str
    perspective: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    strengths: list[EvidenceItem] = Field(default_factory=list)
    risks: list[EvidenceItem] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    criteria_scores: dict[str, float] = Field(default_factory=dict)
    guardrail_flags: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: int = 0


class Vote(BaseModel):
    role: AgentRole
    display_name: str
    decision: Decision
    confidence: float
    weight: float
    weighted_score: float


class CommitteeResult(BaseModel):
    decision: Decision
    confidence: float
    committee_score: float
    priority: int = Field(ge=0, le=5)
    votes: list[Vote]
    executive_summary: str
    key_strengths: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    dissenting_roles: list[AgentRole] = Field(default_factory=list)
    human_review_required: bool = False
    human_review_reasons: list[str] = Field(default_factory=list)


class WorkflowMetrics(BaseModel):
    total_latency_ms: int
    chunk_count: int
    citation_count: int
    valid_citation_count: int
    citation_validity_rate: float
    grounding_rate: float
    mean_confidence: float
    llm_provider: str


class ReviewResult(BaseModel):
    document_id: str
    document_name: str
    facts: BidFacts
    opinions: list[AgentOpinion]
    committee: CommitteeResult
    metrics: WorkflowMetrics


class WorkflowEvent(BaseModel):
    stage: Literal[
        "parsing",
        "indexing",
        "agent_started",
        "agent_completed",
        "committee_started",
        "committee_completed",
        "validation",
        "completed",
        "error",
    ]
    message: str
    payload: dict = Field(default_factory=dict)


class CompanyProfile(BaseModel):
    name: str
    headcount: int
    annual_revenue_krw: int
    regions: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    industry_codes: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    past_projects: list[str] = Field(default_factory=list)
    target_margin: float = 0.15
    min_project_budget_krw: int = 0
    max_concurrent_projects: int = 5
    current_active_projects: int = 0
    min_preparation_days: int = 7

    business_type: str | None = None
    capital_krw: int = 0
    # Delivery capacity is bounded by engineers, not headcount: a 38,000-person
    # manufacturer with 450 technical staff can only staff so many bids at once.
    technical_headcount: int = 0
    preferred_categories: list[str] = Field(default_factory=list)

    @property
    def delivery_headcount(self) -> int:
        """Staff actually available to deliver a bid."""
        return self.technical_headcount or self.headcount


class Recommendation(str, Enum):
    STRONG = "적극추천"
    REVIEW = "검토"
    PASS = "패스"


class ScreenOutcome(BaseModel):
    """Cheap deterministic prefilter applied before any agent runs."""

    blocked: bool = False
    block_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    days_left: int | None = None
    urgent: bool = False


class ScreeningItem(BaseModel):
    bid_id: str
    title: str
    agency: str | None = None
    category: str | None = None
    budget_krw: int | None = None
    deadline: str | None = None
    region: str | None = None
    days_left: int | None = None
    urgent: bool = False
    recommendation: Recommendation
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    screen: ScreenOutcome
    committee: CommitteeResult | None = None
    human_review_required: bool = False
    agent_decisions: dict[str, Decision] = Field(default_factory=dict)
    error: str | None = None


class ScreeningReport(BaseModel):
    generated_at: str
    corpus: str
    company: str
    # Date the deadlines were judged against. Differs from generated_at when the
    # corpus is an archive being screened as of its own announcement window.
    as_of: str
    as_of_source: str = "today"
    total: int
    screened_by_committee: int
    filtered_out: int
    counts: dict[str, int]
    items: list[ScreeningItem]
    total_latency_ms: int


class ReviewRequest(BaseModel):
    document_text: str | None = None
    document_name: str = "bid-notice.md"
    sample_id: str | None = None

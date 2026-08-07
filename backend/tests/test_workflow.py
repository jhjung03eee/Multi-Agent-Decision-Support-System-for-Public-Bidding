import pytest
from fastapi.testclient import TestClient

from app.agents.heuristics import _offers_advance_payment
from app.config import Settings
from app.main import app
from app.rag.embeddings import HashingEmbeddings
from app.rag.chunker import chunk_markdown
from app.rag.retriever import Retriever
from app.rag.store import VectorStore
from app.samples import load_sample
from app.schemas import AgentRole, Decision
from app.supervisor import Supervisor

SETTINGS = Settings()

# A notice from the bundled corpus that overlaps KIA's delivery record.
GO_SAMPLE = "용역_데이터베이스_성능튜닝_및_이전_용역_20250020-2861"

# Scenario notices are written inline rather than pulled from the corpus: the
# committee's behaviour on a region-locked or sub-threshold bid must be
# testable whether or not the installed corpus happens to contain one.
REGION_LOCKED = """# 대구 스마트시티 통합플랫폼 구축

## 1. 사업 개요

- 사업예산: 5,000,000,000원
- 사업지역: 대구광역시
- 사업기간: 12개월

## 2. 입찰참가 자격요건

- 본 입찰은 지역제한 경쟁입찰이며 본점 소재지가 대구광역시인 업체로 한정한다.
- 소프트웨어사업자 신고를 필한 업체
- 정보시스템 감리법인 등록(해당시)

## 3. 평가 방법

- 협상에 의한 계약, 기술평가 80점
"""

# Budget, award method and payment terms sit in one section on purpose: the
# finance agent retrieves by topic, so the signals it weighs have to travel
# together or they never reach it.
LOW_VALUE = """# 노후 안내단말기 유지관리

## 1. 사업 개요

- 사업지역: 서울특별시
- 사업기간: 12개월

## 2. 사업예산 및 대가 지급

- 사업예산: 300,000,000원
- 낙찰자 결정: 적격심사를 거쳐 최저가 입찰자를 낙찰자로 한다.
- 대가 지급: 준공후 일괄지급하며 선금은 지급하지 않는다.

## 3. 입찰참가 자격요건

- 소프트웨어사업자 신고를 필한 업체
"""


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


async def test_retriever_gives_each_role_different_evidence():
    markdown = load_sample(GO_SAMPLE)
    store = VectorStore(HashingEmbeddings())
    await store.index(chunk_markdown(markdown))
    retriever = Retriever(store, top_k=4)

    legal = await retriever.retrieve(["입찰참가 자격요건 및 제한사항"], ("자격",))
    finance = await retriever.retrieve(["사업 예산 및 대가 지급 방법"], ("예산", "지급"))

    assert legal and finance
    assert {c.chunk_id for c in legal} != {c.chunk_id for c in finance}


async def test_go_case_produces_go_with_grounded_citations():
    result = await Supervisor(SETTINGS).run(load_sample(GO_SAMPLE), "db-tuning.md")

    assert result.committee.decision is Decision.GO
    assert len(result.opinions) == 4
    assert {o.role for o in result.opinions} == set(AgentRole)
    assert result.metrics.citation_validity_rate == 1.0
    assert all(o.citations for o in result.opinions)

    chunk_ids = {c.chunk_id for o in result.opinions for c in o.retrieved}
    cited = {c.chunk_id for o in result.opinions for c in o.citations}
    assert cited <= chunk_ids


async def test_region_restriction_triggers_legal_veto():
    result = await Supervisor(SETTINGS).run(REGION_LOCKED, "region-locked.md")

    legal = next(o for o in result.opinions if o.role is AgentRole.LEGAL)
    assert legal.decision is Decision.NO_GO
    assert legal.confidence >= 0.7, "확신이 낮은 법무 의견은 거부권이 되지 않는다"

    # The veto overrides the weighted tally rather than merely contributing to it.
    assert result.committee.decision is Decision.NO_GO
    assert "거부권" in result.committee.executive_summary


async def test_low_value_maintenance_case_is_not_go():
    result = await Supervisor(SETTINGS).run(LOW_VALUE, "low-value.md")

    finance = next(o for o in result.opinions if o.role is AgentRole.FINANCE)
    assert finance.decision is Decision.NO_GO
    assert any("최소 수주 기준" in risk.text for risk in finance.risks)
    assert result.committee.decision is not Decision.GO


def test_declined_advance_payment_is_not_read_as_a_benefit():
    """`선금은 지급하지 않는다` must not count as a 선금 clause."""
    assert not _offers_advance_payment("선금은 지급하지 않는다")
    assert not _offers_advance_payment("기성 대가는 지급 없음")
    assert _offers_advance_payment("선금 30%를 지급한다")
    assert _offers_advance_payment("기성고에 따라 분할지급한다")


async def test_stream_emits_stages_in_order():
    stages = [
        event.stage
        async for event in Supervisor(SETTINGS).stream(load_sample(GO_SAMPLE), "db-tuning.md")
    ]
    assert stages[0] == "parsing"
    assert stages[1] == "indexing"
    assert stages.count("agent_started") == 4
    assert stages.count("agent_completed") == 4
    assert stages[-1] == "completed"
    assert stages.index("committee_completed") > stages.index("agent_completed")


async def test_non_bid_document_is_rejected_before_any_agent_runs():
    stages = [
        event.stage
        async for event in Supervisor(SETTINGS).stream(
            "안녕하세요 저는 정동진입니다.", "greeting.md"
        )
    ]
    assert stages == ["parsing", "error"]

    with pytest.raises(ValueError):
        await Supervisor(SETTINGS).run("안녕하세요 저는 정동진입니다.", "greeting.md")


def test_review_endpoint_rejects_non_bid_document(client):
    response = client.post(
        "/api/review", json={"document_text": "안녕하세요 저는 정동진입니다.", "document_name": "greeting.md"}
    )
    assert response.status_code == 422


def test_health_and_config_endpoints(client):
    assert client.get("/api/health").json()["status"] == "ok"
    config = client.get("/api/config").json()
    assert len(config["agents"]) == 4
    assert config["company"]["name"]


def test_samples_endpoint_lists_bundled_notices(client):
    ids = {s["id"] for s in client.get("/api/samples").json()["samples"]}
    assert GO_SAMPLE in ids
    assert len(ids) == 20


def test_review_endpoint_returns_full_result(client):
    response = client.post("/api/review", json={"sample_id": GO_SAMPLE})
    assert response.status_code == 200
    body = response.json()
    assert body["committee"]["decision"] == "GO"
    assert len(body["opinions"]) == 4


def test_review_rejects_empty_request(client):
    assert client.post("/api/review", json={}).status_code == 422


def test_review_rejects_unknown_sample(client):
    assert client.post("/api/review", json={"sample_id": "../../etc/passwd"}).status_code == 404


def test_stream_endpoint_emits_sse_frames(client):
    with client.stream("POST", "/api/review/stream", json={"sample_id": GO_SAMPLE}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "event: parsing" in body
    assert "event: completed" in body


def test_upload_converts_markdown(client):
    files = {"file": ("notice.md", b"# hi\n\n- gitea", "text/markdown")}
    body = client.post("/api/upload", files=files).json()
    assert body["document_name"] == "notice.md"
    assert "# hi" in body["markdown"]


def test_upload_rejects_unsupported_type(client):
    files = {"file": ("notice.exe", b"binary", "application/octet-stream")}
    assert client.post("/api/upload", files=files).status_code == 415

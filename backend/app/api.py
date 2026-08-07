import json
from typing import AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agents.profiles import AGENT_PROFILES
from app.company import load_company_profile
from app.config import get_settings
from app.rag.parser import to_markdown
from app.samples import list_samples, load_sample
from app.schemas import ReviewRequest, ReviewResult, ScreeningReport
from app.screening.dataset import load_company, load_corpus
from app.screening.screener import BatchScreener
from app.supervisor import Supervisor

router = APIRouter(prefix="/api")

# Vercel caps serverless request bodies at 4.5MB.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_provider": "openai" if settings.live_llm else "heuristic-offline",
        "model": settings.openai_model if settings.live_llm else None,
    }


@router.get("/config")
async def config() -> dict:
    settings = get_settings()
    return {
        "company": load_company_profile().model_dump(),
        "agents": [
            {
                "role": p.role.value,
                "display_name": p.display_name,
                "perspective": p.perspective,
                "mandate": p.mandate,
                "criteria": list(p.criteria),
                "weight": p.weight,
                "veto": p.veto,
            }
            for p in AGENT_PROFILES
        ],
        "confidence_threshold": settings.confidence_threshold,
        "live_llm": settings.live_llm,
    }


@router.get("/samples")
async def samples() -> dict:
    return {"samples": list_samples()}


@router.get("/corpus")
async def corpus() -> dict:
    settings = get_settings()
    root = settings.corpus_path
    if not root.is_dir():
        return {"path": str(root), "available": False, "count": 0, "bids": []}

    records = load_corpus(root)
    company = load_company(root) or load_company_profile()
    return {
        "path": str(root),
        "available": True,
        "count": len(records),
        "company": company.model_dump(),
        "bids": [
            {"bid_id": r.bid_id, "title": r.facts().title, "source": r.source} for r in records
        ],
    }


@router.post("/screen", response_model=ScreeningReport)
async def screen() -> ScreeningReport:
    settings = get_settings()
    root = settings.corpus_path
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"공고 코퍼스를 찾을 수 없습니다: {root}")
    screener = BatchScreener(settings, concurrency=settings.screening_concurrency)
    return await screener.screen_corpus(root)


@router.post("/review", response_model=ReviewResult)
async def review(request: ReviewRequest) -> ReviewResult:
    markdown, name = _resolve_document(request)
    try:
        return await Supervisor().run(markdown, name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/review/stream")
async def review_stream(request: ReviewRequest) -> StreamingResponse:
    markdown, name = _resolve_document(request)
    return StreamingResponse(
        _sse(markdown, name),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/upload")
async def upload(file: UploadFile = File(...), name: str | None = Form(default=None)) -> dict:
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (최대 4MB)")
    filename = name or file.filename or "bid-notice.md"
    if not filename.lower().endswith((".pdf", ".md", ".markdown", ".txt")):
        raise HTTPException(status_code=415, detail="PDF, Markdown, TXT 파일만 지원합니다")
    try:
        markdown = to_markdown(payload, filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"문서를 읽을 수 없습니다: {exc}") from exc
    if not markdown.strip():
        raise HTTPException(status_code=422, detail="문서에서 텍스트를 추출하지 못했습니다")
    return {"document_name": filename, "markdown": markdown}


def _resolve_document(request: ReviewRequest) -> tuple[str, str]:
    if request.sample_id:
        markdown = load_sample(request.sample_id)
        if markdown is None:
            raise HTTPException(status_code=404, detail="샘플 공고를 찾을 수 없습니다")
        return markdown, f"{request.sample_id}.md"
    if request.document_text and request.document_text.strip():
        return request.document_text, request.document_name
    raise HTTPException(status_code=422, detail="document_text 또는 sample_id가 필요합니다")


async def _sse(markdown: str, name: str) -> AsyncIterator[str]:
    supervisor = Supervisor()
    try:
        async for event in supervisor.stream(markdown, name):
            yield _format_event(event.stage, event.model_dump(mode="json"))
    except Exception as exc:
        yield _format_event("error", {"stage": "error", "message": str(exc), "payload": {}})


def _format_event(stage: str, data: dict) -> str:
    return f"event: {stage}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

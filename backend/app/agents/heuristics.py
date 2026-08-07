"""Deterministic role reasoning.

Runs as the offline provider and as the fallback when a live OpenAI call fails, so the
committee always produces an auditable result instead of an empty one.
"""

import re

from app.schemas import BidFacts, CompanyProfile, Decision, RetrievedChunk

CERT_NOUNS = ("인증", "면허", "등록", "신고", "자격", "확인", "평가등급")
RISK_CLAUSES = {
    "지체상금": "지체상금 조항 존재",
    "손해배상": "손해배상 책임 조항 존재",
    "하자보수": "하자보수 보증 의무",
    "저작권": "산출물 저작권 귀속 조건 확인 필요",
    "소스코드": "소스코드 인도 의무 확인 필요",
    "무상유지보수": "무상 유지보수 요구",
}
PAYMENT_GOOD = ("선금", "기성", "분할지급", "기성고")
PAYMENT_BAD = ("준공후", "일괄지급", "검수후 일괄")
# 공고문은 유리한 조건을 부정형으로 적는 일이 잦다: "선금은 지급하지 않는다".
PAYMENT_NEGATION = re.compile(r"지급하지\s*않|미지급|지급\s*없|지급\s*불가|지급하지\s*아니")
LOW_PRICE_SIGNALS = ("최저가", "낙찰하한", "적격심사")
QUALITY_SIGNALS = ("협상에 의한 계약", "기술평가", "제안서 평가", "종합평가")


def _corpus(retrieved: list[RetrievedChunk]) -> str:
    return "\n".join(chunk.text for chunk in retrieved).lower()


def _coverage(needles: list[str], haystack: str) -> tuple[float, list[str], list[str]]:
    if not needles:
        return 0.0, [], []
    matched = [n for n in needles if n.lower() in haystack]
    missing = [n for n in needles if n not in matched]
    return len(matched) / len(needles), matched, missing


# 관련 역량이 3개 확인되면 적합성은 만점으로 본다.
_FIT_SATURATION = 3


def _fit(matched: list[str]) -> float:
    """Capability fit from how many relevant items matched.

    Deliberately not `matched / portfolio_size`: a company that lists 22
    capabilities and matches 3 of them fits the notice at least as well as one
    that lists 4 and matches 1. Scoring by share would punish the broader firm
    for having a portfolio, which is exactly backwards.
    """
    return min(1.0, len(matched) / _FIT_SATURATION)


def _citations(retrieved: list[RetrievedChunk], limit: int = 3) -> list[dict]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "section": chunk.section,
            "quote": " ".join(chunk.text.split())[:180],
        }
        for chunk in retrieved[:limit]
    ]


def _decide(scores: dict[str, float], evidence_count: int) -> tuple[str, float]:
    mean = sum(scores.values()) / len(scores)
    if mean >= 0.7:
        decision = Decision.GO
    elif mean >= 0.45:
        decision = Decision.REVIEW
    else:
        decision = Decision.NO_GO

    spread = max(scores.values()) - min(scores.values())
    evidence_factor = min(evidence_count, 4) / 4
    confidence = 0.45 + 0.35 * evidence_factor + 0.2 * (1 - spread)
    return decision.value, round(min(confidence, 0.95), 2)


def _months(duration: str | None) -> float | None:
    if not duration:
        return None
    month = re.search(r"(\d+)\s*(?:개월|달)", duration)
    if month:
        return float(month.group(1))
    year = re.search(r"(\d+)\s*년", duration)
    if year:
        return float(year.group(1)) * 12
    day = re.search(r"(\d+)\s*일", duration)
    if day:
        return float(day.group(1)) / 30
    return None


def sales_opinion(
    facts: BidFacts, company: CompanyProfile, retrieved: list[RetrievedChunk]
) -> dict:
    corpus = _corpus(retrieved)
    strengths: list[str] = []
    risks: list[str] = []

    _, matched_tech, _ = _coverage(company.tech_stack, corpus)
    domain_fit = _fit(matched_tech)
    if matched_tech:
        strengths.append(f"보유 역량과 겹치는 영역: {', '.join(matched_tech[:4])}")
    else:
        risks.append("공고 내용에서 자사 주력 기술 영역이 확인되지 않음")

    _, matched_projects, _ = _coverage(company.past_projects, corpus)
    if matched_projects:
        strengths.append(f"유사 수행 실적 연관: {', '.join(matched_projects[:3])}")

    budget = facts.budget_krw or 0
    if company.min_project_budget_krw and budget:
        budget_score = min(budget / company.min_project_budget_krw, 1.5) / 1.5
    elif budget:
        budget_score = 0.7
    else:
        budget_score = 0.4
        risks.append("예산 규모가 공고에서 확인되지 않아 사업 가치 판단이 제한됨")

    if facts.region and company.regions:
        region_ok = any(region in facts.region for region in company.regions)
        customer_score = 1.0 if region_ok else 0.35
        if region_ok:
            strengths.append(f"영업 커버리지 내 지역: {facts.region}")
        else:
            risks.append(f"영업 거점 외 지역: {facts.region}")
    else:
        customer_score = 0.6

    if facts.agency:
        strengths.append(f"발주기관: {facts.agency}")

    strategic = 0.5 + 0.5 * domain_fit
    scores = {
        "사업 적합성": round(0.3 + 0.7 * domain_fit, 2),
        "시장성": round(budget_score, 2),
        "고객 적합성": round(customer_score, 2),
        "전략적 가치": round(strategic, 2),
    }
    decision, confidence = _decide(scores, len(retrieved))
    return {
        "decision": decision,
        "confidence": confidence,
        "summary": (
            f"보유 역량 {len(matched_tech)}건이 공고와 일치하고, 예산 규모 "
            f"{_krw(facts.budget_krw)} 기준으로 사업 가치를 평가했다."
        ),
        "strengths": [{"text": s, "grounded": True} for s in strengths],
        "risks": [{"text": r, "grounded": True} for r in risks],
        "criteria_scores": scores,
        "citations": _citations(retrieved),
    }


def technical_opinion(
    facts: BidFacts, company: CompanyProfile, retrieved: list[RetrievedChunk]
) -> dict:
    corpus = _corpus(retrieved)
    strengths: list[str] = []
    risks: list[str] = []

    _, matched, _ = _coverage(company.tech_stack, corpus)
    coverage = _fit(matched)
    if matched:
        strengths.append(f"요구 기술 중 보유 스택 일치: {', '.join(matched[:5])}")
    else:
        risks.append("요구 기술과 보유 스택의 접점이 발췌문에서 확인되지 않음")

    slack = company.max_concurrent_projects - company.current_active_projects
    capacity_score = max(0.0, min(slack / max(company.max_concurrent_projects, 1), 1.0))
    if slack <= 0:
        risks.append("동시 수행 가능 프로젝트 한도 초과 상태")
    else:
        strengths.append(f"동시 수행 여유 {slack}건")

    months = _months(facts.duration)
    budget = facts.budget_krw or 0
    if months is None:
        duration_score = 0.5
        risks.append("사업기간이 명확히 파악되지 않음")
    else:
        # 인당 월 1,200만원을 기준으로 필요한 투입 규모를 역산한다.
        required_person_months = budget / 12_000_000 if budget else 0
        if required_person_months and months:
            team_size = required_person_months / months
            if team_size > company.delivery_headcount * 0.5:
                duration_score = 0.35
                risks.append(
                    f"기간 대비 필요 투입 인력 약 {team_size:.1f}명으로 조직 규모 대비 과중"
                )
            else:
                duration_score = 0.85
                strengths.append(f"{months:.0f}개월 기간에 약 {team_size:.1f}명 투입으로 수행 가능")
        else:
            duration_score = 0.6

    capability_score = min(1.0, 0.4 + 0.6 * coverage)
    scores = {
        "기술 요구사항 충족": round(0.25 + 0.75 * coverage, 2),
        "수행 가능성": round(duration_score, 2),
        "보유 역량": round(capability_score, 2),
        "수행기간 적절성": round(duration_score, 2),
    }
    decision, confidence = _decide(scores, len(retrieved))
    return {
        "decision": decision,
        "confidence": confidence,
        "summary": (
            f"요구 기술과 겹치는 보유 스택 {len(matched)}건, 사업기간 "
            f"{facts.duration or '미확인'} 조건에서 수행 가능성을 평가했다."
        ),
        "strengths": [{"text": s, "grounded": True} for s in strengths],
        "risks": [{"text": r, "grounded": True} for r in risks],
        "criteria_scores": scores,
        "citations": _citations(retrieved),
    }


def finance_opinion(
    facts: BidFacts, company: CompanyProfile, retrieved: list[RetrievedChunk]
) -> dict:
    corpus = _corpus(retrieved)
    strengths: list[str] = []
    risks: list[str] = []

    # 매출이 없으면 자본금을 재무 기준선으로 쓴다. 둘 다 없으면 비율 자체를
    # 계산하지 않는다 — 0으로 나눠 만든 수치는 항상 만점이 되어 무의미하다.
    base_krw, base_label = (
        (company.annual_revenue_krw, "연매출")
        if company.annual_revenue_krw
        else (company.capital_krw, "자본금")
    )

    budget = facts.budget_krw or 0
    if not budget:
        size_score = 0.35
        risks.append("예산 정보를 확인할 수 없어 수익성 판단 근거가 부족함")
    elif company.min_project_budget_krw and budget < company.min_project_budget_krw:
        size_score = 0.3
        risks.append(
            f"예산 {_krw(budget)}이 최소 수주 기준 {_krw(company.min_project_budget_krw)} 미만"
        )
    else:
        size_score = min(1.0, 0.6 + (budget / base_krw if base_krw else 0.2))
        strengths.append(f"예산 규모 {_krw(budget)}로 최소 수주 기준 충족")

    price_competition = any(signal in corpus for signal in LOW_PRICE_SIGNALS)
    quality_competition = any(signal in corpus for signal in QUALITY_SIGNALS)
    if price_competition and not quality_competition:
        margin = company.target_margin * 0.5
        risks.append("최저가/적격심사 방식으로 가격 경쟁 심화 예상")
    elif quality_competition:
        margin = company.target_margin
        strengths.append("기술 중심 평가 방식으로 마진 방어 가능")
    else:
        margin = company.target_margin * 0.8

    profitability = min(1.0, margin / max(company.target_margin, 0.01))

    if budget and base_krw:
        share = budget / base_krw
        roi_score = min(1.0, 0.35 + share * 2)
        strengths.append(f"{base_label} 대비 비중 {share:.1%}")
    else:
        roi_score = 0.5
        risks.append("회사 재무 기준선이 없어 사업 비중을 산출하지 못함")

    # 불리한 조건을 먼저 본다. "선금 없이 준공후 일괄지급"처럼 두 신호가 같이
    # 등장할 때 유리한 쪽만 집어 좋게 읽는 것을 막는다.
    if any(signal in corpus for signal in PAYMENT_BAD):
        payment_score = 0.35
        risks.append("준공 후 일괄 지급 조건으로 현금흐름 부담")
    elif _offers_advance_payment(corpus):
        payment_score = 0.9
        strengths.append("선금/기성 지급 조건으로 현금흐름 부담 완화")
    else:
        payment_score = 0.55
        risks.append("대금 지급 조건이 발췌문에서 확인되지 않음")

    scores = {
        "예산 규모 적정성": round(size_score, 2),
        "예상 수익성": round(profitability, 2),
        "ROI": round(roi_score, 2),
        "대금 지급 조건": round(payment_score, 2),
    }
    decision, confidence = _decide(scores, len(retrieved))
    return {
        "decision": decision,
        "confidence": confidence,
        "summary": (
            f"예산 {_krw(facts.budget_krw)}, 예상 마진 {margin:.0%} 기준으로 수익성을 평가했다."
        ),
        "strengths": [{"text": s, "grounded": True} for s in strengths],
        "risks": [{"text": r, "grounded": True} for r in risks],
        "criteria_scores": scores,
        "citations": _citations(retrieved),
    }


def legal_opinion(
    facts: BidFacts, company: CompanyProfile, retrieved: list[RetrievedChunk]
) -> dict:
    corpus = _corpus(retrieved)
    strengths: list[str] = []
    risks: list[str] = []
    hard_blocker = False

    requirement_lines = facts.qualifications or _requirement_lines(retrieved)
    cert_lines = [line for line in requirement_lines if any(noun in line for noun in CERT_NOUNS)]
    satisfied = [
        line
        for line in cert_lines
        if any(cert.lower() in line.lower() for cert in company.certifications)
    ]
    unmatched = [line for line in cert_lines if line not in satisfied]

    if cert_lines:
        cert_score = len(satisfied) / len(cert_lines)
        if satisfied:
            strengths.append(f"충족 확인된 자격 {len(satisfied)}건")
        for line in unmatched[:3]:
            risks.append(f"보유 여부 미확인 요건: {line[:80]}")
    else:
        cert_score = 0.5
        risks.append("자격요건 조항을 발췌문에서 특정하지 못함")

    if facts.region and company.regions:
        if any(region in facts.region for region in company.regions):
            region_score = 1.0
            strengths.append(f"지역제한 충족: {facts.region}")
        elif "제한" in facts.region or "지역제한" in corpus:
            region_score = 0.0
            hard_blocker = True
            risks.append(f"지역제한 미충족: {facts.region} (본사 소재지 불일치)")
        else:
            region_score = 0.5
            risks.append(f"이행지역 {facts.region}에 대한 지역제한 여부 확인 필요")
    else:
        region_score = 0.6

    found_clauses = [label for token, label in RISK_CLAUSES.items() if token in corpus]
    for clause in found_clauses[:4]:
        risks.append(clause)
    contract_score = max(0.2, 1.0 - 0.15 * len(found_clauses))

    scores = {
        "입찰참가 자격 충족": round(cert_score, 2),
        "계약 위험": round(contract_score, 2),
        "법적 제약": round(region_score, 2),
        "필수 인증 보유": round(cert_score, 2),
    }
    decision, confidence = _decide(scores, len(retrieved))
    if hard_blocker:
        decision = Decision.NO_GO.value
        confidence = max(confidence, 0.8)

    return {
        "decision": decision,
        "confidence": confidence,
        "summary": (
            f"자격요건 {len(satisfied)}/{len(cert_lines)}건 충족 확인, "
            f"계약 위험 조항 {len(found_clauses)}건을 식별했다."
        ),
        "strengths": [{"text": s, "grounded": True} for s in strengths],
        "risks": [{"text": r, "grounded": True} for r in risks],
        "criteria_scores": scores,
        "citations": _citations(retrieved),
    }


def _offers_advance_payment(corpus: str) -> bool:
    """True only for an unnegated 선금/기성 clause.

    Substring matching alone reads "선금은 지급하지 않는다" as a favourable term,
    so each hit is checked against the clause that follows it.
    """
    for token in PAYMENT_GOOD:
        for match in re.finditer(re.escape(token), corpus):
            if not PAYMENT_NEGATION.search(corpus[match.end() : match.end() + 20]):
                return True
    return False


def _requirement_lines(retrieved: list[RetrievedChunk]) -> list[str]:
    lines: list[str] = []
    for chunk in retrieved:
        for line in chunk.text.splitlines():
            cleaned = line.strip(" -*•·\t")
            if len(cleaned) > 6 and any(noun in cleaned for noun in CERT_NOUNS):
                lines.append(cleaned)
    return lines[:10]


def _krw(amount: int | None) -> str:
    if not amount:
        return "미확인"
    if amount >= 10**8:
        return f"{amount / 10**8:.1f}억원"
    if amount >= 10**4:
        return f"{amount / 10**4:.0f}만원"
    return f"{amount:,}원"

from dataclasses import dataclass, field

from app.schemas import AgentRole


@dataclass(frozen=True)
class AgentProfile:
    role: AgentRole
    display_name: str
    perspective: str
    mandate: str
    criteria: tuple[str, ...]
    queries: tuple[str, ...]
    section_boost: tuple[str, ...]
    weight: float
    veto: bool = False
    forbidden_topics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def system_prompt(self) -> str:
        criteria = "\n".join(f"- {c}" for c in self.criteria)
        veto_note = (
            "\n너는 거부권(veto)을 가진다. 필수 요건 위반이 확인되면 다른 관점과 무관하게 NO-GO를 선택하라."
            if self.veto
            else ""
        )
        forbidden = (
            f"\n\n다음 주제는 네 소관이 아니다. 판단에 반영하지 마라: {', '.join(self.forbidden_topics)}."
            if self.forbidden_topics
            else ""
        )
        return (
            f"너는 공공입찰 Go/No-Go 심의위원회의 {self.display_name}이다.\n"
            f"관점: {self.perspective}\n"
            f"임무: {self.mandate}\n\n"
            f"평가 기준:\n{criteria}\n\n"
            "규칙:\n"
            "1. 제공된 발췌문(EVIDENCE)에 근거하지 않은 주장은 하지 마라.\n"
            "2. citations의 chunk_id는 반드시 EVIDENCE에 [cXXX] 형태로 나열된 값만 사용하라.\n"
            "   COMPANY PROFILE과 BID FACTS는 대조용 참고 정보일 뿐 인용 대상이 아니다 — "
            "   그 정보에만 근거한 주장은 citations 없이 strengths/risks에만 적어라.\n"
            "   EVIDENCE에 근거가 없으면 REVIEW로 판단하라.\n"
            "3. 회사 프로필과 공고 내용을 대조하여 판단하라.\n"
            "4. 확신이 없으면 confidence를 낮춰라. 과신하지 마라.\n"
            "5. REVIEW를 안전한 기본값으로 쓰지 마라. criteria_scores의 평균을 근거로 명확히 판단하라:\n"
            "   - 평균 0.65 이상이고 결정적 위험이 없으면 GO\n"
            "   - 평균 0.35 이하이거나 결격·필수요건 위반이 확인되면 NO-GO\n"
            "   - 그 사이이거나, 판단을 가를 핵심 정보가 EVIDENCE에 없을 때만 REVIEW\n"
            "   즉 REVIEW는 '애매해서 피하는 선택'이 아니라 '중간 점수 또는 정보 부족'일 때만 쓰는 판정이다.\n"
            "6. '신고필증 제출', '증빙서류 확인' 같은 서류 제출 여부는 입찰 과정에서 통상적으로 처리되는 절차이며,\n"
            "   그 자체가 EVIDENCE에 없다는 이유만으로 REVIEW를 선택하지 마라. 이런 항목은 risks에 한 줄로\n"
            "   남기되, decision과 criteria_scores는 실제 사업 적합성·역량·리스크 수준으로 매겨라.\n"
            f"{veto_note}{forbidden}\n\n"
            "반드시 아래 JSON 스키마만 출력하라:\n"
            "{\n"
            '  "decision": "GO" | "REVIEW" | "NO-GO",\n'
            '  "confidence": 0.0~1.0,\n'
            '  "summary": "2~3문장 판단 요약",\n'
            '  "strengths": ["..."],\n'
            '  "risks": ["..."],\n'
            '  "criteria_scores": {"기준명": 0.0~1.0},\n'
            '  "citations": [{"chunk_id": "c001", "section": "...", "quote": "원문 인용"}]\n'
            "}"
        )


AGENT_PROFILES: tuple[AgentProfile, ...] = (
    AgentProfile(
        role=AgentRole.SALES,
        display_name="영업 담당 위원",
        perspective="Business Perspective",
        mandate="이 사업이 회사의 사업 방향과 고객 포트폴리오에 맞는지, 수주 가치가 있는지 판단한다.",
        criteria=("사업 적합성", "시장성", "고객 적합성", "전략적 가치"),
        queries=(
            "사업 개요 및 추진 배경",
            "사업 범위와 주요 과업 내용",
            "발주기관 및 수요기관 정보",
            "기대효과 및 사업 목표",
        ),
        section_boost=("개요", "배경", "목적", "과업", "범위"),
        weight=0.25,
        forbidden_topics=("계약 법률 리스크", "상세 원가 계산"),
    ),
    AgentProfile(
        role=AgentRole.TECHNICAL,
        display_name="기술 담당 위원",
        perspective="Engineering Perspective",
        mandate="요구 기술을 우리 역량으로 기간 내 수행 가능한지 판단한다.",
        criteria=("기술 요구사항 충족", "수행 가능성", "보유 역량", "수행기간 적절성"),
        queries=(
            "기술 요구사항 및 규격",
            "과업 수행 범위와 산출물",
            "사업기간 및 일정 계획",
            "요구 인력 구성 및 투입 기준",
        ),
        section_boost=("기술", "요구", "과업", "일정", "인력", "산출물"),
        weight=0.25,
        forbidden_topics=("수익성 판단", "시장성 판단"),
    ),
    AgentProfile(
        role=AgentRole.FINANCE,
        display_name="재무 담당 위원",
        perspective="Financial Perspective",
        mandate="예산 규모, 예상 수익성, ROI 관점에서 참여 타당성을 판단한다.",
        criteria=("예산 규모 적정성", "예상 수익성", "ROI", "대금 지급 조건"),
        queries=(
            "사업 예산 및 추정가격",
            "대가 지급 방법과 지급 조건",
            "입찰 방식 및 낙찰자 결정 기준",
            "계약 보증금 및 하자보수 보증",
        ),
        section_boost=("예산", "금액", "대가", "지급", "낙찰", "보증"),
        weight=0.25,
        forbidden_topics=("기술 구현 난이도 상세 판단",),
    ),
    AgentProfile(
        role=AgentRole.LEGAL,
        display_name="법무 담당 위원",
        perspective="Compliance Perspective",
        mandate="입찰참가 자격을 충족하는지, 계약상 법적 위험이 수용 가능한지 판단한다.",
        criteria=("입찰참가 자격 충족", "계약 위험", "법적 제약", "필수 인증 보유"),
        queries=(
            "입찰참가 자격요건 및 제한사항",
            "필수 인증 및 면허 요건",
            "지역제한 및 실적 제한",
            "계약 조건과 지체상금 손해배상 조항",
        ),
        section_boost=("자격", "제한", "인증", "계약", "준수", "법"),
        weight=0.25,
        veto=True,
        forbidden_topics=("시장성 판단", "수익성 판단"),
    ),
)


def get_profile(role: AgentRole) -> AgentProfile:
    return next(profile for profile in AGENT_PROFILES if profile.role == role)

import { useEffect } from "react";
import { krw } from "../lib/format";

// The profile shape varies by corpus — a source that ships no technical head
// count or margin must render as "—", never crash the page that embeds it.
const count = (value) =>
  typeof value === "number" ? `${value.toLocaleString()}명` : "—";

function Field({ label, value, hint }) {
  return (
    <div>
      <dt className="text-xs tracking-wide text-slate-500 uppercase">{label}</dt>
      <dd className="mt-0.5 font-mono text-sm text-slate-100">{value ?? "—"}</dd>
      {hint && <p className="text-xs text-slate-600">{hint}</p>}
    </div>
  );
}

function Chips({ label, items, tone = "border-slate-700 bg-slate-800/60 text-slate-300", hint }) {
  if (!items?.length) return null;
  return (
    <section>
      <div className="mb-1.5 flex items-baseline gap-2">
        <h3 className="text-xs font-semibold tracking-wide text-slate-400 uppercase">{label}</h3>
        <span className="font-mono text-xs text-slate-600">{items.length}</span>
      </div>
      {hint && <p className="mb-1.5 text-xs leading-relaxed text-slate-500">{hint}</p>}
      <ul className="flex flex-wrap gap-1.5">
        {items.map((item, i) => (
          <li key={i} className={`rounded border px-2 py-0.5 text-[13px] ${tone}`}>
            {item}
          </li>
        ))}
      </ul>
    </section>
  );
}

/**
 * The profile every agent is judging against. It is worth showing in full:
 * the screening tiers only make sense once you can see which capabilities and
 * thresholds produced them.
 */
export default function CompanyProfile({ company, onClose }) {
  useEffect(() => {
    const onKey = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!company) return null;

  const active =
    typeof company.max_concurrent_projects === "number"
      ? `${company.current_active_projects ?? 0} / ${company.max_concurrent_projects}건`
      : "—";

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-950/80 p-4 backdrop-blur-sm print:hidden"
      onClick={onClose}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="자사 프로필"
        onClick={(event) => event.stopPropagation()}
        className="my-8 w-full max-w-3xl rounded-xl border border-slate-700 bg-slate-900 shadow-[0_24px_60px_rgb(0_0_0_/_0.5)]"
      >
        <header className="flex items-start justify-between gap-4 border-b border-slate-800 px-5 py-4">
          <div className="min-w-0">
            <p className="text-xs font-semibold tracking-[0.16em] text-sky-400 uppercase">
              심의 대상 기업
            </p>
            <h2 className="mt-1 text-xl font-bold text-slate-50">{company.name}</h2>
            {company.business_type && (
              <p className="mt-0.5 text-sm text-slate-400">{company.business_type}</p>
            )}
          </div>
          <button
            onClick={onClose}
            aria-label="닫기"
            className="shrink-0 rounded-md border border-slate-700 px-2.5 py-1 text-sm text-slate-400 hover:border-slate-600 hover:text-slate-100"
          >
            닫기 ✕
          </button>
        </header>

        <div className="space-y-5 px-5 py-4">
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            <Field label="자본금" value={company.capital_krw ? krw(company.capital_krw) : "—"} />
            <Field label="임직원" value={count(company.headcount)} />
            <Field
              label="기술인력"
              value={count(company.technical_headcount)}
              hint="수행능력 판단 기준"
            />
            <Field label="최소 수주금액" value={krw(company.min_project_budget_krw)} />
            <Field
              label="목표 마진"
              value={
                typeof company.target_margin === "number"
                  ? `${Math.round(company.target_margin * 100)}%`
                  : "—"
              }
            />
            <Field label="동시 수행" value={active} />
            <Field
              label="최소 준비기간"
              value={
                typeof company.min_preparation_days === "number"
                  ? `${company.min_preparation_days}일`
                  : "—"
              }
            />
            {company.annual_revenue_krw > 0 && (
              <Field label="연매출" value={krw(company.annual_revenue_krw)} />
            )}
          </dl>

          <Chips
            label="주력 분야"
            items={company.preferred_categories}
            tone="border-sky-500/40 bg-sky-500/10 text-sky-200"
            hint="이 분야를 벗어난 공고는 차단하지 않고 경고로 표시한 뒤 위원회로 넘긴다."
          />

          <Chips label="영업 지역" items={company.regions} />

          <Chips
            label="매칭 키워드"
            items={company.tech_stack}
            tone="border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
            hint="수행 실적에서 도출한 역량 키워드. 공고 본문과 겹치는 개수가 영업·기술 위원의 적합성 점수를 만든다."
          />

          <Chips label="보유 면허 · 인증" items={company.certifications} />

          {company.past_projects?.length > 0 && (
            <section>
              <div className="mb-1.5 flex items-baseline gap-2">
                <h3 className="text-xs font-semibold tracking-wide text-slate-400 uppercase">
                  수행 실적
                </h3>
                <span className="font-mono text-xs text-slate-600">
                  {company.past_projects.length}
                </span>
              </div>
              <ul className="space-y-1">
                {company.past_projects.map((project, i) => (
                  <li key={i} className="flex gap-2 text-[13px] text-slate-300">
                    <span className="font-mono text-slate-600">{String(i + 1).padStart(2, "0")}</span>
                    <span>{project}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {company.industry_codes?.length > 0 && (
            <Chips label="보유 업종코드" items={company.industry_codes} />
          )}
        </div>
      </div>
    </div>
  );
}

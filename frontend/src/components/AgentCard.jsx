import { useState } from "react";
import { decisionStyle, percent, score10 } from "../lib/format";
import useFillWidth from "../lib/useFillWidth";
import CommitteeAvatar from "./CommitteeAvatar";

function ConfidenceBar({ value, tone }) {
  const width = useFillWidth(Math.round((value ?? 0) * 100));
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
      <div
        className={`h-full rounded-full ${tone} transition-[width] duration-700 ease-out`}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

function CriteriaBar({ score }) {
  const width = useFillWidth(Math.round(score * 100));
  return (
    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-800">
      <div
        className="h-full rounded-full bg-slate-500 transition-[width] duration-700 ease-out"
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

function Criteria({ scores }) {
  const entries = Object.entries(scores || {});
  if (!entries.length) return null;
  return (
    <div className="space-y-1.5">
      {entries.map(([name, score]) => (
        <div key={name} className="flex items-center gap-2">
          <span className="w-28 shrink-0 truncate text-[13px] text-slate-400">{name}</span>
          <CriteriaBar score={score} />
          <span className="w-14 text-right font-mono text-[13px] text-slate-400">
            {score10(score)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function AgentCard({ profile, opinion, status }) {
  const [showEvidence, setShowEvidence] = useState(false);
  const style = opinion ? decisionStyle(opinion.decision) : null;

  return (
    <article
      className={`flex flex-col rounded-xl border bg-slate-900/40 transition-colors ${
        opinion ? `${style.border} fade-in-up` : "border-slate-800"
      } ${status === "running" ? "pulse-ring" : ""}`}
    >
      <header className="flex items-start justify-between gap-3 border-b border-slate-800 px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <CommitteeAvatar role={profile.role} size={44} />
          <div className="min-w-0">
            <h3 className="flex items-center gap-2 text-base font-semibold text-slate-100">
              {profile.display_name}
              {profile.veto && (
                <span className="rounded bg-rose-500/15 px-1.5 py-0.5 text-xs font-medium text-rose-300">
                  VETO
                </span>
              )}
            </h3>
            <p className="mt-0.5 truncate text-[13px] text-slate-400">{profile.perspective}</p>
          </div>
        </div>
        {opinion ? (
          <span
            className={`shrink-0 rounded-md px-2.5 py-1 text-sm font-bold ${style.bg} ${style.text}`}
          >
            {opinion.decision}
          </span>
        ) : (
          <span className="shrink-0 text-[13px] text-slate-500">
            {status === "running" ? "분석 중…" : "대기"}
          </span>
        )}
      </header>

      <div className="flex-1 space-y-3 p-4">
        {!opinion && (
          <div className="space-y-2">
            <div className="h-2 w-3/4 animate-pulse rounded bg-slate-800" />
            <div className="h-2 w-full animate-pulse rounded bg-slate-800" />
            <div className="h-2 w-2/3 animate-pulse rounded bg-slate-800" />
          </div>
        )}

        {opinion && (
          <>
            <div>
              <div className="mb-1 flex items-baseline justify-between text-[13px]">
                <span className="text-slate-400">Confidence</span>
                <span className={`font-mono font-semibold ${style.text}`}>
                  {percent(opinion.confidence)}
                </span>
              </div>
              <ConfidenceBar value={opinion.confidence} tone={style.dot} />
            </div>

            <p className="text-sm leading-relaxed text-slate-200">{opinion.summary}</p>

            <Criteria scores={opinion.criteria_scores} />

            {opinion.strengths?.length > 0 && (
              <ul className="space-y-1">
                {opinion.strengths.map((item, i) => (
                  <li
                    key={i}
                    className={`flex gap-1.5 text-[13px] ${
                      item.grounded ? "text-emerald-300/90" : "text-slate-400"
                    }`}
                  >
                    <span>+</span>
                    <span>
                      {item.text}
                      {!item.grounded && (
                        <span className="ml-1.5 rounded bg-slate-800 px-1 py-0.5 text-[11px] text-slate-500">
                          회사 프로필 기반 · 공고 근거 아님
                        </span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            )}

            {opinion.risks?.length > 0 && (
              <ul className="space-y-1">
                {opinion.risks.map((item, i) => (
                  <li
                    key={i}
                    className={`flex gap-1.5 text-[13px] ${
                      item.grounded ? "text-rose-300/90" : "text-slate-400"
                    }`}
                  >
                    <span>!</span>
                    <span>
                      {item.text}
                      {!item.grounded && (
                        <span className="ml-1.5 rounded bg-slate-800 px-1 py-0.5 text-[11px] text-slate-500">
                          회사 프로필 기반 · 공고 근거 아님
                        </span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            )}

            {opinion.guardrail_flags?.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {opinion.guardrail_flags.map((flag, i) => (
                  <span
                    key={i}
                    className="rounded bg-amber-500/10 px-1.5 py-0.5 font-mono text-xs text-amber-300"
                  >
                    {flag}
                  </span>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {opinion?.citations?.length > 0 && (
        <footer className="border-t border-slate-800 px-4 py-2.5 print:hidden">
          <button
            onClick={() => setShowEvidence((prev) => !prev)}
            className="flex w-full items-center justify-between text-[13px] text-slate-400 hover:text-slate-200"
          >
            <span>근거 {opinion.citations.length}건</span>
            <span>{showEvidence ? "−" : "+"}</span>
          </button>
          {showEvidence && (
            <ul className="mt-2 space-y-2">
              {opinion.citations.map((citation, i) => (
                <li key={i} className="rounded-md border border-slate-800 bg-slate-950 p-2">
                  <div className="mb-1 flex items-center gap-2">
                    <span className="rounded bg-sky-500/10 px-1.5 font-mono text-xs text-sky-300">
                      {citation.chunk_id}
                    </span>
                    <span className="truncate text-xs text-slate-500">{citation.section}</span>
                  </div>
                  <p className="text-[13px] leading-relaxed text-slate-400">"{citation.quote}"</p>
                </li>
              ))}
            </ul>
          )}
        </footer>
      )}
    </article>
  );
}

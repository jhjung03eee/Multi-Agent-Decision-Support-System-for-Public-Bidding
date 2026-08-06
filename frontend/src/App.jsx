import { useEffect, useRef, useState } from "react";
import CompanyProfile from "./components/CompanyProfile";
import ReviewView from "./components/ReviewView";
import ScreeningView from "./components/ScreeningView";
import { getConfig, getCorpus, getHealth, getSamples, runScreening, streamReview } from "./lib/api";

const STAGE_TO_STEP = {
  parsing: "parsing",
  indexing: "indexing",
  agent_completed: "agents",
  committee_completed: "committee",
  validation: "validation",
  completed: "completed",
};

const initialRun = {
  facts: null,
  opinions: {},
  running: {},
  committee: null,
  metrics: null,
  steps: [],
  log: [],
};

const TABS = [
  { key: "screen", label: "배치 스크리닝", icon: "📋" },
  { key: "review", label: "단건 심의", icon: "🔎" },
];

export default function App() {
  const [tab, setTab] = useState("screen");
  const [config, setConfig] = useState(null);
  const [health, setHealth] = useState(null);
  const [samples, setSamples] = useState([]);
  const [corpus, setCorpus] = useState(null);
  const [report, setReport] = useState(null);
  const [run, setRun] = useState(initialRun);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [showProfile, setShowProfile] = useState(false);
  const abortRef = useRef(null);

  useEffect(() => {
    Promise.all([getConfig(), getHealth(), getSamples(), getCorpus()])
      .then(([configData, healthData, sampleData, corpusData]) => {
        setConfig(configData);
        setHealth(healthData);
        setSamples(sampleData.samples);
        setCorpus(corpusData);
      })
      .catch((err) => setError(err.message));
    return () => abortRef.current?.abort();
  }, []);

  const startScreening = async () => {
    setError("");
    setBusy(true);
    try {
      setReport(await runScreening());
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleEvent = (event) => {
    setRun((prev) => {
      const next = {
        ...prev,
        log: [...prev.log, event.message].slice(-30),
        steps: STAGE_TO_STEP[event.stage]
          ? [...new Set([...prev.steps, STAGE_TO_STEP[event.stage]])]
          : prev.steps,
      };

      switch (event.stage) {
        case "parsing":
          next.facts = event.payload.facts;
          break;
        case "agent_started":
          next.running = { ...prev.running, [event.payload.role]: true };
          break;
        case "agent_completed": {
          const opinion = event.payload.opinion;
          next.opinions = { ...prev.opinions, [opinion.role]: opinion };
          next.running = { ...prev.running, [opinion.role]: false };
          break;
        }
        case "committee_completed":
          next.committee = event.payload.committee;
          break;
        case "validation":
          next.metrics = event.payload.metrics;
          break;
        default:
          break;
      }
      return next;
    });
  };

  const start = async (payload) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError("");
    setRun(initialRun);
    setBusy(true);
    try {
      await streamReview(payload, handleEvent, controller.signal);
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl space-y-5 p-4 lg:p-6">
      <header className="relative overflow-hidden border-b border-slate-800/90 pb-5">
        <div className="brand-gradient-bar absolute inset-x-0 top-0 print:hidden" />
        <div className="flex flex-wrap items-end justify-between gap-5 pt-5">
          <div className="min-w-0">
            <p className="mb-2 text-xs font-semibold tracking-[0.16em] text-sky-400 uppercase">
              Public Bidding Decision Desk
            </p>
            <h1 className="text-2xl font-bold tracking-tight text-slate-50 sm:text-3xl">
              🏛️ AI Go/No-Go 심의위원회
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
              공공 입찰공고 참여 여부를 영업·기술·재무·법무 4개 관점에서 종합 심의합니다.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {config && (
              <button
                onClick={() => setShowProfile(true)}
                title="자사 프로필 상세 보기 — 심의 기준이 되는 역량·자격·임계값"
                className="rounded-md border border-slate-700 bg-slate-900/70 px-3 py-1.5 text-slate-400 transition-colors hover:border-sky-500/50 hover:text-slate-200 print:hidden"
              >
                심의 대상 <strong className="ml-1 font-medium text-slate-100">{config.company.name}</strong>
                <span aria-hidden className="ml-1.5 text-slate-500">ⓘ</span>
              </button>
            )}
            {health && (
              <span
                className={`rounded-md border px-3 py-1.5 font-mono text-xs ${
                  health.llm_provider === "heuristic-offline"
                    ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
                    : "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                }`}
                title="현재 백엔드 AI 실행 모드"
              >
                {health.llm_provider === "heuristic-offline" ? "● OFFLINE" : `● ${health.model}`}
              </span>
            )}
          </div>
        </div>
      </header>

      <nav className="flex w-fit gap-1 rounded-lg border border-slate-800 bg-slate-900/60 p-1 print:hidden" aria-label="심의 모드">
        {TABS.map((entry) => (
          <button
            key={entry.key}
            onClick={() => setTab(entry.key)}
            aria-current={tab === entry.key ? "page" : undefined}
            className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              tab === entry.key
                ? "bg-sky-500/15 text-sky-200 shadow-sm"
                : "text-slate-500 hover:bg-slate-800/70 hover:text-slate-300"
            }`}
          >
            <span aria-hidden className="mr-1.5">{entry.icon}</span>{entry.label}
          </button>
        ))}
      </nav>

      {error && (
        <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-2.5 text-sm text-rose-300 print:hidden">
          {error}
        </div>
      )}

      {tab === "screen" && (
        <ScreeningView
          corpus={corpus}
          report={report}
          busy={busy}
          onRun={startScreening}
        />
      )}

      {tab === "review" && (
        <ReviewView
          config={config}
          samples={samples}
          run={run}
          busy={busy}
          onRun={start}
          onError={setError}
        />
      )}

      <footer className="pt-2 pb-6 text-center text-xs text-slate-600 print:hidden">
        Project 08 · Multi-Agent Decision Support System for Public Bidding
      </footer>

      {showProfile && (
        <CompanyProfile company={config?.company} onClose={() => setShowProfile(false)} />
      )}
    </div>
  );
}

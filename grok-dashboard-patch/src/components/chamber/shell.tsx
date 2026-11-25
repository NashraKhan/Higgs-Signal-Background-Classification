import { useEffect, useState } from "react";
import { Activity, Atom, BookOpen, Crosshair, Images, Layers } from "lucide-react";
import { Collider } from "@/components/chamber/collider";
import { Explainer } from "@/components/chamber/explainer";
import { Bench, Hall, Ledger, Measure, Wall } from "@/components/chamber/stages";
import type { Stage, Tone } from "@/data/higgs";

const API_BASE = import.meta.env.VITE_HIGGS_API_URL ?? "http://127.0.0.1:8000";
const MODEL_OPTIONS = [
  { id: "xgboost", label: "XGBoost · 5,000 rounds" },
  { id: "catboost", label: "CatBoost" },
  { id: "lightgbm", label: "LightGBM" },
  { id: "logistic_regression", label: "Logistic Regression" },
  { id: "mlp", label: "Neural Network (MLP)" },
] as const;

type Prediction = {
  index: number;
  model: string;
  class_name: "Signal" | "Background";
  prediction: 0 | 1;
  p_signal: number;
  confidence: number;
  actual_label: number | null;
  sample_type: "held_out_test" | "source_dataset_demo" | "synthetic_demo";
  source: string;
};

const rooms: { id: Stage; label: string; icon: typeof Atom }[] = [
  { id: "hall", label: "Hall", icon: Atom },
  { id: "measure", label: "Measure", icon: Crosshair },
  { id: "bench", label: "Bench", icon: Layers },
  { id: "ledger", label: "Ledger", icon: Activity },
  { id: "wall", label: "Wall", icon: Images },
];

export function Chamber() {
  const [stage, setStage] = useState<Stage>("hall");
  const [showGuide, setShowGuide] = useState(false);
  const [burst, setBurst] = useState(0);
  const [tone, setTone] = useState<Tone>("signal");
  const [selectedModel, setSelectedModel] = useState<string>("xgboost");
  const [busy, setBusy] = useState(false);
  const [apiReady, setApiReady] = useState(false);
  const [apiError, setApiError] = useState("");
  const [lastPrediction, setLastPrediction] = useState<Prediction | null>(null);

  useEffect(() => {
    fetch(API_BASE + "/health")
      .then((response) => {
        if (!response.ok) throw new Error("Inference API is not responding.");
        return response.json();
      })
      .then((data) => {
        setApiReady(true);
        const current = data.models?.find((item: { id: string; available: boolean }) => item.id === selectedModel);
        if (current && !current.available) setApiError(current.id + " checkpoint is not available locally.");
        else setApiError("");
      })
      .catch(() => {
        setApiReady(false);
        setApiError("Start the Python inference API on http://127.0.0.1:8000 before releasing an event.");
      });
  }, [selectedModel]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return;
      const index = Number(event.key) - 1;
      if (index >= 0 && index < rooms.length) { setStage(rooms[index].id); setShowGuide(false); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function release() {
    if (busy) return;
    setBusy(true);
    setApiError("");
    try {
      const sampleResponse = await fetch(API_BASE + "/sample");
      const sample = await sampleResponse.json();
      if (!sampleResponse.ok) throw new Error(sample.detail ?? "Could not load a held-out test event.");

      const predictionResponse = await fetch(API_BASE + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: selectedModel, features: sample.features, threshold: 0.5 }),
      });
      const result = await predictionResponse.json();
      if (!predictionResponse.ok) throw new Error(result.detail ?? "Model inference failed.");

      setTone(result.prediction === 1 ? "signal" : "background");
      setBurst((value) => value + 1);
      setLastPrediction({
        index: sample.index, model: result.model, class_name: result.class_name,
        prediction: result.prediction, p_signal: result.p_signal, confidence: result.confidence,
        actual_label: sample.actual_label, sample_type: sample.sample_type ?? "held_out_test", source: sample.source,
      });
      setApiReady(true);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }

  const selectedLabel = MODEL_OPTIONS.find((model) => model.id === selectedModel)?.label ?? selectedModel;

  return (
    <main className="relative min-h-dvh">
      <Collider burst={burst} tone={tone} />
      <div className="relative z-10 flex min-h-dvh flex-col">
        <header className="flex flex-wrap items-center justify-between gap-3 px-5 py-4 md:px-12">
          <p className="kicker text-sm font-medium text-cream">
            Higgs <span className="text-mist">Chamber</span>
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <span className={apiReady ? "text-xs text-ion" : "text-xs text-copper"}>
              {apiReady ? "● INFERENCE API CONNECTED" : "● API NOT CONNECTED"}
            </span>
            <label className="text-xs text-mist">
              Model{" "}
              <select
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                className="ml-1 rounded-full border border-line bg-void/90 px-3 py-2 text-cream"
              >
                {MODEL_OPTIONS.map((model) => (
                  <option key={model.id} value={model.id}>{model.label}</option>
                ))}
              </select>
            </label>
          </div>
        </header>

        <div className="flex-1 pb-28">
          {stage === "hall" ? <Hall onRelease={release} tone={tone} /> : null}
          {!showGuide && stage === "hall" ? <div aria-hidden="true" className="h-[62vh] min-h-[380px]" /> : null}\n          {!showGuide && lastPrediction && stage === "hall" ? (
            <div className="relative z-10 mx-5 mb-5 max-w-2xl rounded-2xl border border-ion/30 bg-void/85 p-4 backdrop-blur md:mx-12">
              <p className="kicker text-xs text-ion">
                Live model inference · {lastPrediction.sample_type === "held_out_test" ? "held-out test event" : lastPrediction.sample_type === "source_dataset_demo" ? "source-dataset demo event" : "synthetic demonstration event"} #{lastPrediction.index.toLocaleString()}
              </p>
              <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
                <div>
                  <p className="display text-4xl text-cream">{lastPrediction.class_name.toUpperCase()}</p>
                  <p className="mt-1 text-xs text-mist">{selectedLabel} · 28 input features</p>
                </div>
                <div className="text-right">
                  <p className="num text-3xl text-cream">{(lastPrediction.confidence * 100).toFixed(2)}%</p>
                  <p className="text-xs text-mist">Confidence in predicted class</p>
                </div>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                <p className="text-mist">P(Signal) <strong className="ml-2 text-cream">{(lastPrediction.p_signal * 100).toFixed(2)}%</strong></p>
                <p className="text-mist">P(Background) <strong className="ml-2 text-cream">{((1 - lastPrediction.p_signal) * 100).toFixed(2)}%</strong></p>
              </div>
              {lastPrediction.actual_label !== null ? (
                <p className="mt-3 text-sm text-mist">
                  {lastPrediction.sample_type === "held_out_test" ? "Actual held-out label:" : "Source dataset label (not held-out):"} <strong className="text-cream">{lastPrediction.actual_label === 1 ? "Signal (1)" : "Background (0)"}</strong>
                  {" · "}
                  <strong className={lastPrediction.actual_label === lastPrediction.prediction ? "text-ion" : "text-copper"}>
                    {lastPrediction.actual_label === lastPrediction.prediction ? "Correct prediction" : "Different from actual label"}
                  </strong>
                </p>
              ) : (
                <p className="mt-3 text-xs text-mist">No ground-truth label was present in the local sample file.</p>
              )}
              <p className="mt-2 break-all text-[11px] text-mist/70">Source: {lastPrediction.source}</p>
            </div>
          ) : null}
          {!showGuide && stage !== "hall" ? (
            <>
              {stage === "measure" ? <Measure /> : null}
              {stage === "bench" ? <Bench /> : null}
              {stage === "ledger" ? <Ledger /> : null}
              {stage === "wall" ? <Wall /> : null}
            </>
          ) : null}
        </div>

        {apiError ? (
          <div role="alert" className="relative z-20 mx-5 mb-24 max-w-2xl rounded-xl border border-copper/40 bg-void/90 p-3 text-sm text-copper md:mx-12">
            {apiError}
          </div>
        ) : null}

        <nav className="fixed inset-x-3 bottom-4 z-20 mx-auto max-w-2xl" aria-label="Rooms and guide">
          <div className="dock flex justify-between rounded-full px-2 py-2">
            <button type="button" onClick={() => setShowGuide(true)} className={showGuide ? "flex min-h-11 min-w-11 flex-1 flex-col items-center justify-center gap-1 rounded-full text-ion" : "flex min-h-11 min-w-11 flex-1 flex-col items-center justify-center gap-1 rounded-full text-mist"} aria-current={showGuide ? "page" : undefined}>\n              <BookOpen className="size-4" aria-hidden="true" />\n              <span className="text-xs leading-none">Guide</span>\n            </button>\n            {rooms.map((room) => {
              const Icon = room.icon;
              const active = stage === room.id;
              return (
                <button key={room.id} type="button" onClick={() => { setStage(room.id); setShowGuide(false); }}
                  className={active
                    ? "flex min-h-11 min-w-11 flex-1 flex-col items-center justify-center gap-1 rounded-full text-ion"
                    : "flex min-h-11 min-w-11 flex-1 flex-col items-center justify-center gap-1 rounded-full text-mist"}
                  aria-current={active ? "page" : undefined}>
                  <Icon className="size-4" aria-hidden="true" />
                  <span className="text-xs leading-none">{room.label}</span>
                </button>
              );
            })}
          </div>
        </nav>
      </div>
    </main>
  );
}

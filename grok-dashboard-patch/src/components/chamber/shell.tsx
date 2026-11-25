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
  { id: "logistic_regression", label: "Logistic Regression" },
] as const;

type ModelPrediction = {
  model: string;
  class_name: "Signal" | "Background";
  prediction: 0 | 1;
  p_signal: number;
  confidence: number;
};

type Prediction = {
  index: number;
  modelResults: ModelPrediction[];
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
  const [busy, setBusy] = useState(false);
  const [apiReady, setApiReady] = useState(false);
  const [readyModels, setReadyModels] = useState<string[]>([]);
  const [apiError, setApiError] = useState("");
  const [lastPrediction, setLastPrediction] = useState<Prediction | null>(null);

  useEffect(() => {
    fetch(API_BASE + "/health")
      .then((response) => {
        if (!response.ok) throw new Error("Inference API is not responding.");
        return response.json();
      })
      .then((data) => {
        const models = Array.isArray(data.models) ? data.models : [];
        const ready = models
          .filter((item: { id: string; ready?: boolean }) => item.ready === true)
          .map((item: { id: string }) => item.id);
        setReadyModels(ready);
        setApiReady(true);
        setApiError(
          ready.length === MODEL_OPTIONS.length
            ? ""
            : "Some inference models are not ready. Check the API health response before the demo."
        );
      })
      .catch(() => {
        setApiReady(false);
        setApiError("Start the Python inference API on http://127.0.0.1:8000 before releasing an event.");
      });
  }, []);

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
      if (!sampleResponse.ok) throw new Error(sample.detail ?? "Could not load a demo event.");

      // Run every available model against the exact same 28-feature input.
      const modelResults: ModelPrediction[] = [];
      for (const model of MODEL_OPTIONS) {
        const response = await fetch(API_BASE + "/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: model.id, features: sample.features, threshold: 0.5 }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(model.label + ": " + (result.detail ?? "Inference failed."));
        modelResults.push({
          model: result.model,
          class_name: result.class_name,
          prediction: result.prediction,
          p_signal: result.p_signal,
          confidence: result.confidence,
        });
      }

      // The collider animation follows XGBoost; the result card shows every model.
      setTone(modelResults[0].prediction === 1 ? "signal" : "background");
      setBurst((value) => value + 1);
      setLastPrediction({
        index: sample.index,
        modelResults,
        actual_label: sample.actual_label,
        sample_type: sample.sample_type ?? "held_out_test",
        source: sample.source,
      });
      setApiReady(true);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }

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
              {apiReady ? (readyModels.length > 0 ? "● API ONLINE · " + readyModels.length + "/" + MODEL_OPTIONS.length + " MODELS READY" : "● API ONLINE · READINESS NOT CHECKED") : "● API NOT CONNECTED"}
            </span>
          </div>
        </header>

        <div className="flex-1 pb-28">
          {!showGuide && stage === "hall" ? <Hall onRelease={release} tone={tone} /> : null}
          {showGuide ? <Explainer /> : null}
          {!showGuide && stage === "hall" ? <div aria-hidden="true" className="h-[62vh] min-h-[380px]" /> : null}
          {!showGuide && lastPrediction && stage === "hall" ? (
            <section className="relative z-10 mx-5 mb-5 max-w-3xl rounded-2xl border border-copper/30 bg-void/90 p-4 backdrop-blur md:mx-12">
              <p className="kicker text-xs text-copper">
                {lastPrediction.sample_type === "synthetic_demo"
                  ? "GENERATED DEMO EVENT"
                  : lastPrediction.sample_type === "held_out_test"
                    ? "HELD-OUT TEST EVENT"
                    : "DATASET DEMO EVENT"}{" "}
                · #{lastPrediction.index.toLocaleString()}
              </p>
              <h2 className="mt-2 text-lg font-semibold text-cream">Three-model readout</h2>
              <p className="mt-1 text-xs text-mist">
                Same 28 features · predicted class and model score
              </p>
              <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                {lastPrediction.modelResults.map((item) => {
                  const modelLabel = MODEL_OPTIONS.find((option) => option.id === item.model)?.label ?? item.model;
                  return (
                    <div key={item.model} className="rounded-xl border border-line bg-white/[0.035] p-3">
                      <p className="text-xs font-medium leading-5 text-mist">{modelLabel}</p>
                      <p className="display mt-2 text-xl text-cream">{item.class_name.toUpperCase()}</p>
                      <p className="num mt-1 text-2xl text-cream">{(item.confidence * 100).toFixed(2)}%</p>
                      <p className="text-[10px] leading-4 text-mist">Predicted-class probability</p>
                      <div className="mt-3 space-y-1 text-[11px] text-mist">
                        <p className="flex justify-between gap-2">
                          P(Signal)<strong className="text-cream">{(item.p_signal * 100).toFixed(2)}%</strong>
                        </p>
                        <p className="flex justify-between gap-2">
                          P(Background)<strong className="text-cream">{((1 - item.p_signal) * 100).toFixed(2)}%</strong>
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
              {lastPrediction.actual_label !== null ? (
                <p className="mt-3 text-xs text-mist">
                  Dataset label:{" "}
                  <strong className="text-cream">{lastPrediction.actual_label === 1 ? "Signal" : "Background"}</strong>
                </p>
              ) : (
                <p className="mt-3 text-[10px] text-mist">Synthetic input · no ground-truth label</p>
              )}
            </section>
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
            <button type="button" onClick={() => setShowGuide(true)} className={showGuide ? "flex min-h-11 min-w-11 flex-1 flex-col items-center justify-center gap-1 rounded-full text-ion" : "flex min-h-11 min-w-11 flex-1 flex-col items-center justify-center gap-1 rounded-full text-mist"} aria-current={showGuide ? "page" : undefined}>
              <BookOpen className="size-4" aria-hidden="true" />
              <span className="text-xs leading-none">Guide</span>
            </button>
            {rooms.map((room) => {
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

import { useState } from "react";
import { Atom, BrainCircuit, ChevronRight, GitBranch, Layers3, Network, Play, RotateCcw, ScanLine, Trees } from "lucide-react";

const models = [
  { id: "lr", name: "Logistic Regression", icon: GitBranch, label: "Weighted scorecard", detail: "Learns a weight for each measurement, combines them into one score, then estimates a probability." },
  { id: "xgb", name: "XGBoost", icon: Trees, label: "Boosted decision trees", detail: "Builds small decision trees in sequence. Each new tree focuses on patterns the earlier trees did not capture well." },
  { id: "cat", name: "CatBoost", icon: Layers3, label: "Boosted decision trees", detail: "Combines many decision trees in stages, with techniques designed to make the learning process robust." },
  { id: "lgb", name: "LightGBM", icon: ScanLine, label: "Fast tree learning", detail: "Uses efficient, histogram-based tree building to find useful splits in large datasets." },
  { id: "mlp", name: "Neural Network (MLP)", icon: Network, label: "Layers of connected units", detail: "Passes the 28 values through connected layers that learn combinations of clues." },
];

export function Explainer() {
  const [run, setRun] = useState(0);
  const [showTracks, setShowTracks] = useState(true);
  const [model, setModel] = useState("xgb");
  const active = models.find((item) => item.id === model) ?? models[1];
  const ModelIcon = active.icon;

  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-5 pb-32 pt-7 text-cream md:px-10">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="kicker mb-2 text-[10px] tracking-[0.22em] text-ion">START HERE · 2 MINUTES</p>
          <h1 className="display text-3xl leading-tight md:text-5xl">Higgs signal, <span className="text-ion">made visual.</span></h1>
          <p className="mt-2 max-w-2xl text-xs leading-5 text-mist md:text-sm">A collision leaves clues. Five machine-learning approaches study those clues and estimate: signal or background?</p>
        </div>
        <div className="rounded-full border border-line px-3 py-1.5 text-[10px] text-mist">A visual guide · no physics background needed</div>
      </header>

      <div className="grid items-stretch gap-4 lg:grid-cols-[1.35fr_0.65fr]">
        <div className="overflow-hidden rounded-2xl border border-line bg-[#07121d]">
          <div className="flex items-center justify-between px-4 pt-3">
            <div><p className="kicker text-[10px] tracking-[0.18em] text-ion">01 · INSIDE A COLLISION</p><p className="mt-1 text-xs text-mist">Run the event to see the tracks emerge.</p></div>
            <button onClick={() => setRun((n) => n + 1)} className="inline-flex items-center gap-1.5 rounded-full bg-ion px-3 py-2 text-[11px] font-semibold text-[#06111a]"><Play className="size-3"/> Run collision</button>
          </div>
          <div className="px-2 pb-1 pt-2">
            <svg key={run} viewBox="0 0 720 300" role="img" aria-label="Animated schematic of two particle beams colliding, producing outgoing particle tracks inside detector rings" className="h-auto w-full">
              <defs>
                <radialGradient id="eventGlow"><stop offset="0%" stopColor="#f5fdff" stopOpacity="1"/><stop offset="22%" stopColor="#72dcff" stopOpacity=".8"/><stop offset="100%" stopColor="#72dcff" stopOpacity="0"/></radialGradient>
                <linearGradient id="beamLeft"><stop stopColor="#67d9ff" stopOpacity="0"/><stop offset="100%" stopColor="#67d9ff"/></linearGradient>
                <filter id="softGlow"><feGaussianBlur stdDeviation="5"/></filter>
              </defs>
              <g fill="none" stroke="#24435b">
                <ellipse cx="360" cy="143" rx="306" ry="112" strokeWidth="2"/>
                <ellipse cx="360" cy="143" rx="250" ry="86" strokeWidth="1.5"/>
                <ellipse cx="360" cy="143" rx="190" ry="61" strokeWidth="1"/>
                <ellipse cx="360" cy="143" rx="125" ry="38" strokeWidth="1" opacity=".8"/>
              </g>
              <path d="M32 143 H688" stroke="#17394f" strokeWidth="1" strokeDasharray="4 8"/>
              <path d="M45 143 H350" stroke="url(#beamLeft)" strokeWidth="4" strokeDasharray="12 9">
                <animate attributeName="stroke-dashoffset" from="180" to="0" dur="1.4s" repeatCount="indefinite"/>
              </path>
              <path d="M675 143 H370" stroke="#e9b66f" strokeWidth="3" strokeDasharray="10 10">
                <animate attributeName="stroke-dashoffset" from="0" to="180" dur="1.4s" repeatCount="indefinite"/>
              </path>
              {showTracks && <g fill="none" strokeLinecap="round">
                <path d="M360 143 Q445 45 600 42" stroke="#67d9ff" strokeWidth="3" strokeDasharray="240" strokeDashoffset="240"><animate attributeName="stroke-dashoffset" from="240" to="0" dur=".8s" begin=".2s" fill="freeze"/></path>
                <path d="M360 143 Q480 115 670 105" stroke="#67d9ff" strokeWidth="2.5" strokeDasharray="340" strokeDashoffset="340"><animate attributeName="stroke-dashoffset" from="340" to="0" dur=".9s" begin=".2s" fill="freeze"/></path>
                <path d="M360 143 Q470 220 605 258" stroke="#67d9ff" strokeWidth="3" strokeDasharray="260" strokeDashoffset="260"><animate attributeName="stroke-dashoffset" from="260" to="0" dur=".9s" begin=".2s" fill="freeze"/></path>
                <path d="M360 143 Q265 45 115 55" stroke="#67d9ff" strokeWidth="2.5" strokeDasharray="250" strokeDashoffset="250"><animate attributeName="stroke-dashoffset" from="250" to="0" dur=".8s" begin=".2s" fill="freeze"/></path>
                <path d="M360 143 Q235 165 54 188" stroke="#e9b66f" strokeWidth="2.5" strokeDasharray="320" strokeDashoffset="320"><animate attributeName="stroke-dashoffset" from="320" to="0" dur=".9s" begin=".2s" fill="freeze"/></path>
                <path d="M360 143 Q280 245 155 270" stroke="#e9b66f" strokeWidth="2" strokeDasharray="230" strokeDashoffset="230"><animate attributeName="stroke-dashoffset" from="230" to="0" dur=".9s" begin=".2s" fill="freeze"/></path>
              </g>}
              <circle cx="360" cy="143" r="76" fill="url(#eventGlow)" opacity=".8"><animate attributeName="r" values="22;78;22" dur="2.4s" repeatCount="indefinite"/></circle>
              <circle cx="360" cy="143" r="9" fill="#f5fdff"><animate attributeName="r" values="5;11;5" dur="1.4s" repeatCount="indefinite"/></circle>
              <circle cx="360" cy="143" r="22" fill="none" stroke="#c7f2ff" strokeWidth="1.5"><animate attributeName="r" values="15;54;15" dur="2.4s" repeatCount="indefinite"/></circle>
              <g fill="#a9c5d8" fontSize="11" letterSpacing="1.5"><text x="36" y="22">BEAM A</text><text x="620" y="22">BEAM B</text><text x="360" y="292" textAnchor="middle">SIMPLIFIED EVENT DISPLAY · NOT TO SCALE</text></g>
              <g fill="#67d9ff"><circle cx="600" cy="42" r="4"/><circle cx="670" cy="105" r="4"/><circle cx="605" cy="258" r="4"/><circle cx="115" cy="55" r="4"/></g>
              <g fill="#e9b66f"><circle cx="54" cy="188" r="4"/><circle cx="155" cy="270" r="4"/></g>
            </svg>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line/70 px-4 py-3">
            <div className="flex flex-wrap gap-3 text-[10px] text-mist"><span className="inline-flex items-center gap-1.5"><i className="size-2 rounded-full bg-[#67d9ff]"/>Outgoing tracks</span><span className="inline-flex items-center gap-1.5"><i className="size-2 rounded-full bg-[#e9b66f]"/>Other products</span></div>
            <button onClick={() => setShowTracks((v) => !v)} className="inline-flex items-center gap-1.5 text-[10px] text-ion">{showTracks ? "Hide" : "Show"} particle tracks</button>
          </div>
        </div>

        <aside className="flex flex-col justify-between rounded-2xl border border-line bg-void/75 p-4">
          <div>
            <p className="kicker text-[10px] tracking-[0.18em] text-ion">WHAT ARE WE LOOKING FOR?</p>
            <div className="mt-4 flex items-center gap-3">
              <div className="grid size-12 place-items-center rounded-full border border-ion/30 bg-ion/10"><Atom className="size-6 text-ion"/></div>
              <div><p className="text-sm font-semibold">Higgs boson</p><p className="text-[10px] text-mist">A rare particle</p></div>
            </div>
            <p className="mt-3 text-xs leading-6 text-mist">It cannot be photographed directly here. Scientists look for the pattern of particles it can decay into.</p>
          </div>
          <div className="mt-5 rounded-xl bg-white/[0.04] p-3">
            <p className="text-[10px] text-mist">THE SIMPLE IDEA</p>
            <p className="mt-1 text-sm font-medium">Collision → clues → model → estimate</p>
          </div>
        </aside>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-2xl border border-line bg-void/75 p-4">
          <div className="flex items-center justify-between gap-2"><div><p className="kicker text-[10px] tracking-[0.18em] text-ion">02 · THE DATASET</p><h2 className="display mt-1 text-xl">28 clues per event</h2></div><span className="num text-3xl text-ion">11M</span></div>
          <p className="mt-1 text-[10px] text-mist">Simulated collision events · two labels: Signal / Background</p>
          <div className="mt-4 flex h-7 gap-1">
            {Array.from({length:21},(_,i)=><div key={i} title={`Low-level feature ${i+1}`} className="flex-1 rounded-sm bg-[#67d9ff]/85"/>)}
            {Array.from({length:7},(_,i)=><div key={i} title={`High-level feature ${i+1}`} className="flex-1 rounded-sm bg-[#e9b66f]"/>)}
          </div>
          <div className="mt-2 flex justify-between text-[10px]"><span className="text-ion">21 low-level</span><span className="text-[#e9b66f]">7 high-level</span></div>
          <p className="mt-3 text-[11px] leading-5 text-mist">Low-level = detector / kinematic values. High-level = physics-derived values.</p>
          <div className="mt-3 flex flex-wrap gap-1.5 text-[10px]"><span className="rounded-full bg-ion/10 px-2.5 py-1 text-ion">21 features</span><span className="rounded-full bg-[#e9b66f]/10 px-2.5 py-1 text-[#e9b66f]">7 features</span><span className="rounded-full bg-white/10 px-2.5 py-1 text-cream">28 together</span></div>
        </div>

        <div className="rounded-2xl border border-line bg-void/75 p-4">
          <p className="kicker text-[10px] tracking-[0.18em] text-ion">03 · THE PROJECT PIPELINE</p>
          <h2 className="display mt-1 text-xl">From 11 million rows to a prediction</h2>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <div className="relative rounded-xl bg-white/[0.04] p-3"><p className="text-[9px] text-ion">HIGGS</p><p className="num mt-1 text-2xl">11M</p><p className="text-[10px] text-mist">full dataset</p></div>
            <div className="relative rounded-xl bg-white/[0.04] p-3"><p className="text-[9px] text-ion">SUBSAMPLE</p><p className="num mt-1 text-2xl">1M</p><p className="text-[10px] text-mist">stratified · seed saved</p></div>
            <div className="relative rounded-xl bg-white/[0.04] p-3"><p className="text-[9px] text-ion">SPLIT</p><p className="num mt-1 text-xl">70/15/15</p><p className="text-[10px] text-mist">train / val / test</p></div>
            <div className="relative rounded-xl bg-white/[0.04] p-3"><p className="text-[9px] text-ion">FEATURES</p><p className="num mt-1 text-xl">21·7·28</p><p className="text-[10px] text-mist">three experiments</p></div>
          </div>
          <div className="mt-3 flex h-2 overflow-hidden rounded-full"><div className="w-[70%] bg-[#67d9ff]"/><div className="w-[15%] bg-[#e9b66f]"/><div className="w-[15%] bg-[#a99aff]"/></div>
          <div className="mt-1 flex justify-between text-[9px] text-mist"><span>700k train</span><span>150k validation</span><span>150k test</span></div>
          <div className="mt-3 flex items-center gap-2 text-[10px] text-mist"><BrainCircuit className="size-4 text-ion"/> Five model approaches learn from labelled examples <ChevronRight className="ml-auto size-3"/> <span className="text-ion">Signal / Background</span></div>
        </div>
      </div>

      <div className="mt-5 rounded-2xl border border-line bg-void/75 p-4">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div><p className="kicker text-[10px] tracking-[0.18em] text-ion">04 · FIVE MODELS, FIVE APPROACHES</p><h2 className="display mt-1 text-xl">Choose how the machine learns.</h2></div>
          <span className="text-[10px] text-mist">Tap a model</span>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {models.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} onClick={() => setModel(item.id)} className={model === item.id ? "inline-flex items-center gap-2 rounded-full border border-ion bg-ion/10 px-3 py-2 text-[11px] text-ion" : "inline-flex items-center gap-2 rounded-full border border-line px-3 py-2 text-[11px] text-mist hover:border-ion/50"}><Icon className="size-3.5"/>{item.name}</button>;
          })}
        </div>
        <div className="mt-3 flex items-start gap-3 rounded-xl bg-white/[0.04] p-3">
          <div className="grid size-9 shrink-0 place-items-center rounded-lg bg-ion/10 text-ion"><ModelIcon className="size-5"/></div>
          <div><p className="text-xs font-semibold">{active.label}</p><p className="mt-1 text-[11px] leading-5 text-mist">{active.detail}</p></div>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 rounded-2xl border border-line bg-void/75 px-4 py-3">
        <div><p className="kicker text-[9px] text-ion">BEST VALIDATION AUC</p><p className="num text-2xl text-ion">0.838</p></div>
        <div className="min-w-40 flex-1"><div className="h-2 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-[#477b91] to-[#70dcff]" style={{width:"83.8%"}}/></div><div className="mt-1 flex justify-between text-[9px] text-mist"><span>0 · low ranking ability</span><span>1 · perfect ranking</span></div></div>
        <p className="max-w-xs text-[10px] leading-4 text-mist">AUC describes ranking ability, not accuracy. A model prediction is not proof of a Higgs discovery.</p>
      </div>
      <section className="mt-5 rounded-2xl border border-line bg-void/75 p-4">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div><p className="kicker text-[10px] tracking-[0.18em] text-ion">05 · RECORDED MODEL RESULTS</p><h2 className="display mt-1 text-xl">How well did they separate the two classes?</h2></div>
          <span className="text-[10px] text-mist">28 features · 150,000 validation events each</span>
        </div>
        <p className="mt-2 max-w-3xl text-[11px] leading-5 text-mist">These are saved validation metrics from the project experiments. Accuracy is the share classified correctly; ROC-AUC measures how well a model ranks Signal above Background (1.0 is best). They are not live predictions.</p>
        <div className="mt-4 space-y-3">
          {[
            {name:"Neural Network (MLP) · baseline",accuracy:"75.53%",auc:0.837502},
            {name:"CatBoost",accuracy:"74.99%",auc:0.831869},
            {name:"XGBoost · extended",accuracy:"74.93%",auc:0.831241},
            {name:"LightGBM",accuracy:"74.55%",auc:0.827591},
            {name:"Logistic Regression",accuracy:"64.11%",auc:0.683508},
          ].map((item) => (
            <div key={item.name} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1.5">
              <p className="truncate text-[11px] text-cream">{item.name}</p>
              <div className="flex items-center gap-3 text-[10px] tabular-nums"><span className="text-mist">{item.accuracy} acc.</span><strong className="text-ion">{(item.auc*100).toFixed(2)} AUC</strong></div>
              <div className="col-span-2 grid grid-cols-[34px_minmax(0,1fr)_auto] items-center gap-2">
                <span className="text-[9px] text-mist">AUC</span>
                <div className="h-2 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-[#477b91] to-[#70dcff]" style={{width:`${(item.auc*100).toFixed(2)}%`}} /></div>
                <span className="num w-12 text-right text-[9px] text-ion">{item.auc.toFixed(3)}</span>
                <span className="text-[9px] text-mist">ACC</span>
                <div className="h-2 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-[#e9b66f]" style={{width:item.accuracy}} /></div>
                <span className="num w-12 text-right text-[9px] text-[#e9b66f]">{item.accuracy}</span>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 rounded-xl border border-line/70 bg-white/[0.03] p-3">
          <p className="text-[10px] font-semibold text-cream">Separate held-out test results (not mixed into the validation ranking)</p>
          <p className="mt-1 text-[10px] leading-5 text-mist">CatBoost: 74.94% accuracy · 83.21% ROC-AUC. LightGBM: 74.43% · 82.76%. Logistic Regression: 64.26% · 68.54%.</p>
        </div>
      </section>
      <p className="mt-4 text-center text-[10px] text-mist/70">Collision animation is a simplified illustration, not a detector recording. “Signal” means Higgs-like; “Background” means ordinary collision events.</p>
    </section>
  );
}

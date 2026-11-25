import { useState } from "react";
import { Atom, BrainCircuit, ChevronRight, CircleDot, GitBranch, Layers3, Network, ScanLine, Sparkles, Trees } from "lucide-react";

const steps = [
  { title: "A tiny collision", tag: "01 · THE IDEA", body: "Inside the Large Hadron Collider, tiny particles are accelerated to enormous speeds and made to collide. Most collisions produce ordinary particles. Very rarely, a collision may produce a Higgs boson.", note: "The Higgs boson is not seen directly here. Scientists look for the pattern of particles it can decay into." },
  { title: "The prediction", tag: "02 · THE RESULT", body: "The model returns a score: how signal-like does this event look? At the chosen threshold, the event is labelled Signal or Background.", note: "A model prediction is a statistical estimate, not proof of a Higgs discovery." },
];

const models = [
  { id: "lr", name: "Logistic Regression", icon: GitBranch, simple: "A weighted scorecard", detail: "Combines the 28 measurements using learned weights, then converts the score into a probability.", analogy: "Like adding weighted clues on a checklist." },
  { id: "xgb", name: "XGBoost", icon: Trees, simple: "Many small decision trees", detail: "Builds trees one after another. Each new tree focuses on patterns earlier trees handled less well.", analogy: "Like a team where each next reviewer learns from earlier mistakes." },
  { id: "cat", name: "CatBoost", icon: Layers3, simple: "A boosted tree team", detail: "Also combines decision trees in stages, using techniques designed to make boosting robust.", analogy: "Like several decision guides whose answers are combined." },
  { id: "lgb", name: "LightGBM", icon: ScanLine, simple: "Efficient decision trees", detail: "Uses histogram-based tree learning to find useful splits efficiently on large datasets.", analogy: "Like sorting clues into smart bins before making decisions." },
  { id: "mlp", name: "Neural Network (MLP)", icon: Network, simple: "Layers of connected units", detail: "Passes the measurements through layers of learned connections to capture more complex combinations.", analogy: "Like a chain of stages, each transforming the clues." },
];

export function Explainer() {
  const [step, setStep] = useState(0);
  const [model, setModel] = useState("xgb");
  const active = models.find((item) => item.id === model) ?? models[1];
  const ModelIcon = active.icon;

  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-5 pb-32 pt-8 text-cream md:px-10">
      <div className="mb-8 max-w-3xl">
        <p className="kicker mb-3 text-xs tracking-[0.22em] text-ion">START HERE · A 2-MINUTE GUIDE</p>
        <h1 className="display text-4xl leading-tight md:text-6xl">The Higgs, explained<br /><span className="text-ion">without the jargon.</span></h1>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-mist md:text-base">A particle collision creates clues. Our models study those clues and estimate whether an event looks like a rare Higgs signal or ordinary background.</p>
      </div>

      <div className="grid gap-5">
        <div className="rounded-3xl border border-line bg-void/75 p-5 backdrop-blur md:p-7">
          <div className="mb-4 flex items-center justify-between gap-3">
            <p className="kicker text-xs tracking-widest text-ion">FROM COLLISION TO CLASSIFICATION</p>
            <span className="text-xs text-mist">Interactive guide</span>
          </div>
          <div className="relative overflow-hidden rounded-2xl border border-line bg-[#08111b] p-4 md:p-6">
            <svg viewBox="0 0 640 260" role="img" aria-label="Illustration of two particle beams meeting at a collision point, with detector rings and outgoing particle tracks" className="h-auto w-full">
              <defs>
                <radialGradient id="collisionGlow"><stop offset="0%" stopColor="#e9fbff" stopOpacity="1"/><stop offset="35%" stopColor="#65d8ff" stopOpacity=".55"/><stop offset="100%" stopColor="#65d8ff" stopOpacity="0"/></radialGradient>
                <linearGradient id="beamA"><stop stopColor="#64d8ff" stopOpacity="0"/><stop offset="50%" stopColor="#64d8ff"/><stop offset="100%" stopColor="#64d8ff" stopOpacity="0"/></linearGradient>
              </defs>
              <ellipse cx="320" cy="130" rx="250" ry="94" fill="none" stroke="#31506a" strokeWidth="2" opacity=".75"/>
              <ellipse cx="320" cy="130" rx="205" ry="68" fill="none" stroke="#31506a" strokeWidth="1.5" opacity=".7"/>
              <ellipse cx="320" cy="130" rx="155" ry="43" fill="none" stroke="#31506a" strokeWidth="1" opacity=".65"/>
              <path d="M42 130 H598" stroke="url(#beamA)" strokeWidth="3" strokeDasharray="9 9"/>
              <path d="M320 130 L490 48 M320 130 L550 110 M320 130 L493 206 M320 130 L160 55 M320 130 L90 158 M320 130 L180 218" stroke="#75dfff" strokeWidth="2.5" strokeLinecap="round" opacity=".8"/>
              <path d="M320 130 L420 30 M320 130 L250 20 M320 130 L410 230 M320 130 L230 235" stroke="#f4b86b" strokeWidth="1.5" strokeLinecap="round" opacity=".75"/>
              <circle cx="320" cy="130" r="58" fill="url(#collisionGlow)"/>
              <circle cx="320" cy="130" r="7" fill="#f4fbff"/>
              <circle cx="320" cy="130" r="18" fill="none" stroke="#b6efff" strokeWidth="1.5" opacity=".9"/>
              {[ [490,48],[550,110],[493,206],[160,55],[90,158],[180,218],[420,30],[250,20],[410,230],[230,235] ].map(([cx,cy],i)=><circle key={i} cx={cx} cy={cy} r="4" fill={i<6?"#75dfff":"#f4b86b"}/>)}
              <text x="320" y="250" textAnchor="middle" fill="#8fa9bd" fontSize="11" letterSpacing="2">SCHEMATIC COLLISION · NOT TO SCALE</text>
            </svg>
            <div className="mt-3 flex flex-wrap justify-center gap-4 text-[11px] text-mist">
              <span className="inline-flex items-center gap-2"><i className="h-2 w-2 rounded-full bg-[#75dfff]"/>Particle tracks</span>
              <span className="inline-flex items-center gap-2"><i className="h-2 w-2 rounded-full bg-[#f4b86b]"/>Other measured products</span>
            </div>
          </div>
          <div className="mt-5 flex flex-wrap gap-2">
            {steps.map((item, index) => (
              <button key={item.tag} onClick={() => setStep(index)} className={step === index ? "rounded-full border border-ion bg-ion/15 px-3 py-2 text-xs text-ion" : "rounded-full border border-line px-3 py-2 text-xs text-mist hover:border-ion/60"}>{item.title}</button>
            ))}
          </div>
          <div className="mt-5 min-h-44">
            <p className="kicker text-[10px] tracking-[0.2em] text-ion">{steps[step].tag}</p>
            <h2 className="display mt-2 text-2xl">{steps[step].title}</h2>
            <p className="mt-3 text-sm leading-7 text-mist">{steps[step].body}</p>
            <p className="mt-3 rounded-xl border border-line bg-white/[0.03] p-3 text-xs leading-6 text-cream/80">{steps[step].note}</p>
          </div>
          <div className="mt-4 flex justify-between">
            <button disabled={step === 0} onClick={() => setStep((value) => Math.max(0, value - 1))} className="rounded-full border border-line px-4 py-2 text-xs text-mist disabled:opacity-30">Previous</button>
            <button onClick={() => setStep((value) => (value + 1) % steps.length)} className="inline-flex items-center gap-2 rounded-full bg-ion px-4 py-2 text-xs font-semibold text-[#06111a]">Next step <ChevronRight className="size-3"/></button>
          </div>
        </div>

        <div className="space-y-5">
          <div className="rounded-3xl border border-line bg-void/75 p-5 backdrop-blur md:p-7">
            <p className="kicker text-xs tracking-widest text-ion">THE DATASET</p>
            <h2 className="display mt-2 text-3xl">What each model gets to see.</h2>
            <div className="mt-5 grid grid-cols-2 gap-3 md:grid-cols-4">
              <div className="rounded-2xl border border-line bg-white/[0.03] p-4"><p className="num text-3xl text-ion">11M</p><p className="mt-1 text-xs text-mist">simulated collision events</p></div>
              <div className="rounded-2xl border border-line bg-white/[0.03] p-4"><p className="num text-3xl text-ion">28</p><p className="mt-1 text-xs text-mist">features per event</p></div>
              <div className="rounded-2xl border border-line bg-white/[0.03] p-4"><p className="num text-3xl text-ion">21 + 7</p><p className="mt-1 text-xs text-mist">low-level + high-level</p></div>
              <div className="rounded-2xl border border-line bg-white/[0.03] p-4"><p className="num text-3xl text-ion">2</p><p className="mt-1 text-xs text-mist">classes: Signal / Background</p></div>
            </div>
            <div className="mt-5 grid gap-3 md:grid-cols-2">
              <div className="rounded-2xl border border-ion/30 bg-ion/[0.06] p-4">
                <div className="flex items-center justify-between"><p className="text-sm font-semibold text-ion">21 LOW-LEVEL</p><span className="text-[10px] text-mist">21 / 28</span></div>
                <div className="my-3 flex gap-1">{Array.from({length:21},(_,i)=><i key={i} className="h-3 flex-1 rounded-sm bg-[#63c9ee]"/>)}</div>
                <p className="text-xs leading-6 text-mist">Raw / detector-level kinematic quantities.</p>
              </div>
              <div className="rounded-2xl border border-[#e9b66f]/30 bg-[#e9b66f]/[0.05] p-4">
                <div className="flex items-center justify-between"><p className="text-sm font-semibold text-[#e9b66f]">7 HIGH-LEVEL</p><span className="text-[10px] text-mist">7 / 28</span></div>
                <div className="my-3 flex gap-1">{Array.from({length:7},(_,i)=><i key={i} className="h-3 flex-1 rounded-sm bg-[#e9b66f]"/>)}</div>
                <p className="text-xs leading-6 text-mist">Physics-derived quantities built from domain knowledge.</p>
              </div>
            </div>
            <div className="mt-4 rounded-xl border border-line bg-white/[0.03] p-3 text-xs text-mist">Feature sets explored: <strong className="text-cream">21 low-level</strong> · <strong className="text-cream">7 high-level</strong> · <strong className="text-cream">28 combined</strong>.</div>
          </div>

          <div className="rounded-3xl border border-line bg-void/75 p-5 backdrop-blur md:p-7">
            <p className="kicker text-xs tracking-widest text-ion">PROJECT ROADMAP</p>
            <h2 className="display mt-2 text-3xl">From dataset to model comparison.</h2>
            <div className="mt-6 grid gap-3 md:grid-cols-4">
              <div className="rounded-2xl border border-line bg-white/[0.03] p-4"><p className="kicker text-[10px] text-ion">01 · HIGGS</p><p className="num mt-2 text-3xl">11M</p><p className="mt-1 text-xs text-mist">events in the full dataset</p></div>
              <div className="rounded-2xl border border-line bg-white/[0.03] p-4"><p className="kicker text-[10px] text-ion">02 · SUBSAMPLE</p><p className="num mt-2 text-3xl">1M</p><p className="mt-1 text-xs text-mist">stratified sample · recorded seed</p></div>
              <div className="rounded-2xl border border-line bg-white/[0.03] p-4"><p className="kicker text-[10px] text-ion">03 · SPLIT</p><p className="num mt-2 text-2xl">70 / 15 / 15</p><p className="mt-1 text-xs text-mist">train / validation / test before preprocessing</p></div>
              <div className="rounded-2xl border border-line bg-white/[0.03] p-4"><p className="kicker text-[10px] text-ion">04 · FEATURE SETS</p><p className="num mt-2 text-2xl">21 · 7 · 28</p><p className="mt-1 text-xs text-mist">low · high · combined</p></div>
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2 text-[10px] text-mist"><span className="rounded-full bg-ion/10 px-3 py-2 text-ion">21 low-level</span><span className="rounded-full bg-[#e9b66f]/10 px-3 py-2 text-[#e9b66f]">7 high-level</span><span className="rounded-full bg-white/10 px-3 py-2">28 combined</span><span className="ml-auto">Same task · five model approaches</span></div>
          </div>

          <div className="rounded-3xl border border-line bg-void/75 p-5 backdrop-blur md:p-7">
            <p className="kicker text-xs tracking-widest text-ion">FIVE WAYS TO LEARN</p>
            <h2 className="display mt-2 text-2xl">Same task. Different approaches.</h2>
            <p className="mt-2 text-xs leading-6 text-mist">Choose a model to see its basic idea. All five are included in the project; availability of a live prediction depends on the local model file.</p>
            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
              {models.map((item) => {
                const Icon = item.icon;
                return <button key={item.id} onClick={() => setModel(item.id)} className={model === item.id ? "rounded-xl border border-ion bg-ion/10 p-3 text-left" : "rounded-xl border border-line bg-white/[0.02] p-3 text-left hover:border-ion/50"}>
                  <Icon className={model === item.id ? "size-5 text-ion" : "size-5 text-mist"}/>
                  <p className="mt-2 text-xs font-semibold">{item.name}</p>
                </button>;
              })}
            </div>
            <div className="mt-4 rounded-2xl border border-ion/25 bg-ion/[0.06] p-4">
              <div className="flex items-center gap-2 text-ion"><ModelIcon className="size-5"/><p className="text-sm font-semibold">{active.simple}</p></div>
              <p className="mt-2 text-sm leading-6 text-cream/90">{active.detail}</p>
              <p className="mt-3 text-xs text-mist">Think of it this way: {active.analogy}</p>
            </div>
          </div>

          <div className="rounded-3xl border border-line bg-void/75 p-5 backdrop-blur md:p-7">
            <p className="kicker text-xs tracking-widest text-ion">ONE RESULT FROM THE BENCHMARK</p>
            <div className="mt-2 flex items-end justify-between gap-4"><h2 className="display text-2xl">Best validation AUC</h2><span className="num text-4xl text-ion">0.838</span></div>
            <p className="mt-2 text-xs leading-6 text-mist">AUC measures how well a model ranks signal events above background across thresholds. 0.5 is roughly random ranking; 1.0 is perfect ranking. This is the best validation AUC reported for this project, not accuracy.</p>
            <div className="mt-5">
              <div className="mb-2 flex justify-between text-[10px] text-mist"><span>Random ranking · 0.50</span><span>Perfect · 1.00</span></div>
              <div className="relative h-4 overflow-hidden rounded-full bg-white/10"><div className="absolute inset-y-0 left-0 w-full bg-white/5"/><div className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-[#477b91] to-[#70dcff]" style={{ width: "83.8%" }}/><div className="absolute inset-y-0 left-1/2 w-px bg-cream/70"/></div>
              <div className="mt-2 flex justify-between text-[10px] text-mist"><span>0</span><span>0.5</span><span>1.0</span></div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-5 rounded-3xl border border-line bg-void/75 p-5 backdrop-blur md:p-7">
        <p className="kicker text-xs tracking-widest text-ion">THE 1-MILLION-EVENT BENCHMARK</p>
        <h2 className="display mt-2 text-2xl">Three separate groups.</h2>
        <p className="mt-2 text-sm leading-6 text-mist">Models learn from the training group. Validation helps compare settings. The test group is kept separate for final evaluation.</p>
        <div className="mt-5 flex h-5 overflow-hidden rounded-full">
          <div className="w-[70%] bg-[#63c9ee]"/>
          <div className="w-[15%] bg-[#e9b66f]"/>
          <div className="w-[15%] bg-[#9d91e8]"/>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
          <p className="text-mist">Training<strong className="mt-1 block text-cream">700,000 · 70%</strong></p>
          <p className="text-mist">Validation<strong className="mt-1 block text-cream">150,000 · 15%</strong></p>
          <p className="text-mist">Test<strong className="mt-1 block text-cream">150,000 · 15%</strong></p>
        </div>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <div className="rounded-2xl border border-line bg-void/70 p-4"><p className="num text-3xl text-ion">11M</p><p className="mt-1 text-xs text-mist">Events in the full HIGGS dataset</p></div>
        <div className="rounded-2xl border border-line bg-void/70 p-4"><p className="num text-3xl text-ion">28</p><p className="mt-1 text-xs text-mist">Numerical measurements per event</p></div>
        <div className="rounded-2xl border border-line bg-void/70 p-4"><p className="num text-3xl text-ion">5</p><p className="mt-1 text-xs text-mist">Different model approaches compared</p></div>
      </div>

      <p className="mt-6 max-w-4xl text-[11px] leading-6 text-mist/70">Important: “Signal” means an event classified as Higgs-like by the model; “Background” means an event classified as ordinary background. A prediction is a statistical estimate, not a confirmed particle discovery. The collision drawing is a simplified illustration.</p>
    </section>
  );
}

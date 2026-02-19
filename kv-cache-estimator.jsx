import { useState, useEffect, useCallback, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, AreaChart, Area, LineChart, Line, ReferenceLine
} from "recharts";
import {
  Server, Database, Zap, TrendingUp, BookOpen, X, Download,
  Upload, AlertTriangle, CheckCircle, ChevronRight, Activity,
  DollarSign, Cpu, HardDrive, Clock, Users, Shield, FileText,
  ChevronDown, Info
} from "lucide-react";

// ── CONSTANTS ────────────────────────────────────────────────────────────────

const GPU_PRESETS = {
  H100_SXM5: { name: "H100 SXM5 (8× node)", vram: 640, hourly: 32, capex: 400000, monthly: 85000, dtype: "H100 SXM5 8×GPU" },
  H200_SXM5: { name: "H200 SXM5 (8× node)", vram: 1128, hourly: 48, capex: 600000, monthly: 130000, dtype: "H200 SXM5 8×GPU" },
  A100_SXM4: { name: "A100 SXM4 (8× node)", vram: 640, hourly: 20, capex: 280000, monthly: 55000, dtype: "A100 SXM4 8×GPU" },
  L40S_8x:   { name: "L40S (8× node)",       vram: 384, hourly: 12, capex: 160000, monthly: 32000, dtype: "L40S 8×GPU" },
  H100_SXM5_single: { name: "H100 SXM5 (single GPU)", vram: 80, hourly: 4,  capex: 50000, monthly: 10625, dtype: "H100 SXM5" },
  A100_SXM4_single: { name: "A100 SXM4 (single GPU)", vram: 80, hourly: 2.5,capex: 35000, monthly: 6875,  dtype: "A100 SXM4" },
  L40S_single:      { name: "L40S (single GPU)",       vram: 48, hourly: 1.5,capex: 20000, monthly: 4000,  dtype: "L40S" },
};

const LLM_PRESETS = {
  llama3_8b:   { name: "Llama 3 8B",    params: 8,   layers: 32,  heads: 32,  kv_heads: 8,  head_dim: 128 },
  llama3_70b:  { name: "Llama 3 70B",   params: 70,  layers: 80,  heads: 64,  kv_heads: 8,  head_dim: 128 },
  llama3_405b: { name: "Llama 3 405B",  params: 405, layers: 126, heads: 128, kv_heads: 8,  head_dim: 128 },
  mixtral_8x7b:{ name: "Mixtral 8×7B",  params: 47,  layers: 32,  heads: 32,  kv_heads: 8,  head_dim: 128 },
  mixtral_8x22b:{ name: "Mixtral 8×22B",params: 141, layers: 56,  heads: 48,  kv_heads: 8,  head_dim: 128 },
};

const FRAMEWORK_PRESETS = {
  vllm:       { name: "vLLM",          overhead: 1.15, desc: "PagedAttention, open source" },
  tensorrt:   { name: "TensorRT-LLM",  overhead: 1.25, desc: "NVIDIA optimized, lower latency" },
};

const SCENARIOS = {
  longContext: {
    name: "Long-Context Document Processing",
    icon: "📄",
    desc: "Extended-context inference over large documents — legal, medical, financial filings",
    inputTokens: 32000, outputTokens: 2000, concurrentSessions: 50,
    gpu: "H100_SXM5", llm: "llama3_70b", framework: "vllm",
  },
  highConcurrency: {
    name: "High-Concurrency Enterprise AI",
    icon: "🏢",
    desc: "Customer-facing production deployments — chatbots, copilots, service automation",
    inputTokens: 4000, outputTokens: 1000, concurrentSessions: 500,
    gpu: "H100_SXM5", llm: "llama3_70b", framework: "tensorrt",
  },
  capitalMarkets: {
    name: "Capital Markets & Quant Research",
    icon: "📈",
    desc: "Real-time news/earnings summarization, RAG over filings, analyst copilots — latency SLO critical",
    inputTokens: 8000, outputTokens: 1500, concurrentSessions: 200,
    gpu: "H100_SXM5", llm: "llama3_70b", framework: "tensorrt",
  },
  agentic: {
    name: "Agentic & Multi-Step Reasoning",
    icon: "🤖",
    desc: "Multi-turn reasoning chains, RAG pipelines — high KV Cache reuse across turns",
    inputTokens: 16000, outputTokens: 4000, concurrentSessions: 100,
    gpu: "H200_SXM5", llm: "llama3_405b", framework: "vllm",
  },
};

const DDN_BANDWIDTH = 200;   // GB/s
const NAS_BANDWIDTH = 10;    // GB/s
const BYTES_PER_PARAM = 2;   // FP16
const BILLION = 1e9;

// ── PHYSICS ENGINE ───────────────────────────────────────────────────────────

function computePhysics(cfg) {
  const gpu    = GPU_PRESETS[cfg.gpu];
  const llm    = LLM_PRESETS[cfg.llm];
  const fw     = FRAMEWORK_PRESETS[cfg.framework];
  if (!gpu || !llm || !fw) return null;

  // VRAM components (GB)
  const modelWeightsGB = (llm.params * BILLION * BYTES_PER_PARAM) / 1e9;
  const engineOverheadGB = modelWeightsGB * (fw.overhead - 1);

  // KV Cache per session (GB)
  // 2 (K+V) * layers * kv_heads * head_dim * (inputTokens + outputTokens) * 2 bytes
  const totalTokens = cfg.inputTokens + cfg.outputTokens;
  const kvPerSessionBytes = 2 * llm.layers * llm.kv_heads * llm.head_dim * totalTokens * BYTES_PER_PARAM;
  const kvPerSessionGB = kvPerSessionBytes / 1e9;

  const kvTotalGB = kvPerSessionGB * cfg.concurrentSessions;
  const totalVRAMNeeded = modelWeightsGB + engineOverheadGB + kvTotalGB;

  const physicalVRAM = gpu.vram;
  const availableForKV = physicalVRAM - modelWeightsGB - engineOverheadGB;
  const kvOverflow = Math.max(0, kvTotalGB - availableForKV);
  const isOffloading = kvOverflow > 0;
  const utilizationPct = Math.min(100, (totalVRAMNeeded / physicalVRAM) * 100);

  // Sessions that fit in VRAM without offloading
  const sessionsInVRAM = Math.max(0, Math.floor(availableForKV / kvPerSessionGB));

  // Restore time (seconds) - time to reload one session's KV from storage
  const restoreTimeDDN = kvPerSessionGB / DDN_BANDWIDTH;
  const restoreTimeNAS = kvPerSessionGB / NAS_BANDWIDTH;

  // Sustainable swaps per hour
  const avgSessionTimeSec = 30; // assume 30s avg session
  const swapsPerHourDDN = kvOverflow > 0 ? Math.floor(3600 / (avgSessionTimeSec + restoreTimeDDN)) * (cfg.concurrentSessions - sessionsInVRAM) : cfg.concurrentSessions * 120;
  const swapsPerHourNAS = kvOverflow > 0 ? Math.floor(3600 / (avgSessionTimeSec + restoreTimeNAS)) * (cfg.concurrentSessions - sessionsInVRAM) : cfg.concurrentSessions * 10;

  // GPU nodes needed
  const nodesWithoutDDN = Math.ceil(totalVRAMNeeded / physicalVRAM);
  const nodesWithDDN = Math.ceil((modelWeightsGB + engineOverheadGB + kvPerSessionGB * Math.min(sessionsInVRAM, cfg.concurrentSessions * 0.3)) / physicalVRAM);
  const nodesAvoided = Math.max(0, nodesWithoutDDN - nodesWithDDN);

  // Economics
  const monthlyRentalPerNode = cfg.monthlyRate;
  const monthlyRentalSavings = nodesAvoided * monthlyRentalPerNode;
  const contractSavings = monthlyRentalSavings * cfg.contractMonths;
  const capexAvoidance = nodesAvoided * gpu.capex;
  const annualOpexSavings = nodesAvoided * gpu.hourly * 24 * 365;

  // System overload check
  const isSystemOverload = isOffloading && restoreTimeNAS > 5;

  // Throughput ratio
  const throughputGainPct = kvOverflow > 0 ? Math.round((DDN_BANDWIDTH / NAS_BANDWIDTH - 1) * 100) : 0;

  return {
    modelWeightsGB, engineOverheadGB, kvTotalGB, kvPerSessionGB,
    totalVRAMNeeded, physicalVRAM, availableForKV, kvOverflow,
    isOffloading, utilizationPct, sessionsInVRAM,
    restoreTimeDDN, restoreTimeNAS,
    swapsPerHourDDN, swapsPerHourNAS,
    nodesWithoutDDN, nodesWithDDN, nodesAvoided,
    monthlyRentalSavings, contractSavings, capexAvoidance, annualOpexSavings,
    throughputGainPct, isSystemOverload,
    gpu, llm, fw,
  };
}

// ── HELPERS ──────────────────────────────────────────────────────────────────

const fmt = {
  gb:  (n) => n < 1 ? `${(n * 1024).toFixed(0)} MB` : `${n.toFixed(1)} GB`,
  pct: (n) => `${n.toFixed(1)}%`,
  ms:  (n) => n < 1 ? `${(n * 1000).toFixed(0)}ms` : `${n.toFixed(2)}s`,
  usd: (n) => n >= 1e6 ? `$${(n/1e6).toFixed(2)}M` : `$${Math.round(n).toLocaleString()}`,
  num: (n) => Math.round(n).toLocaleString(),
};

// ── TOOLTIP ──────────────────────────────────────────────────────────────────

function Tip({ label, children }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-block">
      <button
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        className="text-cyan-400 hover:text-cyan-300 ml-1 align-middle"
      >
        <Info size={12} />
      </button>
      {open && (
        <span className="absolute z-50 bottom-6 left-0 w-64 bg-slate-800 border border-slate-600 rounded-lg p-3 text-xs text-slate-300 shadow-2xl">
          <span className="font-semibold text-cyan-400 block mb-1">{label}</span>
          {children}
        </span>
      )}
    </span>
  );
}

// ── METRIC CARD ──────────────────────────────────────────────────────────────

function MetricCard({ label, value, sub, icon: Icon, color = "cyan", warn }) {
  const colors = {
    cyan:   "border-cyan-500/30 bg-cyan-500/5",
    amber:  "border-amber-500/30 bg-amber-500/5",
    red:    "border-red-500/30 bg-red-500/5",
    green:  "border-green-500/30 bg-green-500/5",
    purple: "border-purple-500/30 bg-purple-500/5",
  };
  const textColors = { cyan:"text-cyan-400", amber:"text-amber-400", red:"text-red-400", green:"text-green-400", purple:"text-purple-400" };
  return (
    <div className={`border rounded-xl p-4 ${colors[color]} ${warn ? "border-red-500/60 animate-pulse" : ""}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs text-slate-400 font-medium uppercase tracking-wider mb-1">{label}</div>
          <div className={`text-2xl font-bold font-mono ${textColors[color]}`}>{value}</div>
          {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
        </div>
        {Icon && <Icon size={20} className={`${textColors[color]} opacity-60`} />}
      </div>
    </div>
  );
}

// ── CUSTOM CHART TOOLTIP ─────────────────────────────────────────────────────

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 text-xs shadow-2xl">
      <div className="text-slate-300 font-semibold mb-2">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-400">{p.name}:</span>
          <span className="text-white font-mono">{typeof p.value === "number" ? p.value.toFixed(1) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

// ── SELLER'S GUIDE MODAL ─────────────────────────────────────────────────────

function SellersGuide({ onClose }) {
  const [tab, setTab] = useState("inference");
  const tabs = [
    { id: "inference",   label: "AI Inference 101" },
    { id: "kvcache",     label: "The KV Cache Bottleneck" },
    { id: "capitalmarkets", label: "Capital Markets Use Case" },
    { id: "salesplays",  label: "Sales Plays" },
    { id: "objections",  label: "Objection Handling" },
    { id: "discovery",   label: "Discovery Questions" },
    { id: "compliance",  label: "Compliance Angle" },
  ];

  const content = {
    inference: (
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-cyan-400">Understanding AI Inference</h3>
        <p className="text-slate-300 text-sm leading-relaxed">When a customer deploys a large language model (LLM) in production, they are running <strong className="text-white">inference</strong> — the process of feeding input text (a "prompt") to the model and generating output tokens one at a time. Unlike training, which happens once, inference runs continuously and at scale, serving thousands of simultaneous users.</p>
        <div className="bg-slate-700/50 rounded-lg p-4">
          <h4 className="text-sm font-bold text-white mb-2">The Three Components of GPU Memory Consumption</h4>
          <div className="space-y-2 text-sm">
            <div className="flex gap-3"><span className="text-cyan-400 font-mono w-32 shrink-0">Model Weights</span><span className="text-slate-300">The static parameters of the neural network. A 70B parameter model in FP16 precision requires ~140GB of VRAM. This never changes at runtime.</span></div>
            <div className="flex gap-3"><span className="text-cyan-400 font-mono w-32 shrink-0">Engine Overhead</span><span className="text-slate-300">Memory consumed by the inference framework (vLLM, TensorRT-LLM) for runtime state, CUDA kernels, and paged memory management. Typically 15–25% above model weights.</span></div>
            <div className="flex gap-3"><span className="text-amber-400 font-mono w-32 shrink-0">KV Cache</span><span className="text-slate-300">Dynamic memory that grows with context length and concurrent session count. This is the bottleneck. It scales as: 2 × layers × KV heads × head dimension × total tokens × 2 bytes.</span></div>
          </div>
        </div>
        <p className="text-slate-300 text-sm leading-relaxed">The practical problem: model weights and engine overhead are fixed. KV Cache is the variable that blows up GPU memory budgets as context windows grow and concurrent user counts scale.</p>
      </div>
    ),
    kvcache: (
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-cyan-400">The Hidden Bottleneck: KV Cache</h3>
        <p className="text-slate-300 text-sm leading-relaxed">During autoregressive generation, the model computes <strong className="text-white">key and value attention tensors</strong> for every token in the context. Rather than recomputing these on each forward pass, the framework caches them in GPU HBM — this is the KV Cache. Without it, inference would be 10–100× slower. With it, memory fills up fast.</p>
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
          <h4 className="text-sm font-bold text-amber-400 mb-2">⚠ The Failure Mode in Production</h4>
          <p className="text-slate-300 text-sm">When KV Cache exceeds physical VRAM, the system doesn't crash cleanly. Instead: the scheduler caps concurrency to preserve per-session latency → GPU utilization drops → throughput falls → operators over-provision GPU servers to compensate → cost per inference token spikes. The failure is invisible to application teams. It looks like "we need more GPUs."</p>
        </div>
        <div className="bg-slate-700/50 rounded-lg p-4">
          <h4 className="text-sm font-bold text-white mb-2">Virtual VRAM: What DDN AI Fabric Does</h4>
          <p className="text-slate-300 text-sm leading-relaxed">DDN AI Fabric provides <strong className="text-cyan-400">200+ GB/s sustained read bandwidth</strong> — fast enough that KV Cache state can be offloaded to storage and recalled between generation steps without introducing meaningful latency. This creates <strong className="text-white">Virtual VRAM</strong>: effective GPU memory that extends beyond physical HBM limits. The GPU stays compute-bound instead of becoming state-bound.</p>
          <p className="text-slate-300 text-sm mt-2">Standard NAS delivers ~10 GB/s. At that bandwidth, a single 32K-token session's KV state takes several seconds to restore — far exceeding any acceptable SLO. DDN makes the restore time sub-100ms at scale.</p>
        </div>
      </div>
    ),
    capitalmarkets: (
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-cyan-400">Capital Markets & Quantitative Research</h3>
        <p className="text-slate-300 text-sm leading-relaxed">Boost Run's founding team built and operated HFT infrastructure across 13 data centers globally. This gives Boost Run direct credibility in capital markets conversations. The AI infrastructure use cases in this space are distinct from pure execution — and they are significant GPU consumers.</p>
        <div className="bg-slate-700/50 rounded-lg p-4 space-y-3">
          <h4 className="text-sm font-bold text-white">Primary LLM Inference Use Cases in Capital Markets</h4>
          {[
            ["Real-Time News & Earnings Summarization", "Trading desks run LLM inference against live news feeds, SEC filings, and earnings transcripts. Latency SLO is tight — P99 matters because delayed signal equals missed alpha. High concurrency (many analysts, many instruments simultaneously). KV Cache eviction directly translates to latency spikes at exactly the wrong moment."],
            ["RAG Over Regulatory Filings", "Compliance and risk teams run retrieval-augmented generation over large document corpora — 10-Ks, prospectuses, ISDA agreements, regulatory guidance. Long context windows (16K–128K tokens). Context length is the primary KV Cache driver."],
            ["Multi-Turn Analyst Copilots", "Portfolio managers and quant researchers use persistent multi-turn sessions. KV Cache reuse across turns is the core efficiency mechanism. Without fast storage offload, each turn requires prefill recomputation — destroying the economic case for the copilot."],
            ["Risk Narrative Generation", "Model risk and credit teams generate natural language explanations of quantitative outputs at scale. High throughput, moderate latency. Batch-friendly but volume-sensitive."],
          ].map(([title, desc]) => (
            <div key={title} className="border-l-2 border-cyan-500/40 pl-3">
              <div className="text-sm font-semibold text-white">{title}</div>
              <div className="text-xs text-slate-400 mt-1">{desc}</div>
            </div>
          ))}
        </div>
        <div className="bg-slate-700/50 rounded-lg p-4">
          <h4 className="text-sm font-bold text-amber-400 mb-2">Important Distinction: LLM Inference ≠ HFT Execution</h4>
          <p className="text-slate-300 text-sm">Traditional HFT operates at microsecond latency — below the minimum decode latency of any transformer model. The AI inference use cases above operate at millisecond-to-second latency, where KV Cache management is the dominant variable. Position this correctly: DDN + Boost Run targets the AI inference workloads sitting alongside HFT systems, not HFT execution itself.</p>
        </div>
      </div>
    ),
    salesplays: (
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-cyan-400">Sales Plays</h3>
        {[
          {
            title: "The Over-Provisioning Play",
            trigger: "Customer says they need more GPU servers to handle load",
            play: "Run the physics engine with their current config. Show that KV Cache overflow — not compute shortage — is causing concurrency caps. Quantify how many server nodes they can eliminate with DDN. Present the monthly rental savings vs. DDN storage cost. ROI is typically positive within 2–4 months.",
            target: "VP of Infrastructure, CTO"
          },
          {
            title: "The Latency SLO Play",
            trigger: "Customer reports P99 latency spikes under load",
            play: "Show the KV Cache restore time comparison: NAS at 10 GB/s introduces multi-second stalls under concurrent load. DDN at 200 GB/s restores context in sub-100ms. For capital markets and customer-facing AI, this is the difference between a usable product and an unusable one.",
            target: "Platform Engineering, ML Ops"
          },
          {
            title: "The RFB Bundle Play",
            trigger: "Customer is sizing a new cluster via Boost Run RFB",
            play: "Spec the compute layer first (GPU nodes, networking). Then present DDN AI Fabric as the storage tier that changes the node count math. Show them the cluster they would have ordered vs. the cluster they need with DDN. Bundle DDN into the RFB quote as a single solution.",
            target: "Infrastructure architects, procurement"
          },
          {
            title: "The Compliance Boundary Play",
            trigger: "Customer is in a regulated industry (healthcare, finance, federal)",
            play: "KV Cache contains live inference context — effectively a running record of sensitive user interactions. When it spills to generic NAS or local swap, it exits the certified storage boundary. DDN keeps KV state within the auditable, access-controlled storage tier. Boost Run's HIPAA/SOC2/ISO 27001 certifications + DDN's controlled storage = defensible compliance architecture.",
            target: "CISO, Chief Compliance Officer"
          },
        ].map(({ title, trigger, play, target }) => (
          <div key={title} className="bg-slate-700/40 rounded-lg p-4 space-y-2">
            <div className="text-sm font-bold text-white">{title}</div>
            <div className="text-xs"><span className="text-amber-400 font-semibold">Trigger: </span><span className="text-slate-300">{trigger}</span></div>
            <div className="text-xs"><span className="text-cyan-400 font-semibold">Play: </span><span className="text-slate-300">{play}</span></div>
            <div className="text-xs"><span className="text-purple-400 font-semibold">Target: </span><span className="text-slate-300">{target}</span></div>
          </div>
        ))}
      </div>
    ),
    objections: (
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-cyan-400">Objection Handling</h3>
        {[
          ["\"We'll just add more GPUs.\"", "That's exactly the pattern DDN eliminates. More GPUs don't fix the KV Cache problem — they defer it. As your context windows grow and concurrency scales, you'll hit the same wall on the next cluster. DDN changes the cost curve permanently, not just for this purchase."],
          ["\"Our NAS is fast enough.\"", "Standard NAS delivers 10–20 GB/s. A single 32K-token Llama 70B session generates ~17GB of KV state. At 10 GB/s, restoring that context takes 1.7 seconds — before you've generated a single output token. Multiply by concurrent sessions and you have a system that's spending more time doing I/O than inference. Run the numbers for your specific workload in this tool."],
          ["\"We use vLLM's built-in KV eviction.\"", "vLLM's eviction policy is a concurrency cap, not a solution. When the scheduler evicts KV Cache, it has to recompute the prefill for that session on the next request — a full forward pass over the entire context. For 32K+ token contexts, that's seconds of compute per session. DDN offload preserves the state and recalls it at storage bandwidth, which is faster than recomputation at scale."],
          ["\"DDN is expensive.\"", "The question is expensive relative to what. If DDN eliminates two H100 nodes from a cluster, that's $170K in avoided CAPEX or $170K/year in avoided rental cost. DDN AI Fabric storage at that scale costs a fraction of that. The storage pays for itself in avoided GPU spend. Run the Executive Report tab to see the specific numbers for your configuration."],
          ["\"We're in the cloud, we don't need this.\"", "Cloud GPU instances have the same VRAM constraints as bare metal — often worse, because multi-tenancy limits VRAM allocation further. The KV Cache math is identical. The difference is that cloud costs are fully variable, so the over-provisioning tax is paid every month with no offsetting asset. DDN + Boost Run bare metal gives you the storage solution and a predictable cost structure."],
        ].map(([obj, resp]) => (
          <div key={obj} className="bg-slate-700/40 rounded-lg p-4">
            <div className="text-sm font-semibold text-amber-400 mb-2">{obj}</div>
            <div className="text-sm text-slate-300">{resp}</div>
          </div>
        ))}
      </div>
    ),
    discovery: (
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-cyan-400">Discovery Questions</h3>
        <p className="text-slate-300 text-sm">Use these to qualify the opportunity and gather the inputs needed to run a meaningful analysis in this tool.</p>
        {[
          { cat: "Workload Characterization", qs: [
            "What models are you serving in production, and at what parameter scale?",
            "What are your typical input and output token lengths?",
            "How many concurrent inference sessions do you need to support at peak?",
            "What inference framework are you running — vLLM, TensorRT-LLM, something else?",
            "Are you running multi-turn sessions, or primarily single-turn?",
          ]},
          { cat: "Symptoms & Pain", qs: [
            "What does your GPU utilization look like under peak load?",
            "Have you had to cap concurrent sessions to maintain latency SLOs?",
            "What's your P99 time-to-first-token under load?",
            "How much have you had to over-provision GPU capacity relative to your initial estimate?",
            "Where does your KV Cache go when VRAM fills up — do you know?",
          ]},
          { cat: "Economics", qs: [
            "How many GPU nodes are you currently running for inference?",
            "What's your current monthly GPU infrastructure spend?",
            "Are you buying or renting? If renting, what's your contract term?",
            "What's your projected scale 12 months from now?",
          ]},
          { cat: "Capital Markets Specific", qs: [
            "What's your latency SLO for inference responses — P50 and P99?",
            "Are you running inference against live market data feeds?",
            "Do you have documented data residency requirements for inference context data?",
            "How are you handling audit trails for AI-generated outputs?",
          ]},
          { cat: "Compliance & Security", qs: [
            "Where does your current inference framework write KV Cache when VRAM runs out?",
            "Have you documented your inference data flows for your compliance auditor?",
            "What certifications does your infrastructure need to maintain?",
            "What happens to KV Cache data when a GPU node is reprovisioned or returned?",
          ]},
        ].map(({ cat, qs }) => (
          <div key={cat} className="bg-slate-700/40 rounded-lg p-4">
            <div className="text-sm font-bold text-cyan-400 mb-2">{cat}</div>
            <ul className="space-y-1">
              {qs.map(q => (
                <li key={q} className="text-xs text-slate-300 flex gap-2">
                  <ChevronRight size={12} className="text-cyan-500 mt-0.5 shrink-0" />
                  {q}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    ),
    compliance: (
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-cyan-400">The Compliance Angle: Storage Architecture as a Security Boundary</h3>
        <p className="text-slate-300 text-sm leading-relaxed">Boost Run holds SOC2 Type 1 & 2, HIPAA, and ISO 27001 certifications. This means the compute tier is certified. But compliance auditors are increasingly asking about the <strong className="text-white">inference data pipeline</strong> — specifically, where inference context data lives when it's not in active use on the GPU.</p>
        <div className="space-y-3">
          {[
            { label: "HIPAA", color: "text-green-400", content: "KV Cache in a medical AI application contains live patient context — PHI embedded in inference prompts and multi-turn conversation state. When KV Cache overflows GPU VRAM onto uncontrolled swap or generic NAS, that data leaves the certified infrastructure boundary. DDN AI Fabric, configured within the Boost Run certified environment, keeps inference context data within the auditable, HIPAA-compliant storage tier." },
            { label: "SOC2 / ISO 27001", color: "text-blue-400", content: "SOC2 Type 2 requires demonstrable access controls and audit trails over sensitive data at rest and in transit. Generic NAS solutions typically cannot provide per-session KV Cache access controls or the granular audit trail required for a clean Type 2 opinion. DDN's enterprise storage architecture supports the access control model required." },
            { label: "Federal / FedRAMP", color: "text-purple-400", content: "Federal AI deployments — including Boost Run's Carahsoft channel — have data residency and boundary control requirements that apply to all data, including ephemeral inference state. KV Cache that spills outside the defined infrastructure boundary is a compliance finding. DDN + Boost Run provides a bounded, auditable architecture for federal inference workloads." },
            { label: "Capital Markets (SEC/FINRA)", color: "text-amber-400", content: "Financial firms face increasing regulatory scrutiny of AI system documentation, including data flows. KV Cache — which contains the running context of AI-assisted analyst or compliance workflows — may be subject to recordkeeping requirements. Knowing where it lives and controlling access to it is not optional for regulated broker-dealers and investment advisers." },
          ].map(({ label, color, content }) => (
            <div key={label} className="bg-slate-700/40 rounded-lg p-4">
              <div className={`text-sm font-bold ${color} mb-2 flex items-center gap-2`}><Shield size={14} />{label}</div>
              <div className="text-xs text-slate-300">{content}</div>
            </div>
          ))}
        </div>
      </div>
    ),
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <BookOpen className="text-cyan-400" size={22} />
            <div>
              <div className="text-lg font-bold text-white">Account Manager Playbook</div>
              <div className="text-xs text-slate-400">Boost Run × DDN AI Fabric — Seller's Guide</div>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors p-2 hover:bg-slate-800 rounded-lg">
            <X size={20} />
          </button>
        </div>
        {/* Tab bar */}
        <div className="flex overflow-x-auto border-b border-slate-700 px-6 gap-1 shrink-0">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`text-xs font-medium py-3 px-4 border-b-2 transition-colors whitespace-nowrap ${
                tab === t.id
                  ? "border-cyan-400 text-cyan-400"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {content[tab]}
        </div>
      </div>
    </div>
  );
}

// ── MAIN APP ─────────────────────────────────────────────────────────────────

export default function App() {
  const defaultConfig = {
    gpu: "H100_SXM5",
    llm: "llama3_70b",
    framework: "vllm",
    inputTokens: 8000,
    outputTokens: 1500,
    concurrentSessions: 200,
    ddnBandwidth: DDN_BANDWIDTH,
    nasBandwidth: NAS_BANDWIDTH,
    monthlyRate: 85000,
    contractMonths: 12,
    configMode: "node",
  };

  const [cfg, setCfg] = useState(() => {
    try {
      const saved = localStorage.getItem("kv-cache-cfg");
      return saved ? { ...defaultConfig, ...JSON.parse(saved) } : defaultConfig;
    } catch { return defaultConfig; }
  });

  const [tab, setTab] = useState("physics");
  const [showGuide, setShowGuide] = useState(false);
  const fileRef = useRef();

  useEffect(() => {
    try { localStorage.setItem("kv-cache-cfg", JSON.stringify(cfg)); } catch {}
  }, [cfg]);

  const set = useCallback((k, v) => setCfg(p => ({ ...p, [k]: v })), []);
  const ph = computePhysics(cfg);

  const applyScenario = (key) => {
    const s = SCENARIOS[key];
    setCfg(p => ({
      ...p,
      inputTokens: s.inputTokens,
      outputTokens: s.outputTokens,
      concurrentSessions: s.concurrentSessions,
      gpu: s.gpu,
      llm: s.llm,
      framework: s.framework,
      monthlyRate: GPU_PRESETS[s.gpu].monthly,
    }));
  };

  const exportCfg = () => {
    const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = "kv-cache-config.json"; a.click();
  };

  const importCfg = (e) => {
    const f = e.target.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = (ev) => { try { setCfg(p => ({ ...p, ...JSON.parse(ev.target.result) })); } catch {} };
    r.readAsText(f);
  };

  // ── CHART DATA ─────────────────────────────────────────────────────────────

  const vramChartData = ph ? [
    {
      name: "Current Config",
      "Model Weights": +ph.modelWeightsGB.toFixed(1),
      "Engine Overhead": +ph.engineOverheadGB.toFixed(1),
      "KV Cache (fits)": +Math.min(ph.kvTotalGB, ph.availableForKV).toFixed(1),
      "KV Overflow": +ph.kvOverflow.toFixed(1),
      "Available": +Math.max(0, ph.physicalVRAM - ph.totalVRAMNeeded).toFixed(1),
    }
  ] : [];

  const throughputData = ph ? [
    { name: "Standard NAS\n(~10 GB/s)", swaps: Math.round(ph.swapsPerHourNAS), bandwidth: ph.nasBandwidth },
    { name: "DDN AI Fabric\n(200 GB/s)", swaps: Math.round(ph.swapsPerHourDDN), bandwidth: ph.ddnBandwidth },
  ] : [];

  const sessionScaleData = ph ? Array.from({ length: 10 }, (_, i) => {
    const sessions = Math.round(cfg.concurrentSessions * (0.1 + i * 0.1) * 10 / 10);
    const kv = ph.kvPerSessionGB * sessions;
    const overflow = Math.max(0, kv - ph.availableForKV);
    return {
      sessions,
      "KV Cache (GB)": +kv.toFixed(1),
      "VRAM Ceiling": +ph.physicalVRAM.toFixed(0),
      "Overflow (DDN handles)": +overflow.toFixed(1),
    };
  }) : [];

  const savingsData = ph ? Array.from({ length: 12 }, (_, i) => ({
    month: `M${i + 1}`,
    "Cumulative Rental Savings": Math.round(ph.monthlyRentalSavings * (i + 1)),
    "Cumulative DDN Cost (est.)": Math.round(8000 * (i + 1)), // estimated DDN monthly
  })) : [];

  // ── SIDEBAR ────────────────────────────────────────────────────────────────

  const Sidebar = () => (
    <div className="w-80 shrink-0 bg-slate-900/80 border-r border-slate-800 overflow-y-auto flex flex-col">
      <div className="p-5 space-y-5">
        {/* Scenarios */}
        <div>
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Scenario Presets</div>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(SCENARIOS).map(([k, s]) => (
              <button
                key={k}
                onClick={() => applyScenario(k)}
                className="text-left p-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-cyan-500/50 transition-all group"
              >
                <div className="text-base mb-1">{s.icon}</div>
                <div className="text-xs font-semibold text-slate-200 group-hover:text-cyan-400 leading-tight">{s.name}</div>
              </button>
            ))}
          </div>
        </div>

        <hr className="border-slate-800" />

        {/* GPU */}
        <div>
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
            GPU Configuration
            <Tip label="GPU Node">A server containing multiple GPUs. Boost Run sells server nodes — each with 8 GPUs. Total VRAM is the sum of all GPU memory in the node.</Tip>
          </div>
          <select value={cfg.gpu} onChange={e => { set("gpu", e.target.value); set("monthlyRate", GPU_PRESETS[e.target.value].monthly); }}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none">
            {Object.entries(GPU_PRESETS).map(([k, v]) => <option key={k} value={k}>{v.name} — {v.vram}GB</option>)}
          </select>
        </div>

        {/* LLM */}
        <div>
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
            LLM Model
            <Tip label="LLM (Large Language Model)">The AI model being served. Larger models need more GPU memory for their weights. The number of parameters directly sets the minimum VRAM requirement.</Tip>
          </div>
          <select value={cfg.llm} onChange={e => set("llm", e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none">
            {Object.entries(LLM_PRESETS).map(([k, v]) => <option key={k} value={k}>{v.name} ({v.params}B params)</option>)}
          </select>
        </div>

        {/* Framework */}
        <div>
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
            Inference Framework
            <Tip label="Inference Framework">The software that serves the model. vLLM and TensorRT-LLM are the two dominant production frameworks. TensorRT-LLM is NVIDIA's optimized stack; vLLM is the leading open-source option.</Tip>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(FRAMEWORK_PRESETS).map(([k, v]) => (
              <button key={k} onClick={() => set("framework", k)}
                className={`p-2 rounded-lg text-xs font-semibold border transition-all ${cfg.framework === k ? "bg-cyan-500/20 border-cyan-500 text-cyan-400" : "bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600"}`}>
                {v.name}
              </button>
            ))}
          </div>
        </div>

        <hr className="border-slate-800" />

        {/* Workload */}
        <div>
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Workload Parameters</div>
          {[
            { key: "inputTokens", label: "Input Tokens", tip: "Tokens in the prompt/context. 1 token ≈ 0.75 words. A 32K token context is roughly a 25,000-word document.", min: 128, max: 128000, step: 128 },
            { key: "outputTokens", label: "Output Tokens", tip: "Tokens generated per response. Drives KV Cache growth during the decode phase.", min: 64, max: 8192, step: 64 },
            { key: "concurrentSessions", label: "Concurrent Sessions", tip: "How many simultaneous user sessions the cluster must support. This is the primary multiplier of KV Cache demand.", min: 1, max: 2000, step: 1 },
          ].map(({ key, label, tip, min, max, step }) => (
            <div key={key} className="mb-3">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-slate-400">{label}<Tip label={label}>{tip}</Tip></span>
                <span className="text-xs font-mono text-cyan-400">{cfg[key].toLocaleString()}</span>
              </div>
              <input type="range" min={min} max={max} step={step} value={cfg[key]}
                onChange={e => set(key, +e.target.value)}
                className="w-full accent-cyan-500 h-1.5 rounded" />
            </div>
          ))}
        </div>

        <hr className="border-slate-800" />

        {/* Storage */}
        <div>
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">
            Storage Bandwidth
            <Tip label="Storage Bandwidth">GB/s (gigabytes per second) — how fast data can be read from storage. Higher bandwidth means KV Cache can be recalled faster, reducing the latency penalty of offloading.</Tip>
          </div>
          {[
            { key: "nasBandwidth", label: "Standard NAS (GB/s)", max: 100, color: "accent-amber-500" },
            { key: "ddnBandwidth", label: "DDN AI Fabric (GB/s)", max: 400, color: "accent-cyan-500" },
          ].map(({ key, label, max, color }) => (
            <div key={key} className="mb-3">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-slate-400">{label}</span>
                <span className="text-xs font-mono text-white">{cfg[key]} GB/s</span>
              </div>
              <input type="range" min={1} max={max} value={cfg[key]}
                onChange={e => set(key, +e.target.value)}
                className={`w-full ${color} h-1.5 rounded`} />
            </div>
          ))}
        </div>

        <hr className="border-slate-800" />

        {/* Rental Economics */}
        <div>
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">
            Boost Run Rental Economics
            <Tip label="Rental Economics">Boost Run charges per GPU server node per month. Enter your actual quoted rate to see real dollar savings from reducing your node count with DDN.</Tip>
          </div>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Monthly Rate / Node ($)</label>
              <input type="number" value={cfg.monthlyRate} onChange={e => set("monthlyRate", +e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none font-mono" />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Contract Term (months)</label>
              <input type="number" value={cfg.contractMonths} min={1} max={60} onChange={e => set("contractMonths", +e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none font-mono" />
            </div>
          </div>
        </div>

        <hr className="border-slate-800" />

        {/* Import/Export */}
        <div className="flex gap-2">
          <button onClick={exportCfg} className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-xs text-slate-300 transition-colors">
            <Download size={12} /> Export
          </button>
          <button onClick={() => fileRef.current?.click()} className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-xs text-slate-300 transition-colors">
            <Upload size={12} /> Import
          </button>
          <input ref={fileRef} type="file" accept=".json" className="hidden" onChange={importCfg} />
        </div>
      </div>
    </div>
  );

  // ── INFERENCE PHYSICS TAB ──────────────────────────────────────────────────

  const PhysicsTab = () => {
    if (!ph) return null;
    const barTotal = ph.physicalVRAM;
    const pct = (v) => `${(v / barTotal * 100).toFixed(1)}%`;

    return (
      <div className="space-y-6 p-6">
        {/* Warning banners */}
        {ph.isSystemOverload && (
          <div className="flex items-start gap-3 bg-red-500/10 border border-red-500/40 rounded-xl p-4">
            <AlertTriangle className="text-red-400 shrink-0 mt-0.5" size={18} />
            <div>
              <div className="text-red-400 font-bold text-sm">System Overload — Storage Too Slow</div>
              <div className="text-xs text-slate-300 mt-1">Standard NAS cannot service KV Cache swaps fast enough. Restore time ({fmt.ms(ph.restoreTimeNAS)} per session) exceeds acceptable SLO thresholds. DDN AI Fabric reduces restore time to {fmt.ms(ph.restoreTimeDDN)}.</div>
            </div>
          </div>
        )}
        {ph.isOffloading && !ph.isSystemOverload && (
          <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/40 rounded-xl p-4">
            <AlertTriangle className="text-amber-400 shrink-0 mt-0.5" size={18} />
            <div>
              <div className="text-amber-400 font-bold text-sm">KV Cache Offloading Required — {fmt.gb(ph.kvOverflow)} exceeds VRAM</div>
              <div className="text-xs text-slate-300 mt-1">DDN AI Fabric can absorb this overflow at {cfg.ddnBandwidth} GB/s with a {fmt.ms(ph.restoreTimeDDN)} context restore time. Standard NAS at {cfg.nasBandwidth} GB/s would take {fmt.ms(ph.restoreTimeNAS)} — likely an SLO breach.</div>
            </div>
          </div>
        )}
        {!ph.isOffloading && (
          <div className="flex items-start gap-3 bg-green-500/10 border border-green-500/40 rounded-xl p-4">
            <CheckCircle className="text-green-400 shrink-0 mt-0.5" size={18} />
            <div>
              <div className="text-green-400 font-bold text-sm">KV Cache Fits in VRAM — Scale Up to See Offload Dynamics</div>
              <div className="text-xs text-slate-300 mt-1">Current config fits within {fmt.gb(ph.physicalVRAM)} physical VRAM. Increase concurrent sessions or context length to observe DDN's value under memory pressure.</div>
            </div>
          </div>
        )}

        {/* Key metrics */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard label="Total VRAM Needed" value={fmt.gb(ph.totalVRAMNeeded)} sub={`Physical: ${fmt.gb(ph.physicalVRAM)}`} icon={Cpu} color={ph.isOffloading ? "red" : "cyan"} warn={ph.isOffloading} />
          <MetricCard label="KV Cache Size" value={fmt.gb(ph.kvTotalGB)} sub={`${fmt.gb(ph.kvPerSessionGB)} per session`} icon={Database} color="amber" />
          <MetricCard label="Context Restore — DDN" value={fmt.ms(ph.restoreTimeDDN)} sub={`NAS: ${fmt.ms(ph.restoreTimeNAS)}`} icon={Clock} color="cyan" />
          <MetricCard label="VRAM Utilization" value={fmt.pct(ph.utilizationPct)} sub={ph.isOffloading ? `${fmt.gb(ph.kvOverflow)} overflow` : "Within budget"} icon={Activity} color={ph.utilizationPct > 90 ? "red" : ph.utilizationPct > 70 ? "amber" : "green"} />
        </div>

        {/* HBM Composition stacked bar */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-sm font-bold text-white">GPU HBM Composition
              <Tip label="HBM (High Bandwidth Memory)">The type of RAM inside a GPU. HBM is extremely fast but physically limited — 80GB per H100, for example. Once full, the system must offload data to slower storage.</Tip>
            </h3>
            <div className="text-xs text-slate-400">{ph.gpu.name}</div>
          </div>
          <div className="text-xs text-slate-500 mb-4">How your {fmt.gb(ph.physicalVRAM)} of GPU memory is allocated</div>

          {/* Visual VRAM bar */}
          <div className="relative h-12 rounded-lg overflow-hidden flex mb-4 border border-slate-600">
            <div style={{ width: pct(ph.modelWeightsGB) }} className="bg-blue-600 flex items-center justify-center text-xs font-bold text-white border-r border-slate-700 shrink-0">
              {ph.modelWeightsGB > 20 ? "Weights" : ""}
            </div>
            <div style={{ width: pct(ph.engineOverheadGB) }} className="bg-indigo-500 flex items-center justify-center text-xs font-bold text-white border-r border-slate-700 shrink-0">
              {ph.engineOverheadGB > 15 ? "Overhead" : ""}
            </div>
            <div style={{ width: pct(Math.min(ph.kvTotalGB, ph.availableForKV)) }} className="bg-amber-500 flex items-center justify-center text-xs font-bold text-white border-r border-slate-700 shrink-0">
              {ph.kvTotalGB > 20 ? "KV Cache" : ""}
            </div>
            {ph.kvOverflow > 0 && (
              <div style={{ width: pct(Math.min(ph.kvOverflow, ph.physicalVRAM - ph.modelWeightsGB - ph.engineOverheadGB - Math.min(ph.kvTotalGB, ph.availableForKV))) }} className="bg-red-500 animate-pulse flex items-center justify-center text-xs font-bold text-white shrink-0">
                Overflow
              </div>
            )}
            <div className="flex-1 bg-slate-700 flex items-center justify-center text-xs text-slate-400">
              {ph.kvOverflow === 0 ? "Free" : ""}
            </div>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-4 mb-4">
            {[
              { color: "bg-blue-600",   label: "Model Weights",   val: fmt.gb(ph.modelWeightsGB) },
              { color: "bg-indigo-500", label: "Engine Overhead",  val: fmt.gb(ph.engineOverheadGB) },
              { color: "bg-amber-500",  label: "KV Cache (in VRAM)", val: fmt.gb(Math.min(ph.kvTotalGB, ph.availableForKV)) },
              { color: "bg-red-500",    label: "KV Overflow → DDN", val: ph.kvOverflow > 0 ? fmt.gb(ph.kvOverflow) : "None" },
            ].map(l => (
              <div key={l.label} className="flex items-center gap-1.5 text-xs">
                <span className={`w-2.5 h-2.5 rounded-sm ${l.color}`} />
                <span className="text-slate-400">{l.label}:</span>
                <span className="text-white font-mono">{l.val}</span>
              </div>
            ))}
          </div>

          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={vramChartData} layout="vertical" margin={{ left: 0, right: 20, top: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, ph.physicalVRAM]} unit="GB" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} width={110} />
              <Tooltip content={<ChartTip />} />
              <ReferenceLine x={ph.physicalVRAM} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "VRAM Limit", fill: "#ef4444", fontSize: 10 }} />
              <Bar dataKey="Model Weights" stackId="a" fill="#2563eb" radius={[0,0,0,0]} />
              <Bar dataKey="Engine Overhead" stackId="a" fill="#6366f1" />
              <Bar dataKey="KV Cache (fits)" stackId="a" fill="#f59e0b" />
              <Bar dataKey="KV Overflow" stackId="a" fill="#ef4444" />
              <Bar dataKey="Available" stackId="a" fill="#1e293b" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Context restore comparison */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
            <h3 className="text-sm font-bold text-white mb-4">Context Restore Time
              <Tip label="Context Restore Time">How long it takes to reload one user session's KV Cache from storage back into GPU memory. This time is added to response latency every time a session must be swapped in. Faster storage = lower latency penalty.</Tip>
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1"><span className="text-amber-400">Standard NAS ({cfg.nasBandwidth} GB/s)</span><span className="font-mono text-white">{fmt.ms(ph.restoreTimeNAS)}</span></div>
                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-amber-500 rounded-full" style={{ width: "100%" }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1"><span className="text-cyan-400">DDN AI Fabric ({cfg.ddnBandwidth} GB/s)</span><span className="font-mono text-white">{fmt.ms(ph.restoreTimeDDN)}</span></div>
                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-500 rounded-full" style={{ width: `${Math.max(2, (ph.restoreTimeDDN / ph.restoreTimeNAS) * 100)}%` }} />
                </div>
              </div>
              <div className="bg-cyan-500/10 border border-cyan-500/20 rounded-lg p-3 text-xs text-slate-300">
                DDN is <span className="text-cyan-400 font-bold">{Math.round(cfg.ddnBandwidth / cfg.nasBandwidth)}× faster</span>, reducing per-session restore latency by <span className="text-cyan-400 font-bold">{fmt.ms(ph.restoreTimeNAS - ph.restoreTimeDDN)}</span>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
            <h3 className="text-sm font-bold text-white mb-4">Session Capacity</h3>
            <div className="space-y-3">
              {[
                { label: "Sessions in VRAM (no offload)", val: ph.sessionsInVRAM, color: "text-green-400" },
                { label: "Total sessions required", val: cfg.concurrentSessions, color: "text-white" },
                { label: "Sessions needing DDN offload", val: Math.max(0, cfg.concurrentSessions - ph.sessionsInVRAM), color: "text-amber-400" },
                { label: "Virtual VRAM sessions (DDN)", val: ph.isOffloading ? Math.max(0, cfg.concurrentSessions - ph.sessionsInVRAM) : 0, color: "text-cyan-400" },
              ].map(({ label, val, color }) => (
                <div key={label} className="flex justify-between items-center">
                  <span className="text-xs text-slate-400">{label}</span>
                  <span className={`text-sm font-bold font-mono ${color}`}>{val.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // ── DATA LIFECYCLE TAB ─────────────────────────────────────────────────────

  const DataLifecycleTab = () => {
    if (!ph) return null;
    return (
      <div className="space-y-6 p-6">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard label="KV per Session" value={fmt.gb(ph.kvPerSessionGB)} sub="at current token config" icon={Database} color="amber" />
          <MetricCard label="Total KV Demand" value={fmt.gb(ph.kvTotalGB)} sub={`${cfg.concurrentSessions} sessions`} icon={HardDrive} color={ph.isOffloading ? "red" : "green"} />
          <MetricCard label="DDN Capacity Needed" value={fmt.gb(ph.kvOverflow > 0 ? ph.kvOverflow * 3 : ph.kvTotalGB * 0.5)} sub="with 3× headroom buffer" icon={Server} color="cyan" />
          <MetricCard label="Sessions w/ DDN" value={fmt.num(cfg.concurrentSessions)} sub={`vs ${ph.sessionsInVRAM} without`} icon={Users} color="purple" />
        </div>

        {/* Session scaling chart */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <h3 className="text-sm font-bold text-white mb-1">KV Cache Growth vs. Session Scale
            <Tip label="Scaling Curve">As concurrent sessions increase, KV Cache demand grows linearly. The red dashed line shows physical VRAM. Everything above that line requires DDN offload. Without DDN, sessions above the line cannot be served.</Tip>
          </h3>
          <div className="text-xs text-slate-500 mb-4">How KV Cache demand grows as you scale concurrent sessions — red line is your physical VRAM ceiling</div>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={sessionScaleData} margin={{ top: 10, right: 20, bottom: 0, left: 10 }}>
              <defs>
                <linearGradient id="kvGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="ovGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="sessions" tick={{ fill: "#94a3b8", fontSize: 11 }} label={{ value: "Concurrent Sessions", position: "insideBottom", fill: "#64748b", fontSize: 11, dy: 10 }} />
              <YAxis unit="GB" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip content={<ChartTip />} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} />
              <ReferenceLine y={ph.physicalVRAM} stroke="#ef4444" strokeDasharray="4 4" label={{ value: `VRAM Limit ${fmt.gb(ph.physicalVRAM)}`, fill: "#ef4444", fontSize: 10 }} />
              <Area type="monotone" dataKey="KV Cache (GB)" stroke="#f59e0b" fill="url(#kvGrad)" strokeWidth={2} />
              <Area type="monotone" dataKey="Overflow (DDN handles)" stroke="#ef4444" fill="url(#ovGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Daily/Monthly capacity planning */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <h3 className="text-sm font-bold text-white mb-4">Long-Term Storage Capacity Planning</h3>
          <div className="grid grid-cols-3 gap-4">
            {[
              { period: "Peak Hour", multiplier: 1, unit: "hr" },
              { period: "Daily Total (est.)", multiplier: 24 * 3, unit: "day" },
              { period: "Monthly Total (est.)", multiplier: 24 * 3 * 30, unit: "month" },
            ].map(({ period, multiplier, unit }) => {
              const totalTB = (ph.kvPerSessionGB * cfg.concurrentSessions * multiplier) / 1024;
              return (
                <div key={period} className="bg-slate-900/60 rounded-lg p-4 text-center">
                  <div className="text-xs text-slate-400 mb-1">{period}</div>
                  <div className="text-2xl font-bold text-cyan-400 font-mono">{totalTB < 1 ? `${(totalTB*1024).toFixed(0)}GB` : `${totalTB.toFixed(1)}TB`}</div>
                  <div className="text-xs text-slate-500 mt-1">KV Cache throughput</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  // ── EXECUTIVE REPORT TAB ───────────────────────────────────────────────────

  const ExecutiveTab = () => {
    if (!ph) return null;
    const [priceMode, setPriceMode] = useState("rental");

    return (
      <div className="space-y-6 p-6">
        {/* Mode toggle */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-400">Procurement model:</span>
          {["rental", "capex"].map(m => (
            <button key={m} onClick={() => setPriceMode(m)}
              className={`text-xs font-semibold px-4 py-1.5 rounded-full border transition-all ${priceMode === m ? "bg-cyan-500/20 border-cyan-500 text-cyan-400" : "bg-slate-800 border-slate-700 text-slate-400"}`}>
              {m === "rental" ? "Boost Run Rental" : "Hardware Ownership (CAPEX)"}
            </button>
          ))}
        </div>

        {/* Hero numbers */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {priceMode === "rental" ? <>
            <MetricCard label="Monthly Rental Savings" value={fmt.usd(ph.monthlyRentalSavings)} sub={`${ph.nodesAvoided} nodes × ${fmt.usd(cfg.monthlyRate)}`} icon={DollarSign} color="green" />
            <MetricCard label={`${cfg.contractMonths}-Month Contract Savings`} value={fmt.usd(ph.contractSavings)} sub="vs. current node count" icon={TrendingUp} color="green" />
          </> : <>
            <MetricCard label="CAPEX Avoidance" value={fmt.usd(ph.capexAvoidance)} sub={`${ph.nodesAvoided} nodes avoided`} icon={DollarSign} color="green" />
            <MetricCard label="Annual OPEX Savings" value={fmt.usd(ph.annualOpexSavings)} sub="GPU-hours not consumed" icon={TrendingUp} color="green" />
          </>}
          <MetricCard label="GPU Server Nodes Avoided" value={`${ph.nodesAvoided}`} sub={`${ph.nodesWithDDN} needed vs ${ph.nodesWithoutDDN} without DDN`} icon={Server} color="cyan" />
          <MetricCard label="Throughput Gain" value={`${Math.round(cfg.ddnBandwidth / cfg.nasBandwidth)}×`} sub="DDN vs Standard NAS bandwidth" icon={Zap} color="purple" />
        </div>

        {/* Throughput comparison chart */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
          <h3 className="text-sm font-bold text-white mb-1">Sustainable Inference Sessions Per Hour
            <Tip label="Sustainable Swaps/Hr">How many KV Cache swap operations the storage tier can support per hour without becoming the bottleneck. If this number is lower than your session demand, the GPU is waiting on storage — not computing.</Tip>
          </h3>
          <div className="text-xs text-slate-500 mb-4">Impact on inference throughput per Boost Run server node — Standard NAS vs. DDN AI Fabric</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={throughputData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="swaps" name="Sessions/hr" fill="#22d3ee" radius={[6, 6, 0, 0]}>
                {throughputData.map((_, i) => (
                  <rect key={i} fill={i === 0 ? "#f59e0b" : "#22d3ee"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Hardware savings breakdown */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
            <h3 className="text-sm font-bold text-white mb-4">Node Count: With vs. Without DDN</h3>
            <div className="space-y-3">
              {[
                { label: "Nodes without DDN", val: ph.nodesWithoutDDN, color: "text-red-400", bar: "bg-red-500" },
                { label: "Nodes with DDN AI Fabric", val: ph.nodesWithDDN, color: "text-green-400", bar: "bg-green-500" },
                { label: "Nodes eliminated", val: ph.nodesAvoided, color: "text-cyan-400", bar: "bg-cyan-500" },
              ].map(({ label, val, color, bar }) => (
                <div key={label}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">{label}</span>
                    <span className={`font-mono font-bold ${color}`}>{val}</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div className={`h-full ${bar} rounded-full transition-all`} style={{ width: `${Math.min(100, (val / Math.max(ph.nodesWithoutDDN, 1)) * 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {priceMode === "rental" ? (
            <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
              <h3 className="text-sm font-bold text-white mb-4">Cumulative Rental Savings vs. DDN Cost</h3>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={savingsData} margin={{ top: 5, right: 10, bottom: 0, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="month" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
                  <Tooltip content={<ChartTip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="Cumulative Rental Savings" stroke="#22d3ee" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Cumulative DDN Cost (est.)" stroke="#f59e0b" strokeWidth={2} dot={false} strokeDasharray="4 4" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
              <h3 className="text-sm font-bold text-white mb-4">CAPEX & OPEX Summary</h3>
              <div className="space-y-3 text-sm">
                {[
                  { label: "GPU nodes avoided", val: `${ph.nodesAvoided}`, color: "text-cyan-400" },
                  { label: "CAPEX per node", val: fmt.usd(ph.gpu.capex), color: "text-slate-300" },
                  { label: "Total CAPEX avoidance", val: fmt.usd(ph.capexAvoidance), color: "text-green-400" },
                  { label: "Annual OPEX saving", val: fmt.usd(ph.annualOpexSavings), color: "text-green-400" },
                ].map(({ label, val, color }) => (
                  <div key={label} className="flex justify-between border-b border-slate-700/50 pb-2">
                    <span className="text-slate-400">{label}</span>
                    <span className={`font-mono font-bold ${color}`}>{val}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Export for RFB */}
        <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30 rounded-xl p-5 flex items-center justify-between">
          <div>
            <div className="text-sm font-bold text-white">Ready to Build?</div>
            <div className="text-xs text-slate-400 mt-0.5">Export this analysis to initiate a Boost Run Request for Build with DDN AI Fabric as the storage tier</div>
          </div>
          <button onClick={exportCfg} className="flex items-center gap-2 bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-bold text-sm px-5 py-2.5 rounded-lg transition-colors shrink-0">
            <FileText size={14} /> Export for RFB
          </button>
        </div>
      </div>
    );
  };

  // ── RENDER ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-white font-sans overflow-hidden">
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3.5 border-b border-slate-800 bg-slate-900/90 backdrop-blur-sm shrink-0 z-10">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center">
              <Zap size={14} className="text-white" />
            </div>
            <div>
              <div className="text-sm font-bold text-white leading-none">KV Cache Storage Estimator</div>
              <div className="text-xs text-slate-400 leading-none mt-0.5">Boost Run × DDN AI Fabric</div>
            </div>
          </div>

          {/* Nav tabs */}
          <nav className="flex items-center gap-1 ml-6">
            {[
              { id: "physics",   label: "Inference Physics",   icon: Cpu },
              { id: "lifecycle", label: "Data Lifecycle",       icon: HardDrive },
              { id: "executive", label: "Executive Report",     icon: TrendingUp },
            ].map(({ id, label, icon: Icon }) => (
              <button key={id} onClick={() => setTab(id)}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                  tab === id
                    ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                }`}>
                <Icon size={13} />{label}
              </button>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-3">
          {ph?.isOffloading && (
            <div className="flex items-center gap-1.5 text-xs bg-amber-500/10 border border-amber-500/30 text-amber-400 px-3 py-1.5 rounded-full">
              <AlertTriangle size={11} /> KV Offloading Active
            </div>
          )}
          {ph?.isSystemOverload && (
            <div className="flex items-center gap-1.5 text-xs bg-red-500/10 border border-red-500/30 text-red-400 px-3 py-1.5 rounded-full animate-pulse">
              <AlertTriangle size={11} /> System Overload
            </div>
          )}
          <button onClick={() => setShowGuide(true)}
            className="flex items-center gap-1.5 bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-bold text-xs px-4 py-2 rounded-lg transition-colors">
            <BookOpen size={13} /> Seller's Guide
          </button>
        </div>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          {tab === "physics"   && <PhysicsTab />}
          {tab === "lifecycle" && <DataLifecycleTab />}
          {tab === "executive" && <ExecutiveTab />}
        </main>
      </div>

      {showGuide && <SellersGuide onClose={() => setShowGuide(false)} />}
    </div>
  );
}

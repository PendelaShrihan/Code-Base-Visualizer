import { useState } from 'react'

export default function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6 font-sans">
      <div className="max-w-xl w-full bg-slate-900/90 border border-slate-800 rounded-2xl p-8 shadow-2xl backdrop-blur-sm">
        <div className="flex items-center gap-3 mb-6">
          <span className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-xl font-bold">
            ⚡
          </span>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              CodeBase Visualizer Frontend
            </h1>
            <p className="text-sm text-slate-400">
              Vite + React + Tailwind CSS v4 Pipeline Verified
            </p>
          </div>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-5 border border-slate-800/80 mb-6 space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">Vite Bundler</span>
            <span className="font-mono text-xs px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-semibold">
              Active (HMR)
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">Tailwind Engine</span>
            <span className="font-mono text-xs px-2.5 py-1 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 font-semibold">
              @tailwindcss/vite v4
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">React Runtime</span>
            <span className="font-mono text-xs px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 font-semibold">
              React 19
            </span>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-center gap-4">
          <button
            type="button"
            id="test-counter-btn"
            onClick={() => setCount((c) => c + 1)}
            className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-sm transition-all duration-150 active:scale-95 shadow-lg shadow-cyan-500/20 cursor-pointer"
          >
            Interactive Counter: {count}
          </button>
          <span className="text-xs text-slate-400">
            Click to verify React state and Tailwind transition effects
          </span>
        </div>
      </div>
    </div>
  )
}

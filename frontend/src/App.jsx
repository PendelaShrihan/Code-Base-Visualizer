import { useState } from 'react'

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'system',
      role: 'System',
      time: '16:56:01',
      text: 'Repository graph loaded: PendelaShrihan/Code-Base-Visualizer initialized into AST vector space.'
    },
    {
      id: 2,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:56:05',
      text: 'Welcome! I have indexed your codebase architecture. The Canvas Map on the left (60% width on desktop) renders visual node relationships, while this Chat Drawer (40% width) provides interactive architectural analysis.'
    },
    {
      id: 3,
      sender: 'user',
      role: 'Engineer',
      time: '16:56:12',
      text: 'Can you verify the panel split and confirm that only this drawer scrolls when filled with messages?'
    },
    {
      id: 4,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:56:14',
      text: 'Affirmative. On desktop (>= 768px), the layout utilizes md:flex-row with md:w-[60%] for Canvas Map and md:w-[40%] for Chat Drawer across the full viewport height (h-screen).'
    },
    {
      id: 5,
      sender: 'system',
      role: 'Layout Diagnostic',
      time: '16:56:15',
      text: 'Verified: overflow-y-auto is active exclusively on the message drawer body. Neither the main window nor the canvas map will scroll as chat content expands.'
    },
    {
      id: 6,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:56:20',
      text: 'Module Breakdown: Detected React 19 + Tailwind CSS v4 frontend engine communicating with Python FastAPI AST parser.'
    },
    {
      id: 7,
      sender: 'user',
      role: 'Engineer',
      time: '16:56:30',
      text: 'What happens when the browser window is resized below the md breakpoint?'
    },
    {
      id: 8,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:56:32',
      text: 'The responsive flex-col rule activates automatically below 768px! The Canvas Map stacks gracefully above the Chat Drawer, retaining full accessibility on mobile and tablet devices.'
    },
    {
      id: 9,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:56:45',
      text: 'AST Graph Summary: 14 Python modules, 42 syntax tree nodes, 18 dependency relationships mapped.'
    },
    {
      id: 10,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:57:00',
      text: 'Deep-dive into parser/graph_generator.py reveals topological sorting applied to eliminate circular import warnings.'
    },
    {
      id: 11,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:57:15',
      text: 'Vector Store: ChromaDB / pgvector collection indexed with cosine distance threshold 0.82.'
    },
    {
      id: 12,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:57:30',
      text: 'Security Scanner: No secret leaks detected in workspace commit history.'
    },
    {
      id: 13,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:57:45',
      text: 'Scroll Confirmation Item: You are currently scrolling within the independent 40% Chat Drawer. Notice how the page header, canvas viewport, and overall layout remain locked in position.'
    },
    {
      id: 14,
      sender: 'assistant',
      role: 'CodeBase Copilot',
      time: '16:58:00',
      text: 'End of current log stream. Feel free to use the quick-add button above or the input bar below to generate additional scroll testing paragraphs.'
    }
  ])

  const [inputVal, setInputVal] = useState('')
  const [zoom, setZoom] = useState(100)

  const handleSendMessage = (e) => {
    e?.preventDefault()
    if (!inputVal.trim()) return

    const now = new Date()
    const timeStr = now.toTimeString().split(' ')[0]

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        sender: 'user',
        role: 'Engineer',
        time: timeStr,
        text: inputVal.trim()
      },
      {
        id: Date.now() + 1,
        sender: 'assistant',
        role: 'CodeBase Copilot',
        time: timeStr,
        text: `Echo response to "${inputVal.trim()}": Chat Drawer scroll overflow remains isolated to this panel.`
      }
    ])
    setInputVal('')
  }

  const addTestParagraph = () => {
    const nextId = messages.length + 1
    const now = new Date()
    const timeStr = now.toTimeString().split(' ')[0]
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        sender: 'assistant',
        role: 'Scroll Test Log',
        time: timeStr,
        text: `Diagnostic Test #${nextId}: Additional placeholder paragraph verifying that container scroll height (${prev.length + 1} items) exceeds bounds without inducing window-level scrolling.`
      }
    ])
  }

  return (
    // Full Viewport Height Root Container with Responsive Stacking (flex-col -> md:flex-row)
    <div className="h-screen w-full flex flex-col md:flex-row overflow-hidden bg-slate-950 text-slate-100 font-sans select-none">
      
      {/* ========================================================================= */}
      {/* LEFT PANEL: Canvas Map (60% width on md+, top panel on mobile)           */}
      {/* ========================================================================= */}
      <section
        id="canvas-map-panel"
        className="w-full md:w-[60%] md:basis-[60%] h-1/2 md:h-full flex flex-col shrink-0 border-b md:border-b-0 md:border-r border-slate-800/90 bg-gradient-to-br from-slate-950 via-slate-900/90 to-cyan-950/20 relative overflow-hidden"
      >
        {/* Canvas Top Bar */}
        <header className="h-14 px-4 sm:px-6 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur flex items-center justify-between shrink-0 z-10">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 flex items-center justify-center font-bold text-sm shadow-sm shadow-cyan-500/20">
              ⚡
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-semibold tracking-tight text-white">Canvas Map</h1>
                <span className="hidden sm:inline-block text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                  Left 60% Width
                </span>
              </div>
              <p className="text-[11px] text-slate-400">Interactive Architecture &amp; AST Visualization</p>
            </div>
          </div>

          {/* Responsive breakpoint indicator badge & canvas controls */}
          <div className="flex items-center gap-2">
            <span className="hidden md:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[11px] font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              Desktop: md:flex-row (60/40)
            </span>
            <span className="inline-flex md:hidden items-center gap-1.5 px-2 py-0.5 rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20 text-[10px] font-mono">
              Narrow: flex-col
            </span>

            <div className="hidden sm:flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-0.5">
              <button
                type="button"
                onClick={() => setZoom((z) => Math.max(z - 10, 50))}
                className="w-7 h-7 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 rounded text-xs transition cursor-pointer"
                title="Zoom Out"
              >
                -
              </button>
              <span className="text-[11px] font-mono px-1.5 text-slate-300 min-w-[42px] text-center">
                {zoom}%
              </span>
              <button
                type="button"
                onClick={() => setZoom((z) => Math.min(z + 10, 150))}
                className="w-7 h-7 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 rounded text-xs transition cursor-pointer"
                title="Zoom In"
              >
                +
              </button>
            </div>
          </div>
        </header>

        {/* Visual Canvas Content / Colored Placeholder Area */}
        <div className="flex-1 relative flex flex-col items-center justify-center p-3 sm:p-6 overflow-hidden min-h-0">
          {/* Subtle Cyber Grid Background */}
          <div 
            className="absolute inset-0 opacity-20 pointer-events-none"
            style={{
              backgroundImage: 'radial-gradient(#38bdf8 1px, transparent 1px), radial-gradient(#0284c7 1px, transparent 1px)',
              backgroundSize: '32px 32px',
              backgroundPosition: '0 0, 16px 16px'
            }}
          />

          {/* Prominent Colored Canvas Map Placeholder Card */}
          <div className="relative z-10 w-full max-w-lg rounded-xl sm:rounded-2xl bg-slate-900/85 border border-cyan-500/40 p-4 sm:p-6 shadow-2xl shadow-cyan-950/50 backdrop-blur-md">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                <span className="w-2 h-2 rounded-full bg-cyan-400 -ml-3.5"></span>
                <span className="text-[11px] uppercase tracking-widest font-mono font-bold text-cyan-400">
                  Primary Viewport
                </span>
              </div>
              <span className="text-[10px] font-mono text-slate-400 bg-slate-950/70 px-2 py-0.5 rounded border border-slate-800">
                flex-[60%]
              </span>
            </div>

            <div className="text-center space-y-1 sm:space-y-2 mb-3 sm:mb-5">
              <div className="inline-block px-3.5 py-1 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 font-bold text-base sm:text-xl tracking-tight">
                Canvas Map
              </div>
              <p className="text-[11px] sm:text-xs text-slate-300 max-w-sm mx-auto line-clamp-2">
                Placeholder ready for graph rendering engine, AST trees, and dependency topology maps.
              </p>
            </div>

            {/* Simulated Architecture Nodes */}
            <div className="grid grid-cols-3 gap-2 mb-3 text-left">
              <div className="p-2 rounded-lg bg-slate-950/70 border border-cyan-500/30 hover:border-cyan-400 transition group">
                <span className="text-[9px] font-mono text-cyan-400 block">Node 01</span>
                <span className="text-[11px] font-semibold text-slate-200 group-hover:text-white truncate block">API Gateway</span>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/70 border border-emerald-500/30 hover:border-emerald-400 transition group">
                <span className="text-[9px] font-mono text-emerald-400 block">Node 02</span>
                <span className="text-[11px] font-semibold text-slate-200 group-hover:text-white truncate block">AST Parser</span>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/70 border border-indigo-500/30 hover:border-indigo-400 transition group">
                <span className="text-[9px] font-mono text-indigo-400 block">Node 03</span>
                <span className="text-[11px] font-semibold text-slate-200 group-hover:text-white truncate block">Vector DB</span>
              </div>
            </div>

            <div className="flex items-center justify-between text-[10px] sm:text-[11px] font-mono text-slate-400 pt-2 border-t border-slate-800/80">
              <span>Status: React Flow Ready</span>
              <span className="text-cyan-400 font-semibold">Height: 100vh locked</span>
            </div>
          </div>
        </div>

        {/* Canvas Footer Status */}
        <footer className="h-9 px-4 sm:px-6 bg-slate-950/90 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400 shrink-0">
          <div className="flex items-center gap-3">
            <span>Viewport: 60% Width</span>
            <span className="hidden sm:inline text-slate-600">|</span>
            <span className="hidden sm:inline">Coordinates: X: 480, Y: 320</span>
          </div>
          <div className="flex items-center gap-2 text-cyan-400">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
            <span>Non-scrolling Map Root</span>
          </div>
        </footer>
      </section>

      {/* ========================================================================= */}
      {/* RIGHT PANEL: Chat Drawer (40% width on md+, bottom panel on mobile)        */}
      {/* ========================================================================= */}
      <aside
        id="chat-drawer-panel"
        className="w-full md:w-[40%] md:basis-[40%] h-1/2 md:h-full flex flex-col bg-slate-900/95 relative overflow-hidden"
      >
        {/* Chat Drawer Top Header (Fixed at top of drawer) */}
        <header className="h-14 px-4 sm:px-6 border-b border-slate-800/90 bg-slate-950/90 backdrop-blur flex items-center justify-between shrink-0 z-10">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/15 border border-indigo-500/30 text-indigo-400 flex items-center justify-center font-bold text-sm shadow-sm shadow-indigo-500/20">
              💬
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold tracking-tight text-white">Chat Drawer</h2>
                <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/30">
                  Right 40% Width
                </span>
              </div>
              <p className="text-[11px] text-slate-400">Codebase Q&amp;A · Architecture Reasoning</p>
            </div>
          </div>

          {/* Quick scroll test paragraph injector button */}
          <button
            type="button"
            id="add-message-btn"
            onClick={addTestParagraph}
            className="px-2.5 py-1 rounded-lg bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 border border-indigo-500/30 text-xs font-medium transition cursor-pointer flex items-center gap-1.5"
            title="Inject another paragraph to test vertical scrolling"
          >
            <span>+</span>
            <span className="hidden sm:inline">Add Test &lt;p&gt;</span>
          </button>
        </header>

        {/* Highlighted Banner Explaining Scroll Isolation */}
        <div className="px-4 py-2 bg-indigo-950/40 border-b border-indigo-500/20 flex items-center justify-between text-[11px] text-indigo-200 shrink-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-indigo-400 font-mono">overflow-y-auto:</span>
            <span>Only this chat panel scrolls. Page stays locked.</span>
          </div>
          <span className="font-mono text-[10px] text-indigo-400/80 bg-indigo-900/50 px-2 py-0.5 rounded">
            {messages.length} items
          </span>
        </div>

        {/* Scrollable Message Container (overflow-y-auto active here!) */}
        <div
          id="chat-messages-scroll-area"
          className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-3.5 scroll-smooth"
        >
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`rounded-xl p-3.5 border transition text-xs sm:text-sm leading-relaxed ${
                msg.sender === 'user'
                  ? 'bg-cyan-950/40 border-cyan-500/30 text-cyan-100 ml-4 sm:ml-8'
                  : msg.sender === 'system'
                  ? 'bg-slate-950/70 border-slate-800 text-slate-400 font-mono text-[11px]'
                  : 'bg-slate-950/90 border-slate-800/90 text-slate-200 mr-4 sm:mr-8 shadow-sm'
              }`}
            >
              <div className="flex items-center justify-between mb-1 text-[11px] font-medium opacity-70">
                <span className={msg.sender === 'user' ? 'text-cyan-400 font-semibold' : 'text-indigo-300'}>
                  {msg.role}
                </span>
                <span className="font-mono text-[10px] text-slate-400">{msg.time}</span>
              </div>
              <p className="whitespace-pre-line select-text">{msg.text}</p>
            </div>
          ))}
        </div>

        {/* Chat Drawer Input Bar (Fixed at bottom of drawer) */}
        <footer className="p-3 sm:p-4 bg-slate-950/90 border-t border-slate-800/90 shrink-0">
          <form onSubmit={handleSendMessage} className="flex items-center gap-2">
            <input
              type="text"
              id="chat-input-field"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              placeholder="Ask about codebase architecture or test scroll..."
              className="flex-1 bg-slate-900 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-3.5 py-2 text-xs sm:text-sm text-slate-100 placeholder-slate-500 outline-none transition"
            />
            <button
              type="submit"
              id="chat-send-btn"
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs sm:text-sm transition shadow-lg shadow-indigo-600/20 active:scale-95 cursor-pointer shrink-0"
            >
              Send
            </button>
          </form>
          <div className="flex items-center justify-between mt-2 text-[10px] font-mono text-slate-400">
            <span>Enter to send · Shift+Enter for multiline</span>
            <span>Panel: 40% Drawer</span>
          </div>
        </footer>
      </aside>

    </div>
  )
}

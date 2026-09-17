import { useState, useRef, useCallback, useEffect } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'

// Hardcoded sample graph elements matching the { data: { ... } } shape
const SAMPLE_ELEMENTS = [
  // Graph Nodes
  {
    data: {
      id: 'api_gateway',
      label: 'API Gateway',
      type: 'service',
      desc: 'FastAPI routing, CORS & validation'
    }
  },
  {
    data: {
      id: 'ast_parser',
      label: 'AST Parser',
      type: 'parser',
      desc: 'Tree-sitter module & symbol extraction'
    }
  },
  {
    data: {
      id: 'vector_db',
      label: 'Vector DB',
      type: 'database',
      desc: 'ChromaDB / Qdrant vector index'
    }
  },
  {
    data: {
      id: 'hybrid_retriever',
      label: 'Hybrid Retriever',
      type: 'core',
      desc: 'Dense + BM25 reciprocal rank fusion'
    }
  },
  {
    data: {
      id: 'graph_service',
      label: 'Graph Service',
      type: 'service',
      desc: 'NetworkX dependency & topology engine'
    }
  },
  {
    data: {
      id: 'worker_task',
      label: 'Worker Queue',
      type: 'worker',
      desc: 'Celery / Redis background ingestion'
    }
  },
  {
    data: {
      id: 'frontend_client',
      label: 'Frontend UI',
      type: 'client',
      desc: 'React 19 Cytoscape visualizer'
    }
  },
  {
    data: {
      id: 'llm_engine',
      label: 'LLM Engine',
      type: 'core',
      desc: 'Context-augmented architecture assistant'
    }
  },

  // Graph Edges
  { data: { id: 'e1', source: 'frontend_client', target: 'api_gateway', label: 'HTTP / WS' } },
  { data: { id: 'e2', source: 'api_gateway', target: 'ast_parser', label: 'parse repo' } },
  { data: { id: 'e3', source: 'api_gateway', target: 'graph_service', label: 'query graph' } },
  { data: { id: 'e4', source: 'ast_parser', target: 'graph_service', label: 'nodes & edges' } },
  { data: { id: 'e5', source: 'ast_parser', target: 'worker_task', label: 'async queue' } },
  { data: { id: 'e6', source: 'worker_task', target: 'vector_db', label: 'embeddings' } },
  { data: { id: 'e7', source: 'api_gateway', target: 'hybrid_retriever', label: 'search' } },
  { data: { id: 'e8', source: 'hybrid_retriever', target: 'vector_db', label: 'similarity' } },
  { data: { id: 'e9', source: 'hybrid_retriever', target: 'llm_engine', label: 'augmented ctx' } },
  { data: { id: 'e10', source: 'llm_engine', target: 'api_gateway', label: 'stream response' } }
]

// COSE (Compound Spring Embedder) force-directed layout configuration
const COSE_LAYOUT = {
  name: 'cose'
}

// Cytoscape visual stylesheet tailored to match dark cyber aesthetic
const CYTOSCAPE_STYLES = [
  {
    selector: 'node',
    style: {
      'label': 'data(label)',
      'color': '#f8fafc',
      'font-family': 'Inter, system-ui, sans-serif',
      'font-size': '11px',
      'font-weight': 600,
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '72px',
      'background-color': '#0f172a',
      'border-width': 2,
      'border-color': '#06b6d4',
      'width': 68,
      'height': 68,
      'shape': 'round-rectangle',
      'border-opacity': 0.95,
      'background-opacity': 0.95,
      'transition-property': 'background-color, border-color, width, height, border-width',
      'transition-duration': '0.2s'
    }
  },
  {
    selector: 'node[type = "service"]',
    style: {
      'border-color': '#38bdf8',
      'background-color': '#075985'
    }
  },
  {
    selector: 'node[type = "parser"]',
    style: {
      'border-color': '#10b981',
      'background-color': '#065f46'
    }
  },
  {
    selector: 'node[type = "database"]',
    style: {
      'border-color': '#818cf8',
      'background-color': '#3730a3'
    }
  },
  {
    selector: 'node[type = "core"]',
    style: {
      'border-color': '#f59e0b',
      'background-color': '#78350f'
    }
  },
  {
    selector: 'node[type = "worker"]',
    style: {
      'border-color': '#ec4899',
      'background-color': '#831843'
    }
  },
  {
    selector: 'node[type = "client"]',
    style: {
      'border-color': '#a855f7',
      'background-color': '#581c87'
    }
  },
  {
    selector: 'node:selected',
    style: {
      'border-color': '#ffffff',
      'border-width': 3.5
    }
  },
  {
    selector: 'edge',
    style: {
      'width': 2,
      'line-color': '#475569',
      'target-arrow-color': '#94a3b8',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'arrow-scale': 1.1,
      'opacity': 0.85,
      'label': 'data(label)',
      'font-size': '10px',
      'font-family': 'monospace',
      'color': '#cbd5e1',
      'text-rotation': 'autorotate',
      'text-margin-y': -8,
      'text-background-color': '#090d16',
      'text-background-opacity': 0.9,
      'text-background-padding': '3px',
      'text-background-shape': 'round-rectangle',
      'text-border-color': '#334155',
      'text-border-width': 1,
      'text-border-opacity': 0.6
    }
  },
  {
    selector: 'edge:selected',
    style: {
      'width': 3,
      'line-color': '#38bdf8',
      'target-arrow-color': '#38bdf8',
      'text-border-color': '#38bdf8',
      'opacity': 1
    }
  }
]

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
      text: 'AST Graph Summary: 8 primary modules, 10 dependency relationships actively visualized with Cytoscape force-directed layout.'
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
  const [panCoord, setPanCoord] = useState({ x: 0, y: 0 })
  const [selectedElement, setSelectedElement] = useState(null)

  const cyRef = useRef(null)
  const containerRef = useRef(null)

  // Cytoscape initialization and event binding
  const handleCy = useCallback((cy) => {
    if (cyRef.current === cy) return
    cyRef.current = cy

    const updateViewportStats = () => {
      if (!cy) return
      setZoom(Math.round(cy.zoom() * 100))
      const pan = cy.pan()
      setPanCoord({ x: Math.round(pan.x), y: Math.round(pan.y) })
    }

    cy.on('zoom', updateViewportStats)
    cy.on('pan', updateViewportStats)

    cy.on('select', 'node', (e) => {
      const node = e.target
      setSelectedElement({
        type: 'node',
        id: node.id(),
        label: node.data('label'),
        category: node.data('type'),
        desc: node.data('desc'),
        degree: node.degree()
      })
    })

    cy.on('unselect', 'node', () => {
      setSelectedElement(null)
    })

    cy.on('tap', (e) => {
      if (e.target === cy) {
        setSelectedElement(null)
      }
    })

    // Update viewport stats when layout completes or graph is ready
    cy.once('layoutstop', () => {
      updateViewportStats()
    })

    cy.ready(() => {
      updateViewportStats()
    })
  }, [])

  // Auto-resize Cytoscape viewport on container dimensions change
  useEffect(() => {
    if (!containerRef.current) return
    const resizeObserver = new ResizeObserver(() => {
      if (cyRef.current) {
        cyRef.current.resize()
      }
    })
    resizeObserver.observe(containerRef.current)
    return () => resizeObserver.disconnect()
  }, [])

  const handleZoomIn = () => {
    if (!cyRef.current) return
    const cy = cyRef.current
    cy.zoom({
      level: cy.zoom() * 1.25,
      renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 }
    })
    setZoom(Math.round(cy.zoom() * 100))
  }

  const handleZoomOut = () => {
    if (!cyRef.current) return
    const cy = cyRef.current
    cy.zoom({
      level: cy.zoom() * 0.8,
      renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 }
    })
    setZoom(Math.round(cy.zoom() * 100))
  }

  const handleFit = () => {
    if (!cyRef.current) return
    cyRef.current.fit(null, 45)
    setZoom(Math.round(cyRef.current.zoom() * 100))
  }

  const handleResetLayout = () => {
    if (!cyRef.current) return
    const layout = cyRef.current.layout(COSE_LAYOUT)
    layout.run()
  }

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
        className="w-full md:w-[60%] md:basis-[60%] h-1/2 md:h-full flex flex-col shrink-0 border-b md:border-b-0 md:border-r border-slate-800/90 bg-slate-950 relative overflow-hidden"
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

            {/* Interactive Cytoscape Canvas Controls */}
            <div className="hidden sm:flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-0.5">
              <button
                type="button"
                id="canvas-zoom-out-btn"
                onClick={handleZoomOut}
                className="w-7 h-7 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 rounded text-xs transition cursor-pointer"
                title="Zoom Out"
              >
                -
              </button>
              <span id="canvas-zoom-level" className="text-[11px] font-mono px-1.5 text-slate-300 min-w-[44px] text-center">
                {zoom}%
              </span>
              <button
                type="button"
                id="canvas-zoom-in-btn"
                onClick={handleZoomIn}
                className="w-7 h-7 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 rounded text-xs transition cursor-pointer"
                title="Zoom In"
              >
                +
              </button>
              <button
                type="button"
                id="canvas-fit-btn"
                onClick={handleFit}
                className="px-2 h-7 flex items-center justify-center text-slate-300 hover:text-white hover:bg-slate-800 rounded text-[10px] font-mono transition cursor-pointer"
                title="Fit to Canvas"
              >
                Fit
              </button>
              <button
                type="button"
                id="canvas-relayout-btn"
                onClick={handleResetLayout}
                className="px-2 h-7 flex items-center justify-center text-cyan-400 hover:text-cyan-300 hover:bg-cyan-950/40 rounded text-[10px] font-mono transition cursor-pointer border border-cyan-800/40"
                title="Re-run Force-Directed (COSE) Layout"
              >
                COSE
              </button>
            </div>
          </div>
        </header>

        {/* Visual Cytoscape Graph Canvas Area */}
        <div
          ref={containerRef}
          id="cytoscape-canvas-container"
          className="flex-1 relative w-full h-full min-h-0 overflow-hidden bg-slate-950/60"
        >
          {/* Subtle Cyber Grid Background */}
          <div 
            className="absolute inset-0 opacity-15 pointer-events-none"
            style={{
              backgroundImage: 'radial-gradient(#38bdf8 1px, transparent 1px), radial-gradient(#0284c7 1px, transparent 1px)',
              backgroundSize: '32px 32px',
              backgroundPosition: '0 0, 16px 16px'
            }}
          />

          {/* Actual Cytoscape Component */}
          <CytoscapeComponent
            id="cytoscape-graph"
            elements={SAMPLE_ELEMENTS}
            layout={COSE_LAYOUT}
            stylesheet={CYTOSCAPE_STYLES}
            style={{ width: '100%', height: '100%' }}
            cy={handleCy}
            className="w-full h-full cursor-grab active:cursor-grabbing"
          />

          {/* Overlay Status Badge */}
          <div className="absolute top-3 left-3 z-10 pointer-events-none flex flex-col gap-1.5">
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-950/85 border border-cyan-500/30 backdrop-blur text-[11px] font-mono text-cyan-300 shadow-lg">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
              <span>Layout: COSE Force-Directed</span>
            </div>
          </div>

          {/* Node Inspector Overlay (When node is clicked/selected) */}
          {selectedElement && (
            <div
              id="node-inspector-card"
              className="absolute bottom-3 left-3 z-10 p-3 rounded-xl bg-slate-950/90 border border-cyan-500/50 backdrop-blur-md text-xs shadow-xl shadow-cyan-950/50 max-w-xs"
            >
              <div className="flex items-center justify-between gap-3 mb-1.5">
                <span className="font-bold text-white text-sm">{selectedElement.label}</span>
                <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono text-[10px] uppercase border border-cyan-500/30">
                  {selectedElement.category}
                </span>
              </div>
              <p className="text-slate-300 text-[11px] mb-2">{selectedElement.desc}</p>
              <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 border-t border-slate-800 pt-1.5">
                <span>ID: {selectedElement.id}</span>
                <span className="text-cyan-400 font-semibold">{selectedElement.degree} Connections</span>
              </div>
            </div>
          )}

          {/* Interactive Navigation Tips */}
          <div className="absolute bottom-3 right-3 z-10 hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-950/70 border border-slate-800/80 backdrop-blur text-[10px] font-mono text-slate-400 pointer-events-none">
            <span>🖱️ Pan: Drag BG · Zoom: Scroll · Move: Drag Node</span>
          </div>
        </div>

        {/* Canvas Footer Status */}
        <footer className="h-9 px-4 sm:px-6 bg-slate-950/90 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400 shrink-0">
          <div className="flex items-center gap-3">
            <span>Viewport: 60% Width</span>
            <span className="hidden sm:inline text-slate-600">|</span>
            <span className="hidden sm:inline">Pan: X: {panCoord.x}, Y: {panCoord.y}</span>
            <span className="hidden sm:inline text-slate-600">|</span>
            <span className="text-cyan-300 font-semibold">8 Nodes, 10 Edges</span>
          </div>
          <div className="flex items-center gap-2 text-cyan-400">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
            <span>Cytoscape Force Graph</span>
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

import { useState, useRef, useCallback, useEffect } from 'react'
import GraphCanvas, { FCOSE_LAYOUT, activateLevel4Trace, exitLevel4Trace } from './components/GraphCanvas'

// ---------------------------------------------------------------------------
// Transform  nx.node_link_data() JSON  ->  Cytoscape elements
// ---------------------------------------------------------------------------
// Backend stores graph via nx.node_link_data():
//   { directed, multigraph, graph, nodes: [...], edges: [...] }
// (NX >= 3.0 uses "edges"; older builds used "links" -- we handle both.)
//
// Each node: id, kind, label, path/file, pagerank, commit_count,
//            is_dead_code_candidate, name, ...
// Each edge: source, target, rel, edge_type
//
function transformGraphToCytoscape(rawGraph) {
  if (!rawGraph || typeof rawGraph !== 'object') return []

  const { nodes = [], edges = [], links = [] } = rawGraph
  const edgeList = edges.length > 0 ? edges : links

  const elements = []

  // NODES
  nodes.forEach((node, idx) => {
    const nodeId = node.id ?? `node_${idx}`
    const kind = node.kind ?? 'unknown'

    let label = node.label ?? node.name ?? ''
    if (!label) {
      const parts = String(nodeId).split('::')
      label = parts[parts.length - 1] || nodeId
    }
    if (label.length > 22) label = label.slice(0, 19) + '...'

    const filePath = node.path ?? node.file ?? ''
    const pagerank = typeof node.pagerank === 'number' ? node.pagerank : null
    const commits = typeof node.commit_count === 'number' ? node.commit_count : null
    const isDead = !!node.is_dead_code_candidate
    const level = node.level ?? null

    // Compound node parent: level-3 nodes (functions/classes) render inside
    // their file container when parent_file is present. This enables the
    // fcose compound bounding box layout where symbols live inside file boxes.
    const parentFile = node.parent_file ?? null

    const descParts = []
    if (kind) descParts.push(kind)
    if (filePath) descParts.push(filePath)
    if (pagerank !== null) descParts.push('PR: ' + pagerank.toExponential(2))
    if (commits !== null) descParts.push('commits: ' + commits)
    if (isDead) descParts.push('dead-code candidate')

    const nodeData = {
      data: {
        id: String(nodeId),
        label,
        type: kind,
        level,
        desc: descParts.join(' - ') || nodeId,
        pagerank,
        commit_count: commits,
        is_dead_code: isDead,
        file: filePath,
      },
    }

    // Only set parent when parent_file resolves to a different node id.
    // This prevents self-parenting on file nodes themselves.
    if (parentFile && String(parentFile) !== String(nodeId)) {
      nodeData.data.parent = String(parentFile)
    }

    elements.push(nodeData)
  })

  // EDGES
  const seenEdgeIds = new Set()
  edgeList.forEach((edge) => {
    const src = String(edge.source ?? '')
    const tgt = String(edge.target ?? '')
    if (!src || !tgt) return

    const baseId = 'e_' + src + '__' + tgt
    let edgeId = baseId
    let counter = 0
    while (seenEdgeIds.has(edgeId)) {
      counter++
      edgeId = baseId + '_' + counter
    }
    seenEdgeIds.add(edgeId)

    elements.push({
      data: {
        id: edgeId,
        source: src,
        target: tgt,
        label: edge.rel ?? '',
        edge_type: edge.edge_type ?? '',
      },
    })
  })

  return elements
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

const COSE_LAYOUT = {
  name: 'cose',
  nodeRepulsion: 4500,
  idealEdgeLength: 80,
  gravity: 0.4,
  numIter: 1000,
  animate: false,
}

// ---------------------------------------------------------------------------
// Cytoscape stylesheet
// ---------------------------------------------------------------------------

const CYTOSCAPE_STYLES = [
  {
    selector: 'node',
    style: {
      'label': 'data(label)',
      'color': '#f8fafc',
      'font-family': 'Inter, system-ui, sans-serif',
      'font-size': '10px',
      'font-weight': 600,
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '72px',
      'background-color': '#0f172a',
      'border-width': 2,
      'border-color': '#475569',
      'width': 62,
      'height': 62,
      'shape': 'round-rectangle',
      'border-opacity': 0.95,
      'background-opacity': 0.95,
      'transition-property': 'background-color, border-color, width, height, border-width',
      'transition-duration': '0.2s',
    },
  },
  { selector: 'node[type = "file"]', style: { 'border-color': '#38bdf8', 'background-color': '#0c4a6e' } },
  { selector: 'node[type = "function"]', style: { 'border-color': '#34d399', 'background-color': '#064e3b' } },
  { selector: 'node[type = "class"]', style: { 'border-color': '#f59e0b', 'background-color': '#78350f' } },
  { selector: 'node[type = "import"]', style: { 'border-color': '#a78bfa', 'background-color': '#3b0764' } },
  { selector: 'node[type = "call_target"]', style: { 'border-color': '#fb923c', 'background-color': '#431407' } },
  { selector: 'node[?is_dead_code]', style: { 'border-color': '#f87171', 'border-width': 3 } },
  { selector: 'node:selected', style: { 'border-color': '#ffffff', 'border-width': 3.5 } },
  {
    selector: 'edge',
    style: {
      'width': 1.5,
      'line-color': '#334155',
      'target-arrow-color': '#64748b',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'arrow-scale': 1.0,
      'opacity': 0.7,
      'label': 'data(label)',
      'font-size': '9px',
      'font-family': 'monospace',
      'color': '#94a3b8',
      'text-rotation': 'autorotate',
      'text-margin-y': -7,
      'text-background-color': '#090d16',
      'text-background-opacity': 0.85,
      'text-background-padding': '2px',
      'text-background-shape': 'round-rectangle',
    },
  },
  { selector: 'edge:selected', style: { 'width': 3, 'line-color': '#38bdf8', 'target-arrow-color': '#38bdf8', 'opacity': 1 } },
]

// Kind -> colour for the legend
const KIND_COLORS = {
  file: '#38bdf8',
  function: '#34d399',
  class: '#f59e0b',
  import: '#a78bfa',
  call_target: '#fb923c',
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [elements, setElements] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState(null)
  const [graphMeta, setGraphMeta] = useState({ nodeCount: 0, edgeCount: 0 })

  const [repoId, setRepoId] = useState('dj-database-url')
  const [repoInput, setRepoInput] = useState('dj-database-url')

  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'system',
      role: 'System',
      time: new Date().toTimeString().split(' ')[0],
      text: 'CodeBase Visualizer ready. Enter a repo ID above and click Load to fetch the live graph from Redis.',
    },
  ])
  const [inputVal, setInputVal] = useState('')
  const [zoom, setZoom] = useState(100)
  const [panCoord, setPanCoord] = useState({ x: 0, y: 0 })
  const [selectedElement, setSelectedElement] = useState(null)

  // Parse modal state
  const [parseModalOpen, setParseModalOpen] = useState(false)
  const [parseUrl, setParseUrl] = useState('')
  const [parseStatus, setParseStatus] = useState(null)   // null | 'loading' | 'ok' | 'error'
  const [parseError, setParseError] = useState('')

  // Level 4 Trace mode state
  const [traceMode, setTraceMode] = useState(false)
  const [traceLoading, setTraceLoading] = useState(false)

  // Semantic Zoom Level state
  const [viewLevel, setViewLevel] = useState(2)

  const cyRef = useRef(null)
  const containerRef = useRef(null)

  // -------------------------------------------------------------------------
  // Fetch graph
  // -------------------------------------------------------------------------
  const fetchGraph = useCallback(async (id) => {
    if (!id?.trim()) return
    setIsLoading(true)
    setErrorMessage(null)
    setSelectedElement(null)

    try {
      const res = await fetch('/api/v1/graph/' + encodeURIComponent(id.trim()))

      if (res.status === 404) {
        setErrorMessage(
          'No cached graph found for "' + id + '". ' +
          'Click "Parse Repo" to analyze it first, or run POST /api/v1/graph/parse.'
        )
        setElements([])
        setGraphMeta({ nodeCount: 0, edgeCount: 0 })
        return
      }

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setErrorMessage('Server error ' + res.status + ': ' + (body?.detail ?? res.statusText))
        setElements([])
        setGraphMeta({ nodeCount: 0, edgeCount: 0 })
        return
      }

      const data = await res.json()
      const rawGraph = data.graph ?? {}
      const cyElements = transformGraphToCytoscape(rawGraph)

      const nodeCount = (rawGraph.nodes ?? []).length
      const edgeCount = (rawGraph.edges ?? rawGraph.links ?? []).length

      setElements(cyElements)
      setGraphMeta({ nodeCount, edgeCount })
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: 'system',
          role: 'Graph Engine',
          time: new Date().toTimeString().split(' ')[0],
          text: 'Loaded "' + id + '": ' + nodeCount + ' nodes, ' + edgeCount + ' edges. Canvas updated.',
        },
      ])
    } catch (err) {
      setErrorMessage('Network error: ' + err.message + '. Is the backend running on port 8001?')
      setElements([])
      setGraphMeta({ nodeCount: 0, edgeCount: 0 })
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => { fetchGraph(repoId) }, [repoId, fetchGraph])

  useEffect(() => {
    if (cyRef.current && elements.length > 0) {
      setTimeout(() => {
        cyRef.current?.fit(null, 45)
        setZoom(Math.round((cyRef.current?.zoom() ?? 1) * 100))
      }, 200)
    }
  }, [elements])

  // -------------------------------------------------------------------------
  // Cytoscape lifecycle
  // -------------------------------------------------------------------------
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
        file: node.data('file'),
        pagerank: node.data('pagerank'),
        commit_count: node.data('commit_count'),
        is_dead_code: node.data('is_dead_code'),
        degree: node.degree(),
        indegree: node.indegree(),
        outdegree: node.outdegree(),
      })
    })
    cy.on('unselect', 'node', () => setSelectedElement(null))
    cy.on('tap', (e) => { if (e.target === cy) setSelectedElement(null) })
    cy.once('layoutstop', updateViewportStats)
    cy.ready(updateViewportStats)
  }, [])

  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(() => cyRef.current?.resize())
    obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  // -------------------------------------------------------------------------
  // Canvas controls
  // -------------------------------------------------------------------------
  const handleZoomIn = () => {
    if (!cyRef.current) return
    const cy = cyRef.current
    cy.zoom({ level: cy.zoom() * 1.25, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } })
    setZoom(Math.round(cy.zoom() * 100))
  }

  const handleZoomOut = () => {
    if (!cyRef.current) return
    const cy = cyRef.current
    cy.zoom({ level: cy.zoom() * 0.8, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } })
    setZoom(Math.round(cy.zoom() * 100))
  }

  const handleFit = () => {
    if (!cyRef.current) return
    cyRef.current.fit(null, 45)
    setZoom(Math.round(cyRef.current.zoom() * 100))
  }

  const handleResetLayout = () => {
    if (!cyRef.current) return
    cyRef.current.layout(FCOSE_LAYOUT).run()
  }

  // -------------------------------------------------------------------------
  // Level 4 Trace
  // -------------------------------------------------------------------------
  const handleTrace = async (nodeId) => {
    if (!cyRef.current || !nodeId) return
    setTraceLoading(true)
    try {
      await activateLevel4Trace(nodeId, repoId, cyRef.current)
      setTraceMode(true)
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: 'system',
          role: 'Trace Engine',
          time: new Date().toTimeString().split(' ')[0],
          text: `Level 4 Trace activated for "${nodeId}". Showing 2-hop ego-graph with call_target nodes. Click ✕ Exit Trace to restore the full graph.`,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: 'system',
          role: 'Trace Engine',
          time: new Date().toTimeString().split(' ')[0],
          text: `Trace failed: ${err.message}`,
        },
      ])
    } finally {
      setTraceLoading(false)
    }
  }

  const handleExitTrace = () => {
    exitLevel4Trace(cyRef.current)
    setTraceMode(false)
  }

  // -------------------------------------------------------------------------
  // Chat
  // -------------------------------------------------------------------------
  const handleSendMessage = (e) => {
    e?.preventDefault()
    if (!inputVal.trim()) return
    const now = new Date().toTimeString().split(' ')[0]
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), sender: 'user', role: 'Engineer', time: now, text: inputVal.trim() },
      {
        id: Date.now() + 1, sender: 'assistant', role: 'CodeBase Copilot', time: now,
        text: 'Graph "' + repoId + '" has ' + graphMeta.nodeCount + ' nodes and ' + graphMeta.edgeCount + ' edges. Enter a repo ID in the header to switch repos.',
      },
    ])
    setInputVal('')
  }

  const handleLoadRepo = (e) => {
    e?.preventDefault()
    const id = repoInput.trim()
    if (id && id !== repoId) setRepoId(id)
    else if (id === repoId) fetchGraph(id)
  }

  const handleParseRepo = () => {
    setParseUrl('')
    setParseStatus(null)
    setParseError('')
    setParseModalOpen(true)
  }

  const handleParseSubmit = async (e) => {
    e?.preventDefault()
    let url = parseUrl.trim()
    if (!url) return

    // Auto-prefix https:// if omitted
    if (!/^https?:\/\//i.test(url)) {
      url = 'https://' + url
    }

    // Derive a clean repo_id from the GitHub URL:
    //   https://github.com/owner/repo  →  owner-repo
    //   https://github.com/owner/repo.git  →  owner-repo
    let derivedRepoId = repoId
    try {
      const u = new URL(url)
      const parts = u.pathname.replace(/^\/+|\/+$/g, '').replace(/\.git$/i, '').split('/')
      if (parts.length >= 2 && parts[0] && parts[1]) {
        derivedRepoId = parts[0] + '-' + parts[1]
      }
    } catch { /* keep existing repoId if URL is malformed */ }

    setParseStatus('loading')
    setParseError('')

    try {
      const res = await fetch('/api/v1/graph/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: url, repo_id: derivedRepoId }),
      })
      let data = {}
      try {
        data = await res.json()
      } catch {
        data = { detail: await res.text() }
      }

      if (res.ok) {
        setParseStatus('ok')
        const m = data.metrics ?? {}
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now(),
            sender: 'system',
            role: 'Parse Engine',
            time: new Date().toTimeString().split(' ')[0],
            text:
              'Parse complete for "' + derivedRepoId + '": ' +
              (m.node_count ?? 0) + ' nodes, ' + (m.edge_count ?? 0) + ' edges cached. ' +
              '(' + (m.file_count ?? 0) + ' files, ' + (m.function_count ?? 0) + ' functions)',
          },
        ])
        // Switch to the new repo and load its graph
        setRepoInput(derivedRepoId)
        setRepoId(derivedRepoId)
        fetchGraph(derivedRepoId)
        setTimeout(() => setParseModalOpen(false), 1200)
      } else {
        setParseStatus('error')
        setParseError(data.detail ?? 'Server returned ' + res.status)
      }
    } catch (err) {
      setParseStatus('error')
      setParseError('Network error: ' + err.message + '. Is Docker / backend running on port 8001?')
    }
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="h-screen w-full flex flex-col md:flex-row overflow-hidden bg-slate-950 text-slate-100 font-sans select-none">

      {/* ================================================================== */}
      {/* LEFT PANEL: Canvas Map                                              */}
      {/* ================================================================== */}
      <section
        id="canvas-map-panel"
        className="w-full md:w-[60%] md:basis-[60%] h-1/2 md:h-full flex flex-col shrink-0 border-b md:border-b-0 md:border-r border-slate-800/90 bg-slate-950 relative overflow-hidden"
      >
        {/* Top Bar */}
        <header className="h-14 px-4 sm:px-6 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur flex items-center justify-between shrink-0 z-10 gap-2">
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 flex items-center justify-center font-bold text-sm shadow-sm shadow-cyan-500/20">
              ⚡
            </div>
            <div className="hidden sm:block">
              <h1 className="text-sm font-semibold tracking-tight text-white">Canvas Map</h1>
              <p className="text-[11px] text-slate-400">AST Dependency Graph Visualizer</p>
            </div>
          </div>

          {/* Semantic Zoom Toolbar */}
          <div className="hidden md:flex items-center bg-slate-900 border border-slate-700/60 rounded-lg p-0.5 text-[10px] font-mono shrink-0 shadow-inner">
            <button
              type="button"
              onClick={() => setViewLevel(1)}
              className={`px-3 py-1 rounded-md transition ${viewLevel === 1 ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'}`}
            >
              Level 1: Arch
            </button>
            <button
              type="button"
              onClick={() => setViewLevel(2)}
              className={`px-3 py-1 rounded-md transition ${viewLevel === 2 ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'}`}
            >
              Level 2: Files
            </button>
            <button
              type="button"
              onClick={() => setViewLevel(3)}
              className={`px-3 py-1 rounded-md transition ${viewLevel === 3 ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'}`}
            >
              Level 3: Functions
            </button>
          </div>

          {/* Repo ID input + Parse button */}
          <form onSubmit={handleLoadRepo} className="flex items-center gap-1.5 flex-1 max-w-sm mx-2">
            <input
              type="text"
              id="repo-id-input"
              value={repoInput}
              onChange={(e) => setRepoInput(e.target.value)}
              placeholder="repo-id (e.g. dj-database-url)"
              className="flex-1 bg-slate-900 border border-slate-700 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/50 rounded-lg px-2.5 py-1 text-[11px] font-mono text-slate-100 placeholder-slate-500 outline-none transition"
            />
            <button
              type="submit"
              id="load-repo-btn"
              disabled={isLoading}
              className="px-2.5 py-1 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 disabled:opacity-50 text-cyan-300 border border-cyan-500/30 text-[11px] font-mono font-medium transition cursor-pointer shrink-0"
            >
              {isLoading ? '...' : 'Load'}
            </button>
            <button
              type="button"
              id="parse-repo-btn"
              onClick={handleParseRepo}
              className="px-2.5 py-1 rounded-lg bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 border border-indigo-500/30 text-[11px] font-mono font-medium transition cursor-pointer shrink-0"
            >
              Parse
            </button>
          </form>

          {/* Canvas controls */}
          <div className="hidden sm:flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-0.5 shrink-0">
            <button type="button" id="canvas-zoom-out-btn" onClick={handleZoomOut}
              className="w-7 h-7 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 rounded text-xs transition cursor-pointer" title="Zoom Out">-</button>
            <span id="canvas-zoom-level" className="text-[11px] font-mono px-1.5 text-slate-300 min-w-[44px] text-center">{zoom}%</span>
            <button type="button" id="canvas-zoom-in-btn" onClick={handleZoomIn}
              className="w-7 h-7 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 rounded text-xs transition cursor-pointer" title="Zoom In">+</button>
            <button type="button" id="canvas-fit-btn" onClick={handleFit}
              className="px-2 h-7 flex items-center justify-center text-slate-300 hover:text-white hover:bg-slate-800 rounded text-[10px] font-mono transition cursor-pointer" title="Fit">Fit</button>
            <button type="button" id="canvas-relayout-btn" onClick={handleResetLayout}
              className="px-2 h-7 flex items-center justify-center text-cyan-400 hover:text-cyan-300 hover:bg-cyan-950/40 rounded text-[10px] font-mono transition cursor-pointer border border-cyan-800/40" title="fCoSE Layout">fCoSE</button>
          </div>
        </header>

        {/* Canvas Area */}
        <div ref={containerRef} id="cytoscape-canvas-container"
          className="flex-1 relative w-full h-full min-h-0 overflow-hidden bg-slate-950/60">

          {/* Grid background */}
          <div className="absolute inset-0 opacity-10 pointer-events-none"
            style={{ backgroundImage: 'radial-gradient(#38bdf8 1px, transparent 1px)', backgroundSize: '32px 32px' }} />

          {/* LOADING OVERLAY */}
          {isLoading && (
            <div id="loading-overlay"
              className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-sm gap-4">
              <div className="w-12 h-12 rounded-full border-4 border-cyan-400/30 border-t-cyan-400 animate-spin" />
              <p className="text-sm font-mono text-cyan-300">
                Fetching graph for <span className="font-bold">{repoId}</span>...
              </p>
            </div>
          )}

          {/* ERROR OVERLAY */}
          {!isLoading && errorMessage && (
            <div id="error-overlay"
              className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/85 backdrop-blur-sm p-6">
              <div className="max-w-md w-full rounded-2xl bg-slate-900 border border-red-500/40 shadow-2xl shadow-red-950/40 p-6 flex flex-col gap-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-9 h-9 rounded-lg bg-red-500/15 border border-red-500/30 flex items-center justify-center text-red-400 text-lg shrink-0">
                    ⚠
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-red-300">Graph Not Found</p>
                    <p className="text-[11px] text-slate-400 font-mono">Redis key missing or server error</p>
                  </div>
                </div>
                <p className="text-[12px] text-slate-300 leading-relaxed">{errorMessage}</p>
                <div className="flex gap-2">
                  <button type="button" id="error-retry-btn" onClick={() => fetchGraph(repoId)}
                    className="flex-1 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/30 text-xs font-mono font-medium transition cursor-pointer">
                    Retry
                  </button>
                  <button type="button" id="error-parse-btn" onClick={handleParseRepo}
                    className="flex-1 py-1.5 rounded-lg bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 border border-indigo-500/30 text-xs font-mono font-medium transition cursor-pointer">
                    Parse Repo
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Cytoscape graph */}
          {!isLoading && !errorMessage && (
            <GraphCanvas
              elements={elements}
              viewLevel={viewLevel}
              onSelectElement={setSelectedElement}
              onCyReady={handleCy}
            />
          )}

          {/* Status badge + legend */}
          {!isLoading && !errorMessage && elements.length > 0 && (
            <div className="absolute top-3 left-3 z-10 pointer-events-none flex flex-col gap-1.5">
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-950/85 border border-cyan-500/30 backdrop-blur text-[11px] font-mono text-cyan-300 shadow-lg">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                <span>fCoSE · {repoId}</span>
              </div>
              <div className="flex flex-wrap gap-1.5 px-2.5 py-1.5 rounded-md bg-slate-950/85 border border-slate-700/50 backdrop-blur max-w-xs">
                {Object.entries(KIND_COLORS).map(([kind, color]) => (
                  <span key={kind} className="flex items-center gap-1 text-[10px] font-mono text-slate-300">
                    <span className="w-2 h-2 rounded-sm inline-block shrink-0" style={{ backgroundColor: color }} />
                    {kind}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Node inspector */}
          {selectedElement && (
            <div id="node-inspector-card"
              className="absolute bottom-10 left-3 z-10 p-3.5 rounded-xl bg-slate-950/95 border border-cyan-500/50 backdrop-blur-md text-xs shadow-xl shadow-cyan-950/50 max-w-xs">
              <div className="flex items-center justify-between gap-3 mb-2">
                <span className="font-bold text-white text-sm truncate max-w-[160px]" title={selectedElement.label}>{selectedElement.label}</span>
                <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono text-[10px] uppercase border border-cyan-500/30 shrink-0">
                  {selectedElement.category}
                </span>
              </div>
              {selectedElement.file && (
                <p className="text-slate-400 font-mono text-[10px] mb-1.5 truncate" title={selectedElement.file}>{selectedElement.file}</p>
              )}
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] font-mono text-slate-400 border-t border-slate-800 pt-2">
                <span>Level: <span className="text-violet-400 font-semibold">{selectedElement.level ?? '—'}</span></span>
                <span>Degree: <span className="text-cyan-400 font-semibold">{selectedElement.degree}</span></span>
                <span>In/Out: <span className="text-slate-300">{selectedElement.indegree}/{selectedElement.outdegree}</span></span>
                {selectedElement.pagerank !== null && (
                  <span>PageRank: <span className="text-emerald-400">{selectedElement.pagerank?.toExponential(2)}</span></span>
                )}
                {selectedElement.commit_count !== null && (
                  <span>Commits: <span className="text-amber-400">{selectedElement.commit_count}</span></span>
                )}
                {selectedElement.is_dead_code && (
                  <span className="col-span-2 text-red-400 font-semibold">Dead-code candidate</span>
                )}
              </div>
              <div className="mt-1.5 text-[10px] font-mono text-slate-500 truncate" title={selectedElement.id}>
                ID: {selectedElement.id}
              </div>
              {/* Level 4 Trace button — shown for function & class nodes */}
              {(selectedElement.category === 'function' || selectedElement.category === 'class') && (
                <button
                  id="level4-trace-btn"
                  onClick={() => handleTrace(selectedElement.id)}
                  disabled={traceLoading}
                  className="mt-2.5 w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600/80 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-wait text-white font-semibold text-[11px] transition active:scale-95 border border-violet-500/40 shadow-md shadow-violet-900/40 cursor-pointer"
                >
                  {traceLoading ? '⏳ Loading trace…' : '🔬 Level 4 Trace'}
                </button>
              )}
            </div>
          )}

          {/* Exit Trace floating button — visible only during trace mode */}
          {traceMode && (
            <button
              id="exit-trace-btn"
              onClick={handleExitTrace}
              className="absolute top-3 left-3 z-20 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-600/90 hover:bg-rose-500 text-white font-semibold text-[11px] transition active:scale-95 border border-rose-500/40 shadow-lg shadow-rose-900/50 cursor-pointer"
            >
              ✕ Exit Trace
            </button>
          )}

          {/* Nav tips */}
          <div className="absolute bottom-3 right-3 z-10 hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-950/70 border border-slate-800/80 backdrop-blur text-[10px] font-mono text-slate-400 pointer-events-none">
            <span>Pan: Drag BG - Zoom: Scroll - Move: Drag Node</span>
          </div>
        </div>

        {/* Footer */}
        <footer className="h-9 px-4 sm:px-6 bg-slate-950/90 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400 shrink-0">
          <div className="flex items-center gap-3">
            <span>Pan: {panCoord.x}, {panCoord.y}</span>
            <span className="hidden sm:inline text-slate-600">|</span>
            <span id="graph-stats-badge" className="text-cyan-300 font-semibold hidden sm:inline">
              {graphMeta.nodeCount} Nodes, {graphMeta.edgeCount} Edges
            </span>
          </div>
          <div className="flex items-center gap-2 text-cyan-400">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span>Cytoscape · {repoId}</span>
          </div>
        </footer>
      </section>

      {/* ================================================================== */}
      {/* RIGHT PANEL: Chat Drawer                                            */}
      {/* ================================================================== */}
      <aside id="chat-drawer-panel"
        className="w-full md:w-[40%] md:basis-[40%] h-1/2 md:h-full flex flex-col bg-slate-900/95 relative overflow-hidden">

        <header className="h-14 px-4 sm:px-6 border-b border-slate-800/90 bg-slate-950/90 backdrop-blur flex items-center justify-between shrink-0 z-10">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/15 border border-indigo-500/30 text-indigo-400 flex items-center justify-center font-bold text-sm">
              💬
            </div>
            <div>
              <h2 className="text-sm font-semibold tracking-tight text-white">Chat Drawer</h2>
              <p className="text-[11px] text-slate-400">Codebase Q&A · Architecture Reasoning</p>
            </div>
          </div>
          <span className="text-[10px] font-mono text-slate-500 px-2 py-0.5 rounded bg-slate-800/60 border border-slate-700/50">
            {graphMeta.nodeCount}N · {graphMeta.edgeCount}E
          </span>
        </header>

        <div id="chat-messages-scroll-area"
          className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-3.5 scroll-smooth">
          {messages.map((msg) => (
            <div key={msg.id}
              className={'rounded-xl p-3.5 border transition text-xs sm:text-sm leading-relaxed ' + (
                msg.sender === 'user'
                  ? 'bg-cyan-950/40 border-cyan-500/30 text-cyan-100 ml-4 sm:ml-8'
                  : msg.sender === 'system'
                  ? 'bg-slate-950/70 border-slate-800 text-slate-400 font-mono text-[11px]'
                  : 'bg-slate-950/90 border-slate-800/90 text-slate-200 mr-4 sm:mr-8 shadow-sm'
              )}>
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

        <footer className="p-3 sm:p-4 bg-slate-950/90 border-t border-slate-800/90 shrink-0">
          <form onSubmit={handleSendMessage} className="flex items-center gap-2">
            <input
              type="text"
              id="chat-input-field"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              placeholder="Ask about codebase architecture..."
              className="flex-1 bg-slate-900 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-3.5 py-2 text-xs sm:text-sm text-slate-100 placeholder-slate-500 outline-none transition"
            />
            <button type="submit" id="chat-send-btn"
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs sm:text-sm transition shadow-lg shadow-indigo-600/20 active:scale-95 cursor-pointer shrink-0">
              Send
            </button>
          </form>
          <div className="flex items-center justify-between mt-2 text-[10px] font-mono text-slate-400">
            <span>Enter to send</span>
            <span>{graphMeta.nodeCount} nodes loaded</span>
          </div>
        </footer>
      </aside>

      {/* ================================================================== */}
      {/* PARSE MODAL                                                        */}
      {/* ================================================================== */}
      {parseModalOpen && (
      <div
        id="parse-modal-backdrop"
        className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4"
        onClick={(e) => { if (e.target === e.currentTarget && parseStatus !== 'loading') setParseModalOpen(false) }}
      >
        <div
          id="parse-modal"
          className="w-full max-w-md rounded-2xl bg-slate-900 border border-indigo-500/40 shadow-2xl shadow-indigo-950/50 p-6 flex flex-col gap-4"
        >
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-lg bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400 text-lg shrink-0">
                🔍
              </div>
              <div>
                <p className="text-sm font-semibold text-white">Parse Repository</p>
                <p className="text-[11px] text-slate-400 font-mono">Clone · Scan · Cache graph in Redis</p>
              </div>
            </div>
            {parseStatus !== 'loading' && (
              <button
                type="button"
                id="parse-modal-close-btn"
                onClick={() => setParseModalOpen(false)}
                className="text-slate-500 hover:text-white text-lg leading-none cursor-pointer transition"
              >
                ✕
              </button>
            )}
          </div>

          {/* Form */}
          <form onSubmit={handleParseSubmit} className="flex flex-col gap-3">
            <div className="flex flex-col gap-1">
              <label htmlFor="parse-url-input" className="text-[11px] font-mono text-slate-400">
                GitHub Repository URL
              </label>
              <input
                id="parse-url-input"
                type="url"
                value={parseUrl}
                onChange={(e) => setParseUrl(e.target.value)}
                placeholder="https://github.com/owner/repo"
                disabled={parseStatus === 'loading'}
                autoFocus
                required
                className="bg-slate-800 border border-slate-700 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 rounded-lg px-3 py-2 text-sm font-mono text-slate-100 placeholder-slate-500 outline-none transition disabled:opacity-60"
              />
              <p className="text-[10px] font-mono text-slate-500">
                Repo ID will be auto-derived · e.g. owner-repo
              </p>
            </div>

            {/* Status feedback */}
            {parseStatus === 'loading' && (
              <div id="parse-status-loading" className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg bg-indigo-950/50 border border-indigo-500/30">
                <div className="w-4 h-4 rounded-full border-2 border-indigo-400/30 border-t-indigo-400 animate-spin shrink-0" />
                <p className="text-xs font-mono text-indigo-300">
                  Cloning &amp; scanning… this may take 20–60 seconds.
                </p>
              </div>
            )}
            {parseStatus === 'ok' && (
              <div id="parse-status-ok" className="flex items-center gap-2 px-3 py-2.5 rounded-lg bg-emerald-950/50 border border-emerald-500/30">
                <span className="text-emerald-400 text-base">✓</span>
                <p className="text-xs font-mono text-emerald-300">Graph cached! Loading canvas…</p>
              </div>
            )}
            {parseStatus === 'error' && (
              <div id="parse-status-error" className="flex items-start gap-2 px-3 py-2.5 rounded-lg bg-red-950/50 border border-red-500/30">
                <span className="text-red-400 text-base shrink-0">⚠</span>
                <p className="text-xs font-mono text-red-300 break-words">{parseError}</p>
              </div>
            )}

            <div className="flex gap-2 pt-1">
              <button
                type="button"
                id="parse-modal-cancel-btn"
                onClick={() => setParseModalOpen(false)}
                disabled={parseStatus === 'loading'}
                className="flex-1 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 border border-slate-700 text-xs font-mono transition cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="submit"
                id="parse-modal-submit-btn"
                disabled={parseStatus === 'loading' || parseStatus === 'ok'}
                className="flex-1 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white border border-indigo-500 text-xs font-mono font-medium transition cursor-pointer shadow-lg shadow-indigo-600/20"
              >
                {parseStatus === 'loading' ? 'Parsing…' : 'Parse & Load'}
              </button>
            </div>
          </form>
        </div>
      </div>
    )}
    </div>
  )
}

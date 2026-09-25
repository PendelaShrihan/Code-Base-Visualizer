import { useEffect, useRef, useCallback } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
import fcose from 'cytoscape-fcose'

// ---------------------------------------------------------------------------
// Register extensions once with Cytoscape
// ---------------------------------------------------------------------------
cytoscape.use(fcose)

// ---------------------------------------------------------------------------
// Layout Configuration
// ---------------------------------------------------------------------------
export const FCOSE_LAYOUT = {
  name: 'fcose',
  quality: 'proof',
  animate: false,
  // ── Core physics ────────────────────────────────────────────────────────
  nodeRepulsion: 15000,
  idealEdgeLength: 150,
  edgeElasticity: 0.45,
  gravity: 0.15,
  nestingFactor: 0.1,
  // ── Packing & Component 2D Grid ──────────────────────────────────────────
  // packComponents requires cytoscape-layout-utilities registered on the cy
  // instance — done lazily in handleCy to avoid ESM module-load crashes.
  tile: true,
  packComponents: true,
  tilingPaddingVertical: 20,
  tilingPaddingHorizontal: 20,
  nodeSeparation: 75,
  // ── Fitting ─────────────────────────────────────────────────────────────
  fit: true,
  padding: 40,
  // randomize: true seeds node positions randomly so disconnected components
  // are NOT placed on a diagonal grid. Without this, fcose initialises all
  // nodes on a deterministic diagonal and physics merely pushes them further
  // apart along that same axis — producing the "rigid diagonal line" artifact.
  randomize: true,
  // ── Compound node bounding boxes ─────────────────────────────────────────
  initialEnergyOnIncremental: 0.5,
  interClusterEdgeLengthFactor: 0.95,
}


// ---------------------------------------------------------------------------
// Cytoscape Stylesheet with Dimming & 1-Hop Highlight Support
// ---------------------------------------------------------------------------
export const CYTOSCAPE_STYLES = [
  // Base node styling
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
      'width': 58,
      'height': 58,
      'shape': 'round-rectangle',
      'border-opacity': 0.95,
      'background-opacity': 0.95,
      'transition-property': 'opacity, border-color, width, height, border-width',
      'transition-duration': '0.15s',
    },
  },
  // Color code by AST node type
  { selector: 'node[type = "file"]',        style: { 'border-color': '#38bdf8', 'background-color': '#0c4a6e' } },
  { selector: 'node[type = "function"]',    style: { 'border-color': '#34d399', 'background-color': '#064e3b' } },
  { selector: 'node[type = "class"]',       style: { 'border-color': '#f59e0b', 'background-color': '#78350f' } },
  { selector: 'node[type = "import"]',      style: { 'border-color': '#a78bfa', 'background-color': '#3b0764' } },
  { selector: 'node[type = "call_target"]', style: { 'border-color': '#fb923c', 'background-color': '#431407' } },
  { selector: 'node[?is_dead_code]',        style: { 'border-color': '#f87171', 'border-width': 3 } },

  // ── Compound (parent) container for file nodes ───────────────────────────
  // When a file node has function/class children it becomes a compound node.
  // Rendered as a translucent dashed bounding box that frames its children.
  {
    selector: 'node:parent',
    style: {
      'label': 'data(label)',
      'text-valign': 'top',
      'text-halign': 'center',
      'font-size': '9px',
      'font-weight': 700,
      'color': '#7dd3fc',
      'background-color': '#0c2a40',
      'background-opacity': 0.45,
      'border-color': '#38bdf8',
      'border-width': 1.5,
      'border-opacity': 0.6,
      'border-style': 'dashed',
      // Padding gives children breathing room inside the container
      'padding': '18px',
      'shape': 'round-rectangle',
    },
  },

  // ── Level 1 Folder Nodes ─────────────────────────────────────────────────
  // Empty folders shouldn't disappear, so give them explicit dimensions.
  {
    selector: 'node[level = 1], node[level = "1"]',
    style: {
      'label': 'data(id)',
      'min-width': '100px',
      'min-height': '100px',
      'background-color': '#1e293b',
      'border-width': '2px',
      'shape': 'round-rectangle',
    },
  },

  // Base edge styling: subtle by default with clear directional arrowheads
  {
    selector: 'edge',
    style: {
      'curve-style': 'bezier',
      'width': 1.5,
      'opacity': 0.4,
      'line-color': '#64748b',
      'target-arrow-shape': 'triangle',
      'target-arrow-color': '#64748b',
      'arrow-scale': 1.0,
      'transition-property': 'opacity, line-color, target-arrow-color, width',
      'transition-duration': '0.15s',
    },
  },

  // ── 1-Hop Focus: Dimming ──────────────────────────────────────────────────
  {
    selector: 'node.dimmed',
    style: {
      'opacity': 0.12,
      'border-opacity': 0.08,
      'text-opacity': 0,
    },
  },
  {
    selector: 'edge.dimmed',
    style: {
      'opacity': 0.04,
      'text-opacity': 0,
    },
  },
  // Neighbor nodes: bright sky-blue ring
  {
    selector: 'node.highlighted',
    style: {
      'opacity': 1,
      'border-width': 3,
      'border-color': '#38bdf8',
      'shadow-blur': 16,
      'shadow-color': '#38bdf8',
      'shadow-opacity': 0.75,
      'z-index': 999,
    },
  },
  // The tapped (focal) node itself: brighter white ring + stronger glow
  {
    selector: 'node.highlighted-focal',
    style: {
      'opacity': 1,
      'border-width': 4,
      'border-color': '#f0f9ff',
      'shadow-blur': 24,
      'shadow-color': '#7dd3fc',
      'shadow-opacity': 0.9,
      'z-index': 1000,
    },
  },
  {
    selector: 'edge.highlighted',
    style: {
      'opacity': 1,
      'width': 2.5,
      'line-color': '#38bdf8',
      'target-arrow-color': '#38bdf8',
      'color': '#38bdf8',
      'text-opacity': 1,
      'z-index': 998,
    },
  },

  // ── Progressive Function Visibility ──────────────────────────────────────
  // Level 3 function/symbol nodes stay hidden inside parent file boxes on initial load
  {
    selector: 'node[level = 3], node[level = "3"]',
    style: {
      'display': 'none',
    },
  },
  // When highlighted, focused, or tapped, reveal Level 3 symbols
  {
    selector: 'node[level = 3].highlighted, node[level = 3].highlighted-focal, node[level = "3"].highlighted, node[level = "3"].highlighted-focal',
    style: {
      'display': 'element',
    },
  },

  // ── Chat-answer node highlight ────────────────────────────────────────────
  // Applied imperatively by the highlightedNodeIds useEffect in GraphCanvas.
  // Uses pink/magenta so it is visually distinct from the cyan 1-hop tap
  // highlight and the white focal ring — the two can coexist without conflict.
  {
    selector: 'node.chat-highlight',
    style: {
      'opacity': 1,
      'border-width': 4,
      'border-color': '#f472b6',
      'background-color': '#4a044e',
      'shadow-blur': 20,
      'shadow-color': '#f472b6',
      'shadow-opacity': 0.85,
      'z-index': 1100,
      'transition-property': 'border-color, background-color, opacity, border-width',
      'transition-duration': '0.25s',
    },
  },
]

// ---------------------------------------------------------------------------
// GraphCanvas Component
// ---------------------------------------------------------------------------
export default function GraphCanvas({
  elements = [],
  layout = FCOSE_LAYOUT,
  stylesheet = CYTOSCAPE_STYLES,
  onSelectElement,
  onCyReady,
  viewLevel = 2,
  highlightedNodeIds = [],
}) {
  const cyRef = useRef(null)
  const containerRef = useRef(null)

  const handleCy = useCallback((cy) => {
    if (cyRef.current === cy) return
    cyRef.current = cy

    // -----------------------------------------------------------------------
    // Register cytoscape-layout-utilities lazily on the live cy instance.
    //
    // We cannot call cytoscape.use(layoutUtilities) at module-top because the
    // package accesses browser globals (window/document) during ES module
    // evaluation, which throws before React mounts in Vite's ESM pipeline.
    //
    // Importing + registering inside this callback is safe: handleCy only
    // runs after the CytoscapeComponent has mounted in the browser.
    // The _luRegistered guard prevents double-registration across re-renders.
    // -----------------------------------------------------------------------
    if (!cy._luRegistered) {
      import('cytoscape-layout-utilities').then((mod) => {
        const layoutUtilities = mod.default ?? mod
        try {
          cytoscape.use(layoutUtilities)
        } catch {
          // Already registered — safe to ignore
        }
        cy._luRegistered = true
      }).catch(() => {
        // Package unavailable — packComponents will silently fall back to
        // fcose's built-in tiling which is still better than diagonal.
      })
    }

    if (onCyReady) {
      onCyReady(cy)
    }

    // -----------------------------------------------------------------------
    // Node tap: highlight 1-hop neighborhood & dim all other elements
    //
    // neighborhood() returns: the clicked node + all directly connected nodes
    // and the edges between them — exactly the 1-hop caller/callee set.
    //
    // Compound parents (file containers) are added to the focus set so that
    // file container boxes are never accidentally dimmed when one of their
    // function children is the tapped node.
    //
    // Direct children (Level 3 functions) are added when tapping a file
    // container so its inner functions expand into view.
    // -----------------------------------------------------------------------
    cy.on('tap', 'node', (evt) => {
      const node = evt.target

      // 1-hop neighborhood: direct callers + callees + connecting edges + child symbols
      const children = node.children()
      const neighborhood = node.neighborhood().add(node).add(children)

      // Include compound parents so file containers remain visible
      const compoundParents = neighborhood.nodes().parents()
      const focusSet = neighborhood.union(compoundParents)

      cy.elements().removeClass('highlighted highlighted-focal dimmed')
      cy.elements().difference(focusSet).addClass('dimmed')
      focusSet.addClass('highlighted')

      // Tapped node gets a distinctive brighter "focal" white ring on top
      node.removeClass('highlighted').addClass('highlighted-focal')

      if (onSelectElement) {
        onSelectElement({
          type: 'node',
          id: node.id(),
          label: node.data('label'),
          category: node.data('type'),
          level: node.data('level'),
          desc: node.data('desc'),
          file: node.data('file'),
          pagerank: node.data('pagerank'),
          commit_count: node.data('commit_count'),
          is_dead_code: node.data('is_dead_code'),
          degree: node.degree(),
          indegree: node.indegree(),
          outdegree: node.outdegree(),
        })
      }
    })

    // -----------------------------------------------------------------------
    // Canvas background click: clear all dimming/highlighting focus
    // -----------------------------------------------------------------------
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass('highlighted highlighted-focal dimmed')
        if (onSelectElement) {
          onSelectElement(null)
        }
      }
    })
  }, [onSelectElement, onCyReady])

  // Automatically resize on container dimension change
  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(() => cyRef.current?.resize())
    obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  // Auto-fit on elements change
  useEffect(() => {
    if (cyRef.current && elements.length > 0) {
      setTimeout(() => {
        cyRef.current?.fit(null, 45)
      }, 150)
    }
  }, [elements])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

    if (viewLevel === 1) {
      cy.nodes('[level > 1]').style('display', 'none')
      cy.nodes('[level = 1]').style('display', 'element')
    } else if (viewLevel === 2) {
      cy.nodes('[level > 2]').style('display', 'none')
      cy.nodes('[level <= 2]').style('display', 'element')
    } else if (viewLevel === 3) {
      cy.nodes('[level <= 3]').style('display', 'element')
    }
  }, [viewLevel])

  // ---------------------------------------------------------------------------
  // Chat-answer highlight: apply/clear the .chat-highlight class imperatively.
  //
  // We do NOT touch dimming or the 1-hop highlight — those are driven by tap
  // events inside handleCy and are fully orthogonal to this feature.
  //
  // Algorithm:
  //   1. Remove .chat-highlight from every node (handles the "new query" clear).
  //   2. For each ID in highlightedNodeIds, look up the node and add the class.
  //      cy.getElementById returns an empty collection (length 0) when the ID
  //      is not present — safe to call addClass on it without a guard.
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.nodes().removeClass('chat-highlight')
    highlightedNodeIds.forEach((id) => {
      cy.getElementById(id).addClass('chat-highlight')
    })
  }, [highlightedNodeIds])

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <CytoscapeComponent
        id="cytoscape-graph"
        elements={elements}
        layout={layout}
        stylesheet={stylesheet}
        cy={handleCy}
        className="w-full h-full"
        wheelSensitivity={0.25}
        boxSelectionEnabled={false}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Level 4 Trace Utilities (exported for use in App.jsx)
// ---------------------------------------------------------------------------

/**
 * Transform a node-link JSON payload (from nx.node_link_data) into
 * Cytoscape element descriptors ready for cy.add().
 * Deduplicates against elements already present in cy.
 */
function _tracePayloadToCyElements(rawGraph, cy) {
  const { nodes = [], edges = [], links = [] } = rawGraph
  const edgeList = edges.length > 0 ? edges : links
  const toAdd = []
  const seenEdges = new Set()

  nodes.forEach((node, idx) => {
    const nodeId = String(node.id ?? `trace_node_${idx}`)
    // Skip if already in canvas
    if (cy.getElementById(nodeId).length > 0) return

    const kind = node.kind ?? 'unknown'
    let label = node.label ?? node.name ?? ''
    if (!label) {
      const parts = nodeId.split('::')
      label = parts[parts.length - 1] || nodeId
    }
    if (label.length > 22) label = label.slice(0, 19) + '...'

    const parentFile = node.parent_file ?? null

    const nodeData = {
      group: 'nodes',
      data: {
        id: nodeId,
        label,
        type: kind,
        level: node.level ?? 4,
        desc: [kind, node.path ?? node.file ?? ''].filter(Boolean).join(' - '),
        pagerank: typeof node.pagerank === 'number' ? node.pagerank : null,
        commit_count: typeof node.commit_count === 'number' ? node.commit_count : null,
        is_dead_code: !!node.is_dead_code_candidate,
        file: node.path ?? node.file ?? '',
      },
    }
    if (parentFile && String(parentFile) !== nodeId) {
      nodeData.data.parent = String(parentFile)
    }
    toAdd.push(nodeData)
  })

  edgeList.forEach((edge) => {
    const src = String(edge.source ?? '')
    const tgt = String(edge.target ?? '')
    if (!src || !tgt) return
    const edgeId = `e_trace_${src}__${tgt}`
    if (seenEdges.has(edgeId)) return
    seenEdges.add(edgeId)
    toAdd.push({
      group: 'edges',
      data: {
        id: edgeId,
        source: src,
        target: tgt,
        label: edge.rel ?? '',
        edge_type: edge.edge_type ?? 'TRACE',
      },
    })
  })

  return toAdd
}

/**
 * activateLevel4Trace(nodeId, repoId, cy)
 *
 * 1. Fetches GET /api/v1/graph/{repoId}/trace/{nodeId}
 * 2. Injects new Level 4 nodes/edges into the live canvas
 * 3. Blacks out all other elements
 * 4. Reveals only the trace subgraph + compound parents
 * 5. Runs an animated fcose layout on the visible set
 *
 * Returns the set of trace node IDs added/revealed (useful for UI state).
 * Throws on network / API errors.
 */
export async function activateLevel4Trace(nodeId, repoId, cy) {
  if (!cy || !nodeId || !repoId) return

  const url = `/api/v1/graph/${encodeURIComponent(repoId)}/trace/${encodeURIComponent(nodeId)}`
  const res = await fetch(url)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(`Trace fetch failed (${res.status}): ${body?.detail ?? res.statusText}`)
  }
  const data = await res.json()
  const rawGraph = data.graph ?? {}

  // Build elements to inject (nodes not already in the canvas)
  const newElements = _tracePayloadToCyElements(rawGraph, cy)
  if (newElements.length > 0) {
    cy.add(newElements)
  }

  // Collect the full set of trace node IDs (both new and pre-existing)
  const traceNodeIds = new Set(
    (rawGraph.nodes ?? []).map((n) => String(n.id))
  )

  // Black out everything first
  cy.elements().style('display', 'none')

  // Reveal: trace nodes + their compound parents
  const traceNodes = cy.nodes().filter((n) => traceNodeIds.has(n.id()))
  const parents = traceNodes.parents()
  const traceSet = traceNodes.union(parents)
  const traceEdges = traceSet.edgesWith(traceSet)
  const revealSet = traceSet.union(traceEdges)

  revealSet.style('display', 'element')

  // Run a focused animated fcose layout only on the revealed elements
  revealSet.layout({
    name: 'fcose',
    animate: true,
    animationDuration: 600,
    fit: true,
    padding: 50,
    nodeRepulsion: 6000,
    idealEdgeLength: 45,
    gravity: 0.4,
    tile: true,
    packComponents: true,
  }).run()

  return traceNodeIds
}

/**
 * exitLevel4Trace(cy)
 *
 * Restores all elements to visible and re-runs the main fcose layout,
 * effectively leaving the Level 4 focus mode.
 */
export function exitLevel4Trace(cy) {
  if (!cy) return
  cy.elements().removeStyle('display')
  cy.elements().removeClass('highlighted highlighted-focal dimmed')
  cy.layout(FCOSE_LAYOUT).run()
}


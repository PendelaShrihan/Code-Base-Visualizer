import { useEffect, useRef, useCallback } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
import fcose from 'cytoscape-fcose'

// ---------------------------------------------------------------------------
// Register fcose layout extension once with Cytoscape
// ---------------------------------------------------------------------------
cytoscape.use(fcose)

// ---------------------------------------------------------------------------
// Layout Configuration
// ---------------------------------------------------------------------------
export const FCOSE_LAYOUT = {
  name: 'fcose',
  animate: false,
  nodeRepulsion: 4500,
  idealEdgeLength: 50,
  gravity: 0.25,
  tile: true,
  // Optimization options for clean readability and component packing
  fit: true,
  padding: 40,
  randomize: false,
  packComponents: true,
  nodeSeparation: 75,
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
  { selector: 'node[type = "file"]', style: { 'border-color': '#38bdf8', 'background-color': '#0c4a6e' } },
  { selector: 'node[type = "function"]', style: { 'border-color': '#34d399', 'background-color': '#064e3b' } },
  { selector: 'node[type = "class"]', style: { 'border-color': '#f59e0b', 'background-color': '#78350f' } },
  { selector: 'node[type = "import"]', style: { 'border-color': '#a78bfa', 'background-color': '#3b0764' } },
  { selector: 'node[type = "call_target"]', style: { 'border-color': '#fb923c', 'background-color': '#431407' } },
  { selector: 'node[?is_dead_code]', style: { 'border-color': '#f87171', 'border-width': 3 } },

  // Base edge styling
  {
    selector: 'edge',
    style: {
      'width': 1.5,
      'line-color': '#334155',
      'target-arrow-color': '#64748b',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'arrow-scale': 0.9,
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
      'transition-property': 'opacity, line-color, target-arrow-color, width',
      'transition-duration': '0.15s',
    },
  },

  // -------------------------------------------------------------------------
  // 1-Hop Focus: Dimming & Highlighting Classes
  // -------------------------------------------------------------------------
  {
    selector: 'node.dimmed',
    style: {
      'opacity': 0.15,
      'border-opacity': 0.15,
      'text-opacity': 0,
    },
  },
  {
    selector: 'edge.dimmed',
    style: {
      'opacity': 0.05,
      'text-opacity': 0,
    },
  },
  {
    selector: 'node.highlighted',
    style: {
      'opacity': 1,
      'border-width': 3,
      'border-color': '#38bdf8',
      'shadow-blur': 14,
      'shadow-color': '#38bdf8',
      'shadow-opacity': 0.7,
      'z-index': 999,
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
}) {
  const cyRef = useRef(null)
  const containerRef = useRef(null)

  const handleCy = useCallback((cy) => {
    if (cyRef.current === cy) return
    cyRef.current = cy

    if (onCyReady) {
      onCyReady(cy)
    }

    // -----------------------------------------------------------------------
    // Node click (tap): highlight 1-hop neighborhood & dim all other elements
    // -----------------------------------------------------------------------
    cy.on('tap', 'node', (evt) => {
      const node = evt.target
      const neighborhood = node.neighborhood().add(node)

      // Apply dimmed class to non-neighborhood elements and highlighted to neighborhood
      cy.elements().removeClass('highlighted dimmed')
      cy.elements().difference(neighborhood).addClass('dimmed')
      neighborhood.addClass('highlighted')

      if (onSelectElement) {
        onSelectElement({
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
      }
    })

    // -----------------------------------------------------------------------
    // Canvas background click (tap): clear dimming/highlighting focus
    // -----------------------------------------------------------------------
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass('highlighted dimmed')
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

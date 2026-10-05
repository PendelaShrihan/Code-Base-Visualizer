import { useEffect, useRef, useCallback } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
import dagre from 'cytoscape-dagre'

// ---------------------------------------------------------------------------
// 1. Register Dagre Layout Engine
// ---------------------------------------------------------------------------
try {
  cytoscape.use(dagre)
} catch {
  // Guard against hot module reload re-registration
}

// ---------------------------------------------------------------------------
// Top-to-Bottom Directional Flow Layout Configuration
// ---------------------------------------------------------------------------
export const layoutConfig = {
  name: 'dagre',
  rankDir: 'TB', // Top-to-bottom directional flow
  nodeSep: 50,
  edgeSep: 15,
  rankSep: 100,
  padding: 30,
}

// Backwards-compatible alias
export const DAGRE_LAYOUT = layoutConfig

// ---------------------------------------------------------------------------
// 2. Compound & Leaf Node Stylesheet Rules
// ---------------------------------------------------------------------------
export const stylesheet = [
  // Parent Compound Containers
  {
    selector: ':parent',
    style: {
      'background-color': '#1e293b',
      'background-opacity': 0.35,
      'border-width': 1,
      'border-color': '#334155',
      'border-style': 'dashed',
      'label': 'data(label)',
      'color': '#94a3b8',
      'font-size': '12px',
      'text-valign': 'top',
      'text-halign': 'left',
      'padding': '24px',
    },
  },
  // Leaf Nodes (Files / Functions)
  {
    selector: 'node[!isParent]',
    style: {
      'shape': 'round-rectangle',
      'background-color': '#0f172a',
      'border-width': 1.5,
      'border-color': '#3b82f6',
      'width': 'label',
      'height': 'label',
      'padding': '10px',
      'label': 'data(label)',
      'color': '#f8fafc',
      'font-size': '11px',
      'text-valign': 'center',
      'text-halign': 'center',
    },
  },
  // Color-coded leaf nodes by architectural node type
  {
    selector: 'node[!isParent][type = "service"], node[!isParent][type = "services"]',
    style: {
      'border-color': '#38bdf8',
      'background-color': '#0c4a6e',
    },
  },
  {
    selector: 'node[!isParent][type = "function"]',
    style: {
      'border-color': '#34d399',
      'background-color': '#064e3b',
    },
  },
  {
    selector: 'node[!isParent][type = "class"]',
    style: {
      'border-color': '#f59e0b',
      'background-color': '#78350f',
    },
  },
  {
    selector: 'node[!isParent][type = "router"], node[!isParent][type = "routes"]',
    style: {
      'border-color': '#a78bfa',
      'background-color': '#3b0764',
    },
  },
  {
    selector: 'node[!isParent][type = "model"], node[!isParent][type = "schema"]',
    style: {
      'border-color': '#ec4899',
      'background-color': '#500724',
    },
  },
  {
    selector: 'node[!isParent][type = "utility"], node[!isParent][type = "util"]',
    style: {
      'border-color': '#94a3b8',
      'background-color': '#1e293b',
    },
  },
  {
    selector: 'node[!isParent][type = "file"]',
    style: {
      'border-color': '#3b82f6',
      'background-color': '#172554',
    },
  },
  {
    selector: 'node[?is_dead_code]',
    style: {
      'border-color': '#f87171',
      'border-width': 2.5,
    },
  },
  // Edge Styling
  {
    selector: 'edge',
    style: {
      'width': 2,
      'line-color': '#475569',
      'target-arrow-color': '#3b82f6',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
    },
  },
  // Interactive Selection and Highlighting
  {
    selector: 'node:selected',
    style: {
      'border-color': '#ffffff',
      'border-width': 2.5,
    },
  },
  {
    selector: 'edge:selected',
    style: {
      'width': 3,
      'line-color': '#38bdf8',
      'target-arrow-color': '#38bdf8',
      'opacity': 1.0,
    },
  },
  {
    selector: 'node.dimmed',
    style: {
      'opacity': 0.15,
      'border-opacity': 0.1,
    },
  },
  {
    selector: 'edge.dimmed',
    style: {
      'opacity': 0.05,
    },
  },
  {
    selector: 'node.highlighted',
    style: {
      'opacity': 1,
      'border-width': 2.5,
      'border-color': '#38bdf8',
      'z-index': 999,
    },
  },
  {
    selector: 'node.highlighted-focal',
    style: {
      'opacity': 1,
      'border-width': 3.5,
      'border-color': '#f8fafc',
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
      'z-index': 998,
    },
  },
  {
    selector: 'node.chat-highlight',
    style: {
      'opacity': 1,
      'border-width': 3.5,
      'border-color': '#f472b6',
      'background-color': '#4a044e',
      'z-index': 1100,
    },
  },
]

// Backwards-compatible alias
export const CYTOSCAPE_STYLES = stylesheet

// ---------------------------------------------------------------------------
// CytoscapeGraph Component
// ---------------------------------------------------------------------------
export default function CytoscapeGraph({
  elements = [],
  layout = layoutConfig,
  stylesheet: customStylesheet = stylesheet,
  onSelectElement,
  onCyReady,
  highlightedNodeIds = [],
}) {
  const cyRef = useRef(null)
  const containerRef = useRef(null)

  const handleCy = useCallback((cy) => {
    if (cyRef.current === cy) return
    cyRef.current = cy

    if (onCyReady) {
      onCyReady(cy)
    }

    // Node tap: highlight 1-hop neighborhood & dim non-connected elements
    cy.on('tap', 'node', (evt) => {
      const node = evt.target

      // Include node, its children/parent, and direct neighborhood
      const children = node.children()
      const neighborhood = node.neighborhood().add(node).add(children)
      const compoundParents = neighborhood.nodes().parents()
      const focusSet = neighborhood.union(compoundParents)

      cy.elements().removeClass('highlighted highlighted-focal dimmed')
      cy.elements().difference(focusSet).addClass('dimmed')
      focusSet.addClass('highlighted')
      node.removeClass('highlighted').addClass('highlighted-focal')

      if (onSelectElement) {
        onSelectElement({
          type: 'node',
          id: node.id(),
          label: node.data('label'),
          category: node.data('type') || (node.data('isParent') ? 'module' : 'file'),
          level: node.data('level'),
          parent: node.data('parent'),
          isParent: !!node.data('isParent'),
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

    // Edge tap: highlight connected source and target nodes
    cy.on('tap', 'edge', (evt) => {
      const edge = evt.target
      const connected = edge.connectedNodes().add(edge)
      cy.elements().removeClass('highlighted highlighted-focal dimmed')
      cy.elements().difference(connected).addClass('dimmed')
      connected.addClass('highlighted')

      if (onSelectElement) {
        onSelectElement({
          type: 'edge',
          id: edge.id(),
          source: edge.data('source'),
          target: edge.data('target'),
          label: edge.data('label'),
        })
      }
    })

    // Canvas background tap: reset dimming and selection
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass('highlighted highlighted-focal dimmed')
        if (onSelectElement) {
          onSelectElement(null)
        }
      }
    })
  }, [onSelectElement, onCyReady])

  // ResizeObserver for responsive canvas
  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(() => cyRef.current?.resize())
    obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  // Auto-fit on elements update
  useEffect(() => {
    if (cyRef.current && elements.length > 0) {
      setTimeout(() => {
        cyRef.current?.fit(null, 35)
      }, 150)
    }
  }, [elements])

  // Chat-referenced node highlighting
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
        stylesheet={customStylesheet}
        cy={handleCy}
        className="w-full h-full"
        wheelSensitivity={0.25}
        boxSelectionEnabled={false}
      />
    </div>
  )
}

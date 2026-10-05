import { useEffect, useRef, useCallback } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
import dagre from 'cytoscape-dagre'
import {
  layoutConfig,
  DAGRE_LAYOUT,
  stylesheet,
  CYTOSCAPE_STYLES,
} from './CytoscapeGraph'

// Register Dagre layout engine
try {
  cytoscape.use(dagre)
} catch {
  // Already registered
}

// Re-export layout and stylesheet configurations
export { layoutConfig, DAGRE_LAYOUT, stylesheet, CYTOSCAPE_STYLES }
export const FCOSE_LAYOUT = layoutConfig

// ---------------------------------------------------------------------------
// GraphCanvas Component
// ---------------------------------------------------------------------------
export default function GraphCanvas({
  elements = [],
  layout = layoutConfig,
  stylesheet: customStylesheet = stylesheet,
  onSelectElement,
  onCyReady,
  viewLevel = 2,
  highlightedNodeIds = [],
}) {
  const cyRef = useRef(null)
  const containerRef = useRef(null)

  const handleCy = useCallback(
    (cy) => {
      if (cyRef.current === cy) return
      cyRef.current = cy

      if (onCyReady) {
        onCyReady(cy)
      }

      // Node tap: highlight 1-hop neighborhood & dim other elements
      cy.on('tap', 'node', (evt) => {
        const node = evt.target

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

      // Edge tap: highlight connected elements
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
    },
    [onSelectElement, onCyReady]
  )

  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(() => cyRef.current?.resize())
    obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

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

// ---------------------------------------------------------------------------
// Level 4 Trace Utilities (exported for use in App.jsx)
// ---------------------------------------------------------------------------

function _tracePayloadToCyElements(rawGraph, cy) {
  const { nodes = [], edges = [], links = [] } = rawGraph
  const edgeList = edges.length > 0 ? edges : links
  const toAdd = []
  const seenEdges = new Set()

  nodes.forEach((node, idx) => {
    const nodeId = String(node.id ?? `trace_node_${idx}`)
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

  const newElements = _tracePayloadToCyElements(rawGraph, cy)
  if (newElements.length > 0) {
    cy.add(newElements)
  }

  const traceNodeIds = new Set((rawGraph.nodes ?? []).map((n) => String(n.id)))

  cy.elements().style('display', 'none')

  const traceNodes = cy.nodes().filter((n) => traceNodeIds.has(n.id()))
  const parents = traceNodes.parents()
  const traceSet = traceNodes.union(parents)
  const traceEdges = traceSet.edgesWith(traceSet)
  const revealSet = traceSet.union(traceEdges)

  revealSet.style('display', 'element')

  revealSet.layout(layoutConfig).run()

  return traceNodeIds
}

export function exitLevel4Trace(cy) {
  if (!cy) return
  cy.elements().removeStyle('display')
  cy.elements().removeClass('highlighted highlighted-focal dimmed')
  cy.layout(layoutConfig).run()
}

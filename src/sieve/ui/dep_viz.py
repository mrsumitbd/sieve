"""
sieve/ui/dep_viz.py

Renders an interactive D3 dependency graph for a repository's direct dependencies.
Uses a force-directed layout: repo at center, packages as nodes colored by kind.
"""

import json


def render_dep_graph(deps: list[dict], repo_name: str, height: int = 420) -> str:
    """
    Generate a self-contained HTML component with a D3 force-directed graph.

    Each node represents a dependency. Edges connect the repo to each package.
    Nodes are colored by kind: main (blue), dev (purple), optional (orange).

    Args:
        deps:      List of dicts with keys name, version, kind
        repo_name: Repository name shown at center
        height:    Component height in pixels

    Returns:
        HTML string for use with st.components.v1.html()
    """
    if not deps:
        return f"""<div style="background:#1a1b26;color:#565f89;
            font-family:monospace;padding:20px;text-align:center;height:{height}px;
            display:flex;align-items:center;justify-content:center;">
            No dependency manifest found for this repository.
        </div>"""

    # Build graph data
    nodes = [{"id": repo_name, "kind": "repo", "version": None}]
    links = []
    seen  = set()

    for dep in deps:
        name = dep.get("name", "")
        if not name or name in seen:
            continue
        seen.add(name)
        nodes.append({
            "id":      name,
            "kind":    dep.get("kind", "main"),
            "version": dep.get("version"),
        })
        links.append({"source": repo_name, "target": name})

    graph_json = json.dumps({"nodes": nodes, "links": links})
    short_repo = repo_name.split("/")[-1] if "/" in repo_name else repo_name

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #1a1b26;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    overflow: hidden;
  }}
  #controls {{
    display: flex;
    gap: 12px;
    padding: 6px 12px;
    background: #16161e;
    border-bottom: 1px solid #2a2b3d;
    align-items: center;
    height: 36px;
    flex-wrap: wrap;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 5px;
    font-size: 10px; color: #a9b1d6;
  }}
  .legend-dot {{
    width: 10px; height: 10px; border-radius: 50%;
  }}
  #graph-container {{
    width: 100%;
    height: {height - 36}px;
    position: relative;
    overflow: hidden;
  }}
  svg {{ width: 100%; height: 100%; }}
  .node circle {{
    stroke-width: 1.5px;
    cursor: pointer;
  }}
  .node text {{
    font-size: 9px;
    fill: #c0caf5;
    pointer-events: none;
    font-family: 'JetBrains Mono', monospace;
  }}
  .link {{
    stroke: #3b3d57;
    stroke-width: 1px;
    stroke-opacity: 0.7;
  }}
  .tooltip {{
    position: absolute;
    background: #16161e;
    border: 1px solid #3b3d57;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 10px;
    color: #a9b1d6;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    max-width: 220px;
    word-break: break-all;
    z-index: 10;
    font-family: 'JetBrains Mono', monospace;
  }}
</style>
</head>
<body>
<div id="controls">
  <div class="legend-item">
    <div class="legend-dot" style="background:#7aa2f7"></div> Main
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#bb9af7"></div> Dev
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#e0af68"></div> Optional
  </div>
  <div class="legend-item" style="margin-left:auto;color:#565f89;">
    {len(deps)} direct dependencies · drag to explore
  </div>
</div>
<div id="graph-container">
  <div class="tooltip" id="tooltip"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const GRAPH  = {graph_json};
const W      = document.getElementById("graph-container").clientWidth  || 800;
const H      = document.getElementById("graph-container").clientHeight || {height - 36};
const COLORS = {{
  repo:     "#f7768e",
  main:     "#7aa2f7",
  dev:      "#bb9af7",
  optional: "#e0af68",
}};
const RADII = {{
  repo: 18,
  main: 7,
  dev:  6,
  optional: 6,
}};

const tooltip = document.getElementById("tooltip");

const svg = d3.select("#graph-container").append("svg");
const g   = svg.append("g");

// Zoom
const zoom = d3.zoom().scaleExtent([0.3, 3])
  .on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

// Force simulation
const sim = d3.forceSimulation(GRAPH.nodes)
  .force("link",   d3.forceLink(GRAPH.links).id(d => d.id).distance(90))
  .force("charge", d3.forceManyBody().strength(-180))
  .force("center", d3.forceCenter(W / 2, H / 2))
  .force("collide", d3.forceCollide(30));

// Links
const link = g.selectAll(".link")
  .data(GRAPH.links)
  .join("line")
  .attr("class", "link");

// Nodes
const node = g.selectAll(".node")
  .data(GRAPH.nodes)
  .join("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", (e, d) => {{
      if (!e.active) sim.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }})
    .on("drag",  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on("end",   (e, d) => {{
      if (!e.active) sim.alphaTarget(0);
      d.fx = null; d.fy = null;
    }}))
  .on("mouseover", (e, d) => {{
    const ver = d.version ? `\\n${{d.version}}` : "";
    const lbl = d.kind === "repo" ? d.id : `${{d.id}}${{ver}}\\n[${{d.kind}}]`;
    tooltip.style.opacity = 1;
    tooltip.textContent   = lbl;
    tooltip.style.left    = (e.offsetX + 12) + "px";
    tooltip.style.top     = (e.offsetY - 10) + "px";
  }})
  .on("mouseout", () => tooltip.style.opacity = 0);

node.append("circle")
  .attr("r",      d => RADII[d.kind] || 7)
  .attr("fill",   d => COLORS[d.kind] || "#a9b1d6")
  .attr("stroke", d => d3.color(COLORS[d.kind] || "#a9b1d6").darker(0.8));

node.append("text")
  .attr("x", d => (RADII[d.kind] || 7) + 4)
  .attr("dominant-baseline", "middle")
  .text(d => {{
    const label = d.kind === "repo"
      ? "{short_repo}"
      : (d.id.length > 20 ? d.id.slice(0, 18) + "…" : d.id);
    return label;
  }});

// Tick
sim.on("tick", () => {{
  link
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});
</script>
</body>
</html>"""
"""
sieve/ui/ast_viz.py

Renders an interactive D3 collapsible tree of a code snippet's AST.
Uses a box-style layout (type label + token text) growing top-to-bottom.
Used in the Random Sample Viewer in Home.py.

Supports Python, Java, JavaScript, and C++ via tree-sitter.
"""

import json
from typing import Optional


# ── Node type color categories ────────────────────────────────────────────────

_COLORS = {
    "definition":   "#7aa2f7",  # blue
    "control":      "#bb9af7",  # purple
    "literal":      "#9ece6a",  # green
    "identifier":   "#e0af68",  # orange
    "operator":     "#f7768e",  # red/pink
    "type":         "#2ac3de",  # cyan
    "default":      "#a9b1d6",  # muted lavender
}

_TYPE_CATEGORY = {
    "function_definition":        "definition",
    "async_function_definition":  "definition",
    "class_definition":           "definition",
    "method_declaration":         "definition",
    "class_declaration":          "definition",
    "function_declaration":       "definition",
    "arrow_function":             "definition",
    "constructor_declaration":    "definition",
    "if_statement":               "control",
    "for_statement":              "control",
    "while_statement":            "control",
    "return_statement":           "control",
    "try_statement":              "control",
    "except_clause":              "control",
    "with_statement":             "control",
    "match_statement":            "control",
    "string":                     "literal",
    "integer":                    "literal",
    "float":                      "literal",
    "true":                       "literal",
    "false":                      "literal",
    "none":                       "literal",
    "string_literal":             "literal",
    "number_literal":             "literal",
    "identifier":                 "identifier",
    "attribute":                  "identifier",
    "field_identifier":           "identifier",
    "binary_operator":            "operator",
    "unary_operator":             "operator",
    "comparison_operator":        "operator",
    "augmented_assignment":       "operator",
    "assignment":                 "operator",
    "type":                       "type",
    "generic_type":               "type",
    "annotated_type":             "type",
    "type_identifier":            "type",
}


def _node_to_dict(node, source_bytes: bytes, max_depth: int, depth: int = 0) -> dict:
    """Recursively convert a tree-sitter node to a JSON-serializable dict."""
    node_type = node.type
    is_leaf   = len(node.children) == 0
    text      = node.text.decode("utf-8", errors="replace") if is_leaf else None
    if text and len(text) > 30:
        text = text[:27] + "..."

    category = _TYPE_CATEGORY.get(node_type, "default")
    color    = _COLORS[category]

    result = {
        "node_type": node_type,
        "text":      text,        # token text for leaf nodes, None for internal nodes
        "color":     color,
        "name":      node_type,   # kept for test compatibility
        "children":  [],
    }

    if depth < max_depth:
        for child in node.children:
            if child.type in (",", ";", "(", ")", "{", "}", "[", "]", ":", "."):
                continue
            result["children"].append(
                _node_to_dict(child, source_bytes, max_depth, depth + 1)
            )

    return result


def build_ast_json(
    source: str,
    language: str,
    max_depth: int = 6,
) -> Optional[dict]:
    """
    Parse source code and return a JSON tree for D3 visualization.

    Args:
        source:    Source code string
        language:  One of Python, Java, JavaScript, C++
        max_depth: Maximum tree depth to render (default 6)

    Returns:
        dict suitable for JSON serialization, or None on parse failure
    """
    try:
        from tree_sitter import Language, Parser

        if language == "Python":
            import tree_sitter_python as ts_lang
        elif language == "Java":
            import tree_sitter_java as ts_lang
        elif language == "JavaScript":
            import tree_sitter_javascript as ts_lang
        elif language == "C++":
            import tree_sitter_cpp as ts_lang
        else:
            return None

        lang         = Language(ts_lang.language())
        parser       = Parser(lang)
        source_bytes = source.encode("utf-8")
        tree         = parser.parse(source_bytes)

        return _node_to_dict(tree.root_node, source_bytes, max_depth)

    except Exception:
        return None


def render_ast_component(ast_json: dict, height: int = 500) -> str:
    """
    Generate a self-contained HTML component with a D3 box-style AST tree.
    Layout grows top-to-bottom. Each node is a rounded rectangle with:
      - Top section: node type (colored by category)
      - Bottom section: token text (for leaf nodes only)

    Args:
        ast_json: JSON tree from build_ast_json()
        height:   Component height in pixels

    Returns:
        HTML string for use with st.components.v1.html()
    """
    data_json = json.dumps(ast_json)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #1a1b26;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    overflow: hidden;
  }}
  #controls {{
    display: flex;
    gap: 8px;
    padding: 6px 10px;
    background: #16161e;
    border-bottom: 1px solid #2a2b3d;
    align-items: center;
    height: 36px;
  }}
  button {{
    background: #2a2b3d;
    color: #a9b1d6;
    border: 1px solid #3b3d57;
    border-radius: 4px;
    padding: 3px 9px;
    font-size: 10px;
    cursor: pointer;
    font-family: inherit;
  }}
  button:hover {{ background: #3b3d57; color: #c0caf5; }}
  label {{ color: #565f89; font-size: 10px; }}
  input[type=range] {{ width: 70px; accent-color: #7aa2f7; }}
  #depth-val {{ color: #7aa2f7; font-size: 10px; min-width: 10px; }}
  #tree-container {{
    width: 100%;
    height: {height - 36}px;
    overflow: auto;
    position: relative;
  }}
  svg {{ overflow: visible; }}
  .link {{
    fill: none;
    stroke: #3b3d57;
    stroke-width: 1.5px;
  }}
  .node-rect {{
    rx: 6; ry: 6;
    stroke-width: 1.5px;
    cursor: pointer;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.4));
  }}
  .node-type {{
    font-size: 9px;
    font-weight: bold;
    fill: #1a1b26;
    text-anchor: middle;
    dominant-baseline: middle;
    pointer-events: none;
  }}
  .node-text {{
    font-size: 8px;
    fill: #c0caf5;
    text-anchor: middle;
    dominant-baseline: middle;
    pointer-events: none;
  }}
  .node-divider {{
    stroke: rgba(0,0,0,0.2);
    stroke-width: 1px;
  }}
  .collapsed-badge {{
    font-size: 8px;
    fill: #565f89;
    text-anchor: middle;
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
    max-width: 240px;
    word-break: break-all;
    white-space: pre-wrap;
    z-index: 10;
  }}
</style>
</head>
<body>
<div id="controls">
  <button id="btn-expand">Expand All</button>
  <button id="btn-collapse">Collapse All</button>
  <button id="btn-fit">Fit View</button>
  <label>Depth:</label>
  <input type="range" id="depth-slider" min="1" max="10" value="6">
  <span id="depth-val">6</span>
</div>
<div id="tree-container">
  <div class="tooltip" id="tooltip"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const RAW_DATA = {data_json};
let maxDepth = 6;

// Box dimensions
const BOX_W  = 130;   // box width
const BOX_TH = 22;    // type row height
const BOX_TXT_H = 16; // text row height (leaf only)
const BOX_GAP_X = 30; // horizontal gap between boxes
const BOX_GAP_Y = 60; // vertical gap between levels

function nodeHeight(d) {{
  return d.data.text ? BOX_TH + BOX_TXT_H : BOX_TH;
}}

function pruneToDepth(node, depth) {{
  if (depth <= 0) {{
    return {{ ...node, children: undefined, _children: node.children }};
  }}
  const children = (node.children || []).map(c => pruneToDepth(c, depth - 1));
  return {{ ...node, children: children.length ? children : undefined }};
}}

function cloneDeep(obj) {{ return JSON.parse(JSON.stringify(obj)); }}

const container = document.getElementById("tree-container");
const tooltip   = document.getElementById("tooltip");

const svg = d3.select(container).append("svg");
const g   = svg.append("g");

const zoom = d3.zoom()
  .scaleExtent([0.1, 4])
  .on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

function draw() {{
  g.selectAll("*").remove();

  const data = pruneToDepth(cloneDeep(RAW_DATA), maxDepth);
  const root = d3.hierarchy(data, d => d.children);

  // Custom tree layout with box sizes
  const treeLayout = d3.tree()
    .nodeSize([BOX_W + BOX_GAP_X, BOX_TH + BOX_GAP_Y]);
  treeLayout(root);

  const nodes = root.descendants();
  const minX  = d3.min(nodes, d => d.x) - BOX_W / 2 - 20;
  const maxX  = d3.max(nodes, d => d.x) + BOX_W / 2 + 20;
  const minY  = -20;
  const maxY  = d3.max(nodes, d => d.y) + BOX_TH + BOX_TXT_H + 20;
  const svgW  = maxX - minX;
  const svgH  = maxY - minY;

  svg.attr("width", Math.max(svgW, container.clientWidth))
     .attr("height", Math.max(svgH, container.clientHeight - 36));

  // Center root
  g.attr("transform", `translate(${{-minX}}, ${{-minY}})`);

  // Links — connect bottom-center of parent to top-center of child
  g.selectAll(".link")
    .data(root.links())
    .join("path")
    .attr("class", "link")
    .attr("d", d => {{
      const ph = nodeHeight(d.source.data);
      const sx = d.source.x, sy = d.source.y + ph;
      const tx = d.target.x, ty = d.target.y;
      const my = (sy + ty) / 2;
      return `M${{sx}},${{sy}} C${{sx}},${{my}} ${{tx}},${{my}} ${{tx}},${{ty}}`;
    }});

  // Node groups
  const node = g.selectAll(".node")
    .data(nodes)
    .join("g")
    .attr("class", "node")
    .attr("transform", d => `translate(${{d.x - BOX_W / 2}},${{d.y}})`)
    .on("mouseover", (e, d) => {{
      const txt = d.data.text ? `${{d.data.node_type}}\\n"${{d.data.text}}"` : d.data.node_type;
      tooltip.style.opacity = 1;
      tooltip.textContent   = txt;
      tooltip.style.left    = (e.offsetX + 12) + "px";
      tooltip.style.top     = (e.offsetY - 10) + "px";
    }})
    .on("mouseout", () => tooltip.style.opacity = 0);

  // Type section (top, colored)
  node.append("rect")
    .attr("class", "node-rect")
    .attr("width",  BOX_W)
    .attr("height", d => nodeHeight(d.data))
    .attr("fill",   d => d.data.color || "#a9b1d6")
    .attr("stroke", d => d3.color(d.data.color || "#a9b1d6").darker(1));

  // Type label
  node.append("text")
    .attr("class", "node-type")
    .attr("x", BOX_W / 2)
    .attr("y", BOX_TH / 2)
    .text(d => {{
      const t = d.data.node_type || "";
      return t.length > 18 ? t.slice(0, 16) + "…" : t;
    }});

  // Divider + text row for leaf nodes
  const leaves = node.filter(d => d.data.text);

  leaves.append("line")
    .attr("class", "node-divider")
    .attr("x1", 0).attr("x2", BOX_W)
    .attr("y1", BOX_TH).attr("y2", BOX_TH);

  // Text row background (slightly darker)
  leaves.append("rect")
    .attr("x", 1).attr("y", BOX_TH)
    .attr("width", BOX_W - 2)
    .attr("height", BOX_TXT_H - 1)
    .attr("rx", 0).attr("ry", 0)
    .attr("fill", d => d3.color(d.data.color || "#a9b1d6").darker(1.5));

  leaves.append("text")
    .attr("class", "node-text")
    .attr("x", BOX_W / 2)
    .attr("y", BOX_TH + BOX_TXT_H / 2)
    .text(d => {{
      const t = d.data.text || "";
      return t.length > 16 ? t.slice(0, 14) + "…" : t;
    }});

  // Collapsed badge
  node.filter(d => !d.children && d.data._children)
    .append("text")
    .attr("class", "collapsed-badge")
    .attr("x", BOX_W / 2)
    .attr("y", d => nodeHeight(d.data) + 12)
    .text(d => `[+${{d.data._children.length}}]`);
}}

// Controls
document.getElementById("depth-slider").addEventListener("input", e => {{
  maxDepth = +e.target.value;
  document.getElementById("depth-val").textContent = maxDepth;
  draw();
}});

document.getElementById("btn-expand").addEventListener("click", () => {{
  maxDepth = 10;
  document.getElementById("depth-slider").value = 10;
  document.getElementById("depth-val").textContent = 10;
  draw();
}});

document.getElementById("btn-collapse").addEventListener("click", () => {{
  maxDepth = 1;
  document.getElementById("depth-slider").value = 1;
  document.getElementById("depth-val").textContent = 1;
  draw();
}});

document.getElementById("btn-fit").addEventListener("click", () => {{
  const svgEl   = svg.node();
  const contEl  = container;
  const svgW    = +svg.attr("width")  || svgEl.getBoundingClientRect().width;
  const svgH    = +svg.attr("height") || svgEl.getBoundingClientRect().height;
  const contW   = contEl.clientWidth;
  const contH   = contEl.clientHeight;
  const scale   = Math.min(contW / svgW, contH / svgH, 1) * 0.9;
  const tx      = (contW - svgW * scale) / 2;
  const ty      = (contH - svgH * scale) / 2;
  svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}});

draw();
</script>
</body>
</html>"""
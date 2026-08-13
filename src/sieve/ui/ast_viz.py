"""
sieve/ui/ast_viz.py

Renders an interactive D3 collapsible tree of a code snippet's AST.
Used in the Random Sample Viewer in Home.py.

Supports Python, Java, JavaScript, and C++ via tree-sitter.
"""

import json
from typing import Optional


# ── Node type color categories ────────────────────────────────────────────────

_COLORS = {
    # Declarations / definitions
    "definition":   "#7aa2f7",  # blue
    # Control flow
    "control":      "#bb9af7",  # purple
    # Literals
    "literal":      "#9ece6a",  # green
    # Identifiers / names
    "identifier":   "#e0af68",  # orange
    # Operators / punctuation
    "operator":     "#f7768e",  # red/pink
    # Types / annotations
    "type":         "#2ac3de",  # cyan
    # Default
    "default":      "#a9b1d6",  # muted lavender
}

_TYPE_CATEGORY = {
    # Definitions
    "function_definition":        "definition",
    "async_function_definition":  "definition",
    "class_definition":           "definition",
    "method_declaration":         "definition",
    "class_declaration":          "definition",
    "function_declaration":       "definition",
    "arrow_function":             "definition",
    "constructor_declaration":    "definition",
    # Control flow
    "if_statement":               "control",
    "for_statement":              "control",
    "while_statement":            "control",
    "return_statement":           "control",
    "try_statement":              "control",
    "except_clause":              "control",
    "with_statement":             "control",
    "match_statement":            "control",
    # Literals
    "string":                     "literal",
    "integer":                    "literal",
    "float":                      "literal",
    "true":                       "literal",
    "false":                      "literal",
    "none":                       "literal",
    "string_literal":             "literal",
    "number_literal":             "literal",
    # Identifiers
    "identifier":                 "identifier",
    "attribute":                  "identifier",
    "field_identifier":           "identifier",
    # Operators
    "binary_operator":            "operator",
    "unary_operator":             "operator",
    "comparison_operator":        "operator",
    "augmented_assignment":       "operator",
    "assignment":                 "operator",
    # Types
    "type":                       "type",
    "generic_type":               "type",
    "annotated_type":             "type",
    "type_identifier":            "type",
}


def _node_to_dict(node, source_bytes: bytes, max_depth: int, depth: int = 0) -> dict:
    """Recursively convert a tree-sitter node to a JSON-serializable dict."""
    node_type = node.type

    # Leaf nodes — include the token text
    is_leaf = len(node.children) == 0
    text = node.text.decode("utf-8", errors="replace") if is_leaf else None

    # Truncate long tokens
    if text and len(text) > 40:
        text = text[:37] + "..."

    label = node_type if not text else f"{node_type}: {repr(text)}"
    category = _TYPE_CATEGORY.get(node_type, "default")
    color = _COLORS[category]

    result = {
        "name":     label,
        "type":     node_type,
        "color":    color,
        "children": [],
    }

    if depth < max_depth:
        for child in node.children:
            # Skip pure punctuation nodes to reduce noise
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

        lang    = Language(ts_lang.language())
        parser  = Parser(lang)
        source_bytes = source.encode("utf-8")
        tree    = parser.parse(source_bytes)

        return _node_to_dict(tree.root_node, source_bytes, max_depth)

    except Exception:
        return None


def render_ast_component(ast_json: dict, height: int = 500) -> str:
    """
    Generate a self-contained HTML component with a D3 collapsible tree.

    Args:
        ast_json: JSON tree from build_ast_json()
        height:   Component height in pixels

    Returns:
        HTML string for use with st.components.v1.html()
    """
    data_json = json.dumps(ast_json)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #1a1b26;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    overflow: auto;
  }}

  #controls {{
    display: flex;
    gap: 8px;
    padding: 8px 12px;
    background: #16161e;
    border-bottom: 1px solid #2a2b3d;
    align-items: center;
  }}

  button {{
    background: #2a2b3d;
    color: #a9b1d6;
    border: 1px solid #3b3d57;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
    cursor: pointer;
    font-family: inherit;
    transition: background 0.15s;
  }}

  button:hover {{ background: #3b3d57; color: #c0caf5; }}

  label {{
    color: #565f89;
    font-size: 11px;
  }}

  input[type=range] {{
    width: 80px;
    accent-color: #7aa2f7;
  }}

  #depth-val {{
    color: #7aa2f7;
    font-size: 11px;
    min-width: 12px;
  }}

  #tree-container {{
    width: 100%;
    height: {height - 44}px;
    overflow: auto;
  }}

  svg {{
    min-width: 100%;
  }}

  .node circle {{
    stroke-width: 1.5px;
    cursor: pointer;
    transition: r 0.15s;
  }}

  .node circle:hover {{
    r: 7;
  }}

  .node text {{
    font-size: 10px;
    fill: #c0caf5;
    dominant-baseline: middle;
    pointer-events: none;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }}

  .node text.collapsed-indicator {{
    fill: #565f89;
    font-size: 9px;
  }}

  .link {{
    fill: none;
    stroke: #2a2b3d;
    stroke-width: 1.2px;
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
    max-width: 200px;
    word-break: break-all;
  }}
</style>
</head>
<body>

<div id="controls">
  <button id="btn-expand">Expand All</button>
  <button id="btn-collapse">Collapse All</button>
  <button id="btn-reset">Reset View</button>
  <label>Depth: </label>
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

// ── Tree helpers ──────────────────────────────────────────────────────────────

function pruneToDepth(node, depth) {{
  if (depth <= 0) {{
    return {{ ...node, children: undefined, _children: node.children }};
  }}
  const children = (node.children || []).map(c => pruneToDepth(c, depth - 1));
  return {{ ...node, children: children.length ? children : undefined }};
}}

function cloneDeep(obj) {{
  return JSON.parse(JSON.stringify(obj));
}}

// ── Layout constants ──────────────────────────────────────────────────────────

const NODE_H = 22;
const NODE_W = 220;
const MARGIN = {{ top: 20, right: 20, bottom: 20, left: 20 }};

// ── D3 setup ──────────────────────────────────────────────────────────────────

const container = document.getElementById("tree-container");
const tooltip   = document.getElementById("tooltip");

const svg = d3.select(container).append("svg");
const g   = svg.append("g").attr("transform", `translate(${{MARGIN.left}},${{MARGIN.top}})`);

const zoom = d3.zoom()
  .scaleExtent([0.2, 3])
  .on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

// ── Draw ──────────────────────────────────────────────────────────────────────

function draw() {{
  g.selectAll("*").remove();

  const data  = pruneToDepth(cloneDeep(RAW_DATA), maxDepth);
  const root  = d3.hierarchy(data, d => d.children);
  const treeLayout = d3.tree().nodeSize([NODE_H, NODE_W]);
  treeLayout(root);

  // Size SVG to content
  const nodes = root.descendants();
  const minY  = d3.min(nodes, d => d.x);
  const maxY  = d3.max(nodes, d => d.x);
  const maxX  = d3.max(nodes, d => d.y);
  const svgH  = maxY - minY + MARGIN.top + MARGIN.bottom + 40;
  const svgW  = maxX + NODE_W + MARGIN.left + MARGIN.right + 60;
  svg.attr("width", svgW).attr("height", svgH);
  g.attr("transform", `translate(${{MARGIN.left}},${{MARGIN.top - minY + 20}})`);

  // Links
  g.selectAll(".link")
    .data(root.links())
    .join("path")
    .attr("class", "link")
    .attr("d", d3.linkHorizontal()
      .x(d => d.y)
      .y(d => d.x));

  // Nodes
  const node = g.selectAll(".node")
    .data(nodes)
    .join("g")
    .attr("class", "node")
    .attr("transform", d => `translate(${{d.y}},${{d.x}})`)
    .on("mouseover", (e, d) => {{
      tooltip.style.opacity = 1;
      tooltip.textContent   = d.data.name;
      tooltip.style.left    = (e.pageX + 10) + "px";
      tooltip.style.top     = (e.pageY - 10) + "px";
    }})
    .on("mousemove", e => {{
      tooltip.style.left = (e.pageX + 10) + "px";
      tooltip.style.top  = (e.pageY - 10) + "px";
    }})
    .on("mouseout", () => tooltip.style.opacity = 0);

  node.append("circle")
    .attr("r", 5)
    .attr("fill",   d => d.data.color || "#a9b1d6")
    .attr("stroke", d => d3.color(d.data.color || "#a9b1d6").darker(0.8));

  // Node labels — truncate long names
  node.append("text")
    .attr("x", d => d.children ? -9 : 9)
    .attr("text-anchor", d => d.children ? "end" : "start")
    .text(d => {{
      const name = d.data.name || "";
      return name.length > 30 ? name.slice(0, 28) + "…" : name;
    }});

  // Collapsed indicator
  node.filter(d => !d.children && d.data._children)
    .append("text")
    .attr("class", "collapsed-indicator")
    .attr("x", 9)
    .attr("dy", "1.2em")
    .attr("text-anchor", "start")
    .text(d => `[+${{d.data._children.length}} hidden]`);
}}

// ── Controls ──────────────────────────────────────────────────────────────────

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

document.getElementById("btn-reset").addEventListener("click", () => {{
  svg.call(zoom.transform, d3.zoomIdentity.translate(MARGIN.left, MARGIN.top));
}});

// Initial render
draw();
</script>
</body>
</html>
"""
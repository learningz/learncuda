#!/usr/bin/env python3
"""
Beautiful ASCII-art → SVG converter for CUDA theory tutorials.

Parses box-drawing structures from fenced code blocks, then renders
proper SVG with rounded rectangles, gradients, color-coded sections,
and clean typography. Falls back to styled monospace for unparseable areas.

Usage:
    python3 generate_diagrams.py            # generate + patch markdown
    python3 generate_diagrams.py --dry-run  # preview counts only
"""

import os, re, sys, glob, html, unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Configuration ────────────────────────────────────────────────────────────

DIAGRAM_DIR = "diagrams"
CELL_W = 8.4
CELL_H = 21
PAD = 28

FONT_STACK = "'Inter','Noto Sans SC','Helvetica Neue',sans-serif"
MONO_STACK = ("'Cascadia Code','Fira Code','JetBrains Mono',"
              "'Menlo','Consolas','Noto Sans Mono CJK SC',monospace")
FONT_SIZE = 13
MONO_SIZE = 12.5
LINE_H = CELL_H

BG       = "#1e1e2e"
SURF0    = "#313244"
SURF1    = "#45475a"
OVERLAY  = "#585b70"
TEXT     = "#cdd6f4"
SUBTEXT  = "#a6adc8"
BLUE     = "#89b4fa"
GREEN    = "#a6e3a1"
PEACH    = "#fab387"
MAUVE    = "#cba6f7"
RED      = "#f38ba8"
YELLOW   = "#f9e2af"
TEAL     = "#94e2d5"
SKY      = "#89dceb"
PINK     = "#f5c2e7"

DEPTH_PALETTE = [
    (BLUE,  "#89b4fa18"), (GREEN, "#a6e3a118"), (PEACH, "#fab38718"),
    (MAUVE, "#cba6f718"), (TEAL,  "#94e2d518"), (YELLOW,"#f9e2af18"),
]

KW_COLORS = {
    "sm": BLUE, "gpc": MAUVE, "tpc": TEAL, "hbm": RED, "dram": RED,
    "l1": GREEN, "l2": YELLOW, "cache": GREEN, "shared mem": GREEN,
    "register": PEACH, "warp": SKY, "thread": SKY, "block": BLUE,
    "grid": MAUVE, "alu": PEACH, "fp32": PEACH, "fp64": PEACH,
    "tensor": PINK, "nvlink": TEAL, "pcie": YELLOW,
    "python": GREEN, "cuda": BLUE, "driver": MAUVE, "kernel": RED,
    "gpu": BLUE, "cpu": PEACH, "cublas": GREEN, "cudnn": GREEN,
    "nccl": TEAL, "sram": GREEN, "memory": SKY,
}

BOX_STRUCT = set("┌┐└┘│─┬┴├┤┼╔╗╚╝║═╦╩╠╣╬╭╮╯╰━┃┏┓┗┛┣┫┳┻╋╒╕╘╛╞╡╤╧╥╨╪")
CORNERS_TL = set("┌╔╭┏")
CORNERS_TR = set("┐╗╮┓")
CORNERS_BL = set("└╚╰┗")
CORNERS_BR = set("┘╝╯┛")
HORIZ = set("─═━")
VERT  = set("│║┃")
TEE_L = set("├╠┣╞")
TEE_R = set("┤╣┫╡")
DIVIDERS = TEE_L | TEE_R
ARROW_CHARS = set("→←↑↓►◄▲▼⟶⟵")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _dw(ch):
    if ch == "\t": return 4
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1

def dw(s):
    return sum(_dw(c) for c in s)

def char_at_col(line, col):
    c = 0
    for ch in line:
        if c == col: return ch
        c += _dw(ch)
        if c > col: return ch
    return " "

def col_of_char(line, idx):
    return sum(_dw(line[i]) for i in range(idx))

def strip_box(t):
    return "".join(ch for ch in t if ch not in BOX_STRUCT).strip()

def kw_color(text):
    tl = text.lower()
    for kw, c in KW_COLORS.items():
        if kw in tl: return c
    return BLUE

def has_structural(text):
    return any(ch in BOX_STRUCT for ch in text)

def esc(s): return html.escape(s)

# ── Box parser ───────────────────────────────────────────────────────────────

@dataclass
class Box:
    r: int; c: int; w: int; h: int
    sections: list = field(default_factory=list)
    depth: int = 0

def _build_col_grid(lines):
    """Build a 2D grid indexed by (row, display_col) → char."""
    max_col = max((dw(l) for l in lines), default=0)
    grid = {}
    for ri, line in enumerate(lines):
        col = 0
        for ch in line:
            grid[(ri, col)] = ch
            col += _dw(ch)
    return grid, len(lines), max_col

def _find_boxes(grid, nrows, ncols):
    """Find all rectangles formed by box-drawing corners."""
    boxes = []
    used_corners = set()
    tl_positions = []
    for r in range(nrows):
        for c in range(ncols + 1):
            if grid.get((r, c), " ") in CORNERS_TL:
                tl_positions.append((r, c))

    for (r0, c0) in tl_positions:
        c1 = c0 + 1
        while c1 <= ncols:
            ch = grid.get((r0, c1), " ")
            if ch in CORNERS_TR:
                r1 = r0 + 1
                while r1 < nrows:
                    bl = grid.get((r1, c0), " ")
                    br = grid.get((r1, c1), " ")
                    if bl in (CORNERS_BL | TEE_L | VERT) and br in (CORNERS_BR | TEE_R | VERT):
                        if bl in CORNERS_BL and br in CORNERS_BR:
                            boxes.append(Box(r=r0, c=c0, w=c1-c0, h=r1-r0))
                            break
                    elif bl not in VERT and bl not in TEE_L and bl not in CORNERS_BL:
                        break
                    r1 += 1
                break
            elif ch not in HORIZ and ch not in set("┬╦┳╤"):
                break
            c1 += 1
    return boxes

def _assign_depth(boxes):
    """Assign nesting depth: a box contained inside another gets depth+1."""
    boxes.sort(key=lambda b: b.w * b.h, reverse=True)
    for i, b in enumerate(boxes):
        b.depth = 0
        for j in range(i):
            p = boxes[j]
            if (p.r < b.r and p.c < b.c and
                p.r + p.h > b.r + b.h and p.c + p.w > b.c + b.w):
                b.depth = max(b.depth, p.depth + 1)

def _extract_sections(box, lines):
    """Extract text sections from a box, split by horizontal dividers."""
    sections = []
    cur = []
    for ri in range(box.r + 1, box.r + box.h):
        line = lines[ri] if ri < len(lines) else ""
        row_text = ""
        col = 0
        inside = False
        chars = []
        for ch in line:
            w = _dw(ch)
            if col == box.c:
                inside = True
            elif col >= box.c + box.w:
                inside = False
            if inside and col > box.c and col < box.c + box.w:
                if ch in TEE_L or ch in TEE_R or (ch in HORIZ and col > box.c + 1):
                    pass
                elif ch not in VERT:
                    chars.append(ch)
            col += w
        row_text = "".join(chars).strip()

        is_divider = False
        lch = char_at_col(line, box.c) if len(line) > 0 else " "
        if lch in TEE_L:
            is_divider = True
        if is_divider and cur:
            sections.append(cur)
            cur = []
        elif row_text:
            cur.append(row_text)
    if cur:
        sections.append(cur)
    box.sections = sections

def parse_diagram(text):
    """Parse ASCII art text into boxes + free text lines."""
    lines = text.split("\n")
    grid, nrows, ncols = _build_col_grid(lines)
    boxes = _find_boxes(grid, nrows, ncols)
    _assign_depth(boxes)
    for b in boxes:
        _extract_sections(b, lines)

    box_rows = set()
    for b in boxes:
        for ri in range(b.r, b.r + b.h + 1):
            box_rows.add(ri)

    free_lines = []
    for ri, line in enumerate(lines):
        if ri not in box_rows:
            stripped = strip_box(line)
            if stripped:
                free_lines.append((ri, stripped))

    annotation_lines = []
    for ri, line in enumerate(lines):
        if ri in box_rows and "←" in line:
            idx = line.index("←")
            ann = line[idx+1:].strip()
            ann = strip_box(ann)
            if ann:
                annotation_lines.append((ri, ann))

    return boxes, free_lines, annotation_lines, nrows, ncols

# ── SVG renderer ─────────────────────────────────────────────────────────────

def _svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">\n'
            f'<defs>\n'
            f'  <filter id="shadow" x="-4%" y="-4%" width="108%" height="108%">\n'
            f'    <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#00000055"/>\n'
            f'  </filter>\n'
            f'</defs>\n'
            f'<rect width="100%" height="100%" rx="10" fill="{BG}"/>\n')

def _svg_close():
    return "</svg>\n"

def _svg_text(x, y, text, size=FONT_SIZE, color=TEXT, anchor="start",
              font=None, weight="normal", opacity=1.0):
    f = font or FONT_STACK
    style = (f'font-family:{f};font-size:{size}px;fill:{color};'
             f'font-weight:{weight};opacity:{opacity}')
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'style="{style}">{esc(text)}</text>\n')

def _svg_rect(x, y, w, h, fill, stroke="none", stroke_w=1, rx=8, opacity=1.0,
              filter_id=None):
    f = f' filter="url(#{filter_id})"' if filter_id else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}" '
            f'opacity="{opacity}"{f}/>\n')

def _svg_line(x1, y1, x2, y2, color=OVERLAY, width=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>\n')

def _svg_arrow(x1, y1, x2, y2, color=SUBTEXT):
    mid_x = (x1 + x2) / 2
    return (f'<path d="M{x1:.0f},{y1:.0f} C{mid_x:.0f},{y1:.0f} '
            f'{mid_x:.0f},{y2:.0f} {x2:.0f},{y2:.0f}" '
            f'fill="none" stroke="{color}" stroke-width="1.5" '
            f'marker-end="url(#arrowhead)"/>\n')

def _render_beautiful(text, svg_path):
    """Main rendering pipeline: parse → layout → SVG."""
    boxes, free_lines, annotations, nrows, ncols = parse_diagram(text)
    lines = text.split("\n")

    canvas_w = max(int(ncols * CELL_W + 2 * PAD), 400)
    canvas_h = int(nrows * CELL_H + 2 * PAD)

    svg = [_svg_open(canvas_w, canvas_h)]

    svg.append(
        '<defs><marker id="arrowhead" markerWidth="8" markerHeight="6" '
        'refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" '
        f'fill="{SUBTEXT}"/></marker></defs>\n'
    )

    if boxes:
        _render_boxes(svg, boxes, annotations, canvas_w)

    if free_lines:
        _render_free_text(svg, free_lines, canvas_w)

    if not boxes and not free_lines:
        _render_fallback_mono(svg, lines)

    svg.append(_svg_close())
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("".join(svg))


def _render_boxes(svg, boxes, annotations, canvas_w):
    """Render parsed boxes as styled rounded rectangles."""
    boxes_sorted = sorted(boxes, key=lambda b: (b.depth, b.r, b.c))

    ann_map = {}
    for (ri, ann_text) in annotations:
        ann_map[ri] = ann_text

    for box in boxes_sorted:
        dp = box.depth % len(DEPTH_PALETTE)
        stroke_color, fill_color = DEPTH_PALETTE[dp]

        x = PAD + box.c * CELL_W
        y = PAD + box.r * CELL_H
        w = box.w * CELL_W
        h = box.h * CELL_H

        if box.depth == 0:
            svg.append(_svg_rect(x, y, w, h, fill_color, stroke_color,
                                 stroke_w=1.5, rx=10, filter_id="shadow"))
        else:
            svg.append(_svg_rect(x, y, w, h, fill_color, stroke_color,
                                 stroke_w=1, rx=6))

        if box.sections:
            _render_sections(svg, box, x, y, w, h, stroke_color, ann_map)


def _render_sections(svg, box, bx, by, bw, bh, stroke_color, ann_map):
    """Render text sections inside a box, with divider lines."""
    n_sections = len(box.sections)
    if n_sections == 0:
        return

    section_h = bh / n_sections
    text_y_offset = 0

    row_cursor = box.r + 1

    for si, section_lines in enumerate(box.sections):
        sy = by + si * section_h

        if si > 0:
            svg.append(_svg_line(bx + 4, sy, bx + bw - 4, sy,
                                 stroke_color, 0.5, "4 2"))

        for li, line_text in enumerate(section_lines):
            tx = bx + 12
            ty = sy + 16 + li * (FONT_SIZE + 5)

            color_for_line = kw_color(line_text)

            if li == 0 and n_sections > 1:
                svg.append(_svg_text(tx, ty, line_text, FONT_SIZE,
                                     color_for_line, weight="600"))
            else:
                svg.append(_svg_text(tx, ty, line_text, MONO_SIZE,
                                     TEXT, font=MONO_STACK))

            if row_cursor in ann_map:
                ann_text = ann_map[row_cursor]
                ax = bx + bw + 8
                svg.append(_svg_text(ax, ty, f"← {ann_text}", MONO_SIZE - 1,
                                     SUBTEXT, font=MONO_STACK))
            row_cursor += 1
        if not section_lines:
            row_cursor += 1


def _render_free_text(svg, free_lines, canvas_w):
    """Render text that's outside any box."""
    for (ri, text) in free_lines:
        tx = PAD + 4
        ty = PAD + ri * CELL_H + CELL_H * 0.6

        if text.startswith("→") or text.startswith("←"):
            svg.append(_svg_text(tx, ty, text, MONO_SIZE, GREEN,
                                 font=MONO_STACK, weight="600"))
        elif ":" in text and len(text) < 60:
            svg.append(_svg_text(tx, ty, text, FONT_SIZE, TEXT, weight="600"))
        else:
            svg.append(_svg_text(tx, ty, text, MONO_SIZE, SUBTEXT,
                                 font=MONO_STACK))


def _render_fallback_mono(svg, lines):
    """Fallback: render full text as styled monospace with highlights."""
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        tx = PAD
        ty = PAD + i * CELL_H + CELL_H * 0.55

        has_box = any(ch in BOX_STRUCT for ch in line)
        if has_box:
            _render_mono_line_colored(svg, line, tx, ty)
        else:
            svg.append(_svg_text(tx, ty, line, MONO_SIZE, TEXT,
                                 font=MONO_STACK))


def _render_mono_line_colored(svg, line, x0, y):
    """Render a single monospace line with color-coded box chars."""
    segments = []
    col = 0
    cur_color = None
    cur_chars = []
    cur_x = x0

    for ch in line:
        w = _dw(ch)
        x_here = x0 + col * CELL_W

        if ch in BOX_STRUCT:
            color = BLUE
        elif ch in ARROW_CHARS:
            color = GREEN
        else:
            color = TEXT

        need_break = (color != cur_color)
        if w == 2 and cur_chars:
            need_break = True
        if cur_chars and _dw(cur_chars[-1]) == 2:
            need_break = True

        if need_break:
            if cur_chars:
                segments.append((cur_x, cur_color, "".join(cur_chars)))
            cur_chars = [ch]
            cur_color = color
            cur_x = x_here
        else:
            cur_chars.append(ch)
        col += w

    if cur_chars:
        segments.append((cur_x, cur_color, "".join(cur_chars)))

    parts = []
    for sx, sc, st in segments:
        fill = f' fill="{sc}"' if sc else ""
        parts.append(f'<tspan x="{sx:.1f}"{fill}>{esc(st)}</tspan>')

    svg.append(f'<text y="{y:.1f}" style="font-family:{MONO_STACK};'
               f'font-size:{MONO_SIZE}px">{"".join(parts)}</text>\n')

# ── Markdown processing ──────────────────────────────────────────────────────

FENCE_RE = re.compile(r"^(```[^\n]*)\n(.*?)\n(```)$", re.MULTILINE | re.DOTALL)

def process_file(md_path, dry_run=False):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    matches = list(FENCE_RE.finditer(content))
    if not matches:
        return 0

    basename = os.path.splitext(os.path.basename(md_path))[0]
    count = 0
    insertions = []

    for m in matches:
        block_text = m.group(2)
        if not has_structural(block_text):
            continue
        count += 1
        svg_name = f"{basename}_fig{count:02d}.svg"
        svg_path = os.path.join(DIAGRAM_DIR, svg_name)

        if not dry_run:
            try:
                _render_beautiful(block_text, svg_path)
            except Exception as e:
                print(f"    WARN: {svg_name} fallback due to: {e}")
                _render_fallback_file(block_text, svg_path)

        img_tag = (
            f'\n<p align="center">'
            f'<img src="{DIAGRAM_DIR}/{svg_name}" '
            f'alt="{basename} figure {count}" />'
            f'</p>\n'
        )
        insertions.append((m.end(), img_tag))

    if insertions and not dry_run:
        parts = []
        prev = 0
        for pos, tag in insertions:
            parts.append(content[prev:pos])
            parts.append(tag)
            prev = pos
        parts.append(content[prev:])
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("".join(parts))

    return count


def _render_fallback_file(text, svg_path):
    """Full monospace fallback for diagrams that fail to parse."""
    lines = text.split("\n")
    max_cols = max((dw(l) for l in lines), default=1)
    w = int(max_cols * CELL_W + 2 * PAD)
    h = int(len(lines) * CELL_H + 2 * PAD)
    svg = [_svg_open(w, h)]
    _render_fallback_mono(svg, lines)
    svg.append(_svg_close())
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("".join(svg))


def main():
    dry_run = "--dry-run" in sys.argv
    os.makedirs(DIAGRAM_DIR, exist_ok=True)

    md_files = sorted(glob.glob("*.md"))
    total = 0
    for md in md_files:
        n = process_file(md, dry_run=dry_run)
        if n:
            print(f"  {md}: {n} diagram(s)")
            total += n

    mode = "previewed" if dry_run else "generated"
    print(f"\nDone — {total} SVGs {mode} in {DIAGRAM_DIR}/")

if __name__ == "__main__":
    main()

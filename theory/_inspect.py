#!/usr/bin/env python3
"""Extract all structural ASCII-art blocks and print them with metadata."""
import re, glob

BOX_STRUCT = set("┌┐└┘│─┬┴├┤┼╔╗╚╝║═╦╩╠╣╬╭╮╯╰━┃┏┓┗┛┣┫┳┻╋")

for md in sorted(glob.glob("*.md")):
    content = open(md).read()
    blocks = list(re.finditer(r"^```[^\n]*\n(.*?)\n```$", content, re.M | re.S))
    idx = 0
    for m in blocks:
        txt = m.group(1)
        if not any(c in BOX_STRUCT for c in txt):
            continue
        idx += 1
        lines = txt.split("\n")
        # Count boxes (top-left corners)
        box_count = sum(l.count("┌") + l.count("╔") + l.count("╭") + l.count("┏") for l in lines)
        # Check for nesting (indented boxes)
        has_nesting = any(l.lstrip() != l and ("┌" in l or "╔" in l) for l in lines)
        # Check for arrows
        has_arrows = any(c in txt for c in "→←↑↓►◄▲▼")
        # Check for horizontal dividers inside boxes
        has_dividers = any("├" in l or "╠" in l or "┣" in l for l in lines)
        
        print(f"\n{'='*60}")
        print(f"{md} fig{idx:02d}: {len(lines)} lines, {box_count} boxes, "
              f"nest={has_nesting}, arrows={has_arrows}, dividers={has_dividers}")
        print(f"{'='*60}")
        for l in lines[:8]:
            print(l)
        if len(lines) > 8:
            print(f"  ... ({len(lines)-8} more lines)")

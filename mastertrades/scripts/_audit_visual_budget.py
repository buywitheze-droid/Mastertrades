"""Re-runnable design-token budget audit. Compares the live state of app.py
to the healthy ceiling defined in src/ui/tokens.py.

Run after every UI session to make sure the surface area is shrinking, not
growing. Bumps the canvas at canvases/mastertrades-ui-audit.canvas.tsx are
manual — refresh those numbers from this output.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

src = (ROOT / "app.py").read_text(encoding="utf-8")

# ── Counts ────────────────────────────────────────────────────────────────
hexes        = sorted(set(re.findall(r"#[0-9a-fA-F]{6}\b", src)))
font_sizes   = sorted({int(x) for x in re.findall(r"font-size:\s*([0-9]+)px", src)})
radii        = sorted({int(x) for x in re.findall(r"border-radius:\s*([0-9]+)px", src)})
gradients    = len(re.findall(r"linear-gradient\(|radial-gradient\(", src))
emojis       = re.findall(r"[\U0001F300-\U0001FAFF]|[\U00002600-\U000027BF]", src)
unique_emoji = sorted(set(emojis))
n_st_html    = len(re.findall(r"\bst\.html\(", src))
inline_style = len(re.findall(r"style=\"", src))
total_lines  = src.count("\n")

# Healthy ceilings (from canvases/mastertrades-ui-audit.canvas.tsx)
ceil = {
    "Hex colors":   12,
    "Font sizes":    7,
    "Border radii":  4,
    "Gradients":     0,
    "Unique emojis": 6,
}

# Pre-refactor numbers (from initial audit, 2026-05-13 baseline)
baseline = {
    "Hex colors":         74,
    "Font sizes":         21,
    "Border radii":        9,
    "Gradients":          26,
    "Unique emojis":      29,
    "Inline style attrs":964,
    "st.html() calls":    70,
    "Total lines":      5680,
}

current = {
    "Hex colors":         len(hexes),
    "Font sizes":         len(font_sizes),
    "Border radii":       len(radii),
    "Gradients":          gradients,
    "Unique emojis":      len(unique_emoji),
    "Inline style attrs": inline_style,
    "st.html() calls":    n_st_html,
    "Total lines":        total_lines,
}

print(f"=== app.py visual-budget audit ({total_lines:,} lines) ===\n")
print(f"{'Metric':<22} {'Baseline':>10} {'Current':>10} {'Δ':>10} "
      f"{'Ceiling':>10} {'Verdict':>10}")
print("-" * 80)
for k in baseline:
    cur  = current[k]
    base = baseline[k]
    delta = cur - base
    delta_str = f"{delta:+d}" if delta != 0 else "0"
    if k in ceil:
        ok = cur <= ceil[k]
        verdict = "  OK" if ok else "OVER"
        cap_str = str(ceil[k])
    else:
        verdict = ""
        cap_str = "—"
    print(f"{k:<22} {base:>10} {cur:>10} {delta_str:>10} {cap_str:>10} {verdict:>10}")

print(f"\n--- detail ---")
print(f"Font sizes still in app.py: {font_sizes}")
print(f"Border radii still in app.py: {radii}")
if hexes:
    print(f"\nHex colors still in app.py ({len(hexes)}):")
    for h in hexes:
        print(f"  {h}")
if unique_emoji:
    print(f"\nUnique emojis still in app.py ({len(unique_emoji)}):")
    print(f"  {' '.join(unique_emoji)}")

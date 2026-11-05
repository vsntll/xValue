"""Draws merged_predictor_architecture.png (the diagram embedded in README.md).

The layout is plain data below - boxes in NODES, arrows in EDGES - written out
as an SVG page and rendered to PNG with headless Chrome (or Edge), so updating
the diagram is an edit here rather than a redraw.

Run:  py -3.11 docs/architecture_diagram.py
"""

from __future__ import annotations

import html
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "merged_predictor_architecture.png"
W, H = 2000, 1990          # layout units; rendered at SCALE for a crisp PNG
SCALE = 1.84
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chrome"]

# kind -> (fill, border, title colour, text colour)
KINDS = {
    "source":  ("#efecfb", "#8f7fd0", "#33266e", "#5b4f8a"),
    "feature": ("#e4f5ec", "#62bd8c", "#1d6a43", "#2f6e4c"),
    "derived": ("#fdebe3", "#e39b7a", "#a1452a", "#a8583c"),
    "model":   ("#e3eefa", "#73a6da", "#1f4e7a", "#305a85"),
    "output":  ("#f3efe5", "#b8a985", "#56482a", "#5f533a"),
}
LEGEND = [("source", "Data source"), ("feature", "Feature table"), ("derived", "Derived table"),
          ("model", "Model"), ("output", "Output")]

# id: (kind, x, y, w, h, title, [lines])
NODES = {
    "arch": ("source", 50, 210, 235, 100, "Match archive", ["football-data.co.uk + odds, 2014–26"]),
    "live": ("source", 311, 210, 235, 100, "Live feeds",
             ["ESPN / FotMob / football-data.org", "+ ESPN per-player box scores"]),
    "fotmob": ("source", 571, 210, 235, 100, "FotMob box scores",
               ["club + international · XIs · events", "committed per season, no cache"]),
    "understat": ("source", 831, 210, 235, 100, "Understat",
                  ["per-match & per-player xG / xA", "down since Sept 2026"]),
    "fbref": ("source", 1091, 210, 235, 100, "FBref (finished)", ["player stats + team match logs"]),
    "tm": ("source", 1352, 210, 260, 100, "Transfermarkt + Sofascore",
           ["+ wfR mirror · market values 2015–27"]),
    "intl": ("source", 1637, 210, 244, 100, "International results", ["martj42 · every result since 1872"]),
    "check": ("derived", 690, 352, 420, 92, "Cross-check + Understat gap-fill",
              ["FotMob vs Understat / ESPN / football-data → verified",
               "Understat's missing matches rebuilt from FotMob"]),
    "unified": ("feature", 50, 490, 320, 100, "Unified match table",
                ["matches_all.csv — ~13.5k, all comps", "+ match_features.csv odds"]),
    "xis": ("feature", 395, 490, 193, 100, "Confirmed XIs", ["match_lineup_features.csv"]),
    "pm": ("feature", 613, 490, 278, 100, "Player-match output",
           ["understat_player_matches.csv", "+ gap-fill + FotMob per-match rows"]),
    "pss": ("feature", 916, 490, 353, 100, "Player season stats",
            ["fbref_player_season_stats.csv (~18.9k)", "current season from Understat + FotMob"]),
    "vh": ("feature", 1310, 490, 303, 100, "Value history",
           ["value_history.csv", "prev-value lags back to 2015"]),
    "intlf": ("feature", 1637, 490, 244, 100, "Internationals",
              ["fotmob_intl_player_matches.csv", "+ national_team_elo.csv"]),
    "rollup": ("derived", 193, 708, 244, 100, "Squad rollup", ["squad_season_features.csv"]),
    "valreg": ("model", 916, 708, 420, 100, "Value regression",
               ["two ridge stacks (HGB + ExtraTrees + ridge) +", "cold-start stack, clipped at €220M"]),
    "aging": ("derived", 1378, 708, 235, 100, "Aging curves",
              ["per-position age shape", "→ age adj + value forecast"]),
    "momentum": ("derived", 504, 935, 320, 100, "Squad momentum",
                 ["peer-baseline value vs. actual", "output, confirmed XI"]),
    "mmt": ("derived", 50, 1162, 589, 100, "Match model table",
            ["goals-Elo + xG-Elo, split form, odds, rest, H2H,", "xi_value_ratio, h/a_momentum"]),
    "outc": ("model", 50, 1389, 589, 100, "Outcome classifier",
             ["Poisson-Skellam (Dixon-Coles) + logreg + HGB blend;",
              "hybrid variant adds market opening odds"]),
    "elo": ("model", 739, 1389, 437, 100, "Player Elo",
            ["role-weighted attack / defence / passing,", "club + international, no market value"]),
    "outp": ("output", 50, 1616, 589, 100, "Outcome predictions",
             ["outcome_model_predictions*.csv — pure / hybrid / all", "→ live win prob · betting backtest"]),
    "ratings": ("output", 739, 1616, 437, 100, "Player ratings", ["player_elo.csv — one row per player-match"]),
    "valp": ("output", 1243, 1616, 370, 100, "Value predictions",
             ["value_model_predictions.csv + value_forecast.csv", "every player, every season"]),
    "site": ("output", 386, 1843, 891, 100, "Site",
             ["site/index.html (Players · Fixtures · Teams · Rankings · Sim)",
              "rebuilt + committed with data/processed/ every 2 days"]),
}

# (from, to, x leaving the source's bottom, x entering the target's top, channel y)
# a channel of None is a straight drop (from x must equal to x)
EDGES = [
    ("arch", "unified", 168, 168, None),
    ("live", "unified", 428, 277, 455),
    ("fotmob", "xis", 600, 491, 465),
    ("fotmob", "check", 730, 730, None),
    ("fotmob", "intlf", 780, 1688, 322),
    ("understat", "check", 925, 925, None),
    ("fbref", "unified", 1150, 336, 336),
    ("fbref", "pss", 1210, 1192, 470),
    ("check", "pm", 806, 806, None),
    ("check", "pss", 1000, 1000, None),
    ("tm", "vh", 1482, 1482, None),
    ("intl", "intlf", 1760, 1760, None),
    ("unified", "mmt", 126, 126, None),
    ("xis", "mmt", 453, 453, None),
    ("xis", "momentum", 554, 554, None),
    ("pm", "momentum", 688, 688, None),
    ("pm", "elo", 873, 873, None),
    ("pss", "rollup", 965, 315, 657),
    ("pss", "valreg", 1092, 1092, None),
    ("vh", "valreg", 1360, 1243, 670),
    ("vh", "aging", 1511, 1511, None),
    ("valreg", "momentum", 965, 781, 875),
    ("valreg", "valp", 1276, 1276, None),
    ("aging", "valp", 1511, 1511, None),
    ("rollup", "mmt", 315, 315, None),
    ("momentum", "mmt", 588, 588, None),
    ("mmt", "outc", 344, 344, None),
    ("intlf", "elo", 1760, 1126, 1346),
    ("outc", "outp", 344, 344, None),
    ("elo", "ratings", 957, 957, None),
    ("outp", "site", 344, 588, 1783),
    ("ratings", "site", 957, 840, 1783),
    ("valp", "site", 1428, 1092, 1783),
]
# arrows that don't leave a box's bottom: (points)
SIDE_EDGES = [
    [(480, 310), (480, 398), (690, 398)],       # live feeds (ESPN box scores) -> cross-check
    [(1378, 758), (1336, 758)],                 # aging -> value regression
    [(639, 1212), (796, 1212), (796, 1389)],    # match model table -> player Elo
]
R = 9  # corner radius


def _path(points: list[tuple[float, float]]) -> str:
    """Polyline with rounded corners."""
    d = f"M{points[0][0]},{points[0][1]}"
    for (x0, y0), (x1, y1), (x2, y2) in zip(points, points[1:], points[2:]):
        r = min(R, abs(x1 - x0) / 2 or R, abs(y1 - y0) / 2 or R, abs(x2 - x1) / 2 or R, abs(y2 - y1) / 2 or R)
        ax = x1 - r * (x1 > x0) + r * (x1 < x0)
        ay = y1 - r * (y1 > y0) + r * (y1 < y0)
        bx = x1 + r * (x2 > x1) - r * (x2 < x1)
        by = y1 + r * (y2 > y1) - r * (y2 < y1)
        d += f" L{ax},{ay} Q{x1},{y1} {bx},{by}"
    return d + f" L{points[-1][0]},{points[-1][1]}"


def svg() -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#9b9ba8"/></marker></defs>',
           f'<rect width="{W}" height="{H}" fill="#fafafc"/>']
    lx = 1060
    for kind, label in LEGEND:
        fill, border, _, _ = KINDS[kind]
        out.append(f'<rect x="{lx}" y="29" width="25" height="25" rx="5" fill="{fill}" stroke="{border}" stroke-width="1.6"/>')
        out.append(f'<text x="{lx + 35}" y="47" class="leg">{label}</text>')
        lx += {"source": 163, "feature": 163, "derived": 164, "model": 128}.get(kind, 0)
    out.append(f'<text x="{W / 2 - 34}" y="102" class="title">xValue — Merged Predictor Architecture</text>')
    out.append(f'<text x="{W / 2 - 34}" y="145" class="sub">big-5 leagues, 2020-21 – 2026-27 · value regression '
               '+ outcome classifier + player Elo over a shared data layer</text>')

    for f, t, fx, tx, ch in EDGES:
        _, x0, y0, w0, h0, *_ = NODES[f]
        _, x1, y1, *_ = NODES[t]
        pts = [(fx, y0 + h0), (tx, y1)] if ch is None else [(fx, y0 + h0), (fx, ch), (tx, ch), (tx, y1)]
        out.append(f'<path d="{_path(pts)}" class="e" marker-end="url(#a)"/>')
    for pts in SIDE_EDGES:
        out.append(f'<path d="{_path(pts)}" class="e" marker-end="url(#a)"/>')

    for kind, x, y, w, h, title, lines in NODES.values():
        fill, border, tc, xc = KINDS[kind]
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
                   f'stroke="{border}" stroke-width="2"/>')
        ty = y + (h / 2 - 6 if len(lines) == 1 else h / 2 - 15)
        out.append(f'<text x="{x + w / 2}" y="{ty}" class="t" fill="{tc}">{html.escape(title)}</text>')
        for i, line in enumerate(lines):
            out.append(f'<text x="{x + w / 2}" y="{ty + 30 + 19 * i}" class="l" fill="{xc}">'
                       f'{html.escape(line)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def page() -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=block" rel="stylesheet">
<style>
html,body{{margin:0;background:#fafafc}}
text{{font-family:Poppins,'Segoe UI',sans-serif}}
.t{{font-size:18px;font-weight:600;text-anchor:middle}}
.l{{font-size:12.2px;text-anchor:middle}}
.title{{font-size:35px;font-weight:700;fill:#1f1c3a;text-anchor:middle}}
.sub{{font-size:15.5px;fill:#6c6a80;text-anchor:middle}}
.leg{{font-size:14px;fill:#55536a}}
.e{{fill:none;stroke:#9b9ba8;stroke-width:1.7}}
rect{{filter:drop-shadow(2px 3px 2px rgba(60,60,90,.08))}}
</style></head><body>{svg()}</body></html>"""


def main() -> None:
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if browser is None:
        raise SystemExit("needs Chrome / Edge / Chromium to render the PNG")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "diagram.html"
        src.write_text(page(), encoding="utf-8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--force-device-scale-factor={SCALE}", f"--window-size={W},{H}",
                        "--virtual-time-budget=8000", f"--screenshot={OUT}", src.as_uri()],
                       check=True, capture_output=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

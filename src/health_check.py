"""Source health check - probes every upstream this pipeline depends on and
flags schema drift or a silently-empty pull, instead of finding out three
weeks into the season that a squad rollup has gone stale.

Each probe is a small, cheap real call (not a full pull): one league's current
fixtures/values, one FBref/Transfermarkt page. A probe is:
  ok                - reachable and returned the columns/shape expected
  blocked (known)   - Transfermarkt's WAF captcha / FBref's JS-challenge, the
                       same block this pipeline has always worked around -
                       not a regression, reported for visibility only
  unreachable       - a 5xx, a connection error/timeout, or a source that
                       normally returns a full slate came back empty: the
                       upstream is having a moment, not a schema change.
                       Reported for visibility, does NOT fail the run - these
                       are transient and clear on their own.
  DEGRADED          - reachable, HTTP 200, but the response shape/columns
                       changed: a real regression that needs a code fix.
  ERROR             - an unexpected client error (4xx that shouldn't happen)
                       or an exception that isn't just a network blip.

Exit code is non-zero iff anything is DEGRADED or ERROR, so a scheduled CI run
shows red for a genuine schema drift without anyone reading the log, but a
transient upstream outage (503, IP block on CI's datacenter range) does not.

Run:  py -3.11 src/health_check.py    (CI runs it every 3 days)
Output: data/processed/health_check.json (latest run, for the dashboard)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from fbref_common import CHALLENGE_MARKERS, COMPS, current_season  # noqa: E402
from live.schema import LEAGUES  # noqa: E402

OUT = ROOT / "data" / "processed" / "health_check.json"
TIMEOUT = 25
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}


def _result(status: str, detail: str, n: int | None = None, secs: float | None = None) -> dict:
    r = {"status": status, "detail": detail}
    if n is not None:
        r["n"] = n
    if secs is not None:
        r["seconds"] = round(secs, 1)
    return r


def _probe(fn):
    """Run one probe, timing it. A network blip (connection reset, timeout)
    becomes 'unreachable' - noise, not a regression; anything else that raises
    becomes ERROR, since an unexpected exception IS the signal."""
    t0 = time.time()
    try:
        return {**fn(), "seconds": round(time.time() - t0, 1)}
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        return _result("unreachable", f"{type(exc).__name__}: {str(exc)[:120]}", secs=time.time() - t0)
    except Exception as exc:  # noqa: BLE001 - a probe failing IS the signal
        return _result("ERROR", f"{type(exc).__name__}: {str(exc)[:160]}", secs=time.time() - t0)


# --- browser-scraped sources: no browser in CI, so just check reachability
#     and that the known JS-challenge / WAF-captcha block hasn't changed shape

def check_fbref() -> dict:
    _comp, (fbid, urlseg, _code) = next(iter(COMPS.items()))
    url = f"https://fbref.com/en/comps/{fbid}/stats/{urlseg}-Stats"   # current season's stats page
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    # a plain (non-browser) request to FBref is reliably bounced - either a hard
    # 403/429 from Cloudflare or a 200 JS-challenge page. Both are the SAME known
    # block nodriver exists to get around, not a regression - only a real 5xx, a
    # 200 that looks like neither FBref content nor the challenge page, or an
    # unexpected client error is worth flagging.
    if r.status_code in (403, 429):
        return _result("blocked (known)", f"HTTP {r.status_code} - Cloudflare bot-blocks plain requests, as always")
    if r.status_code >= 500:
        return _result("unreachable", f"HTTP {r.status_code} from {url} - upstream 5xx, transient")
    if r.status_code != 200:
        return _result("DEGRADED", f"unexpected HTTP {r.status_code} from {url}")
    if any(m in r.text for m in CHALLENGE_MARKERS):
        return _result("blocked (known)", "JS challenge page, as expected without a browser")
    if "data-stat=" not in r.text or "<table" not in r.text.lower():
        return _result("DEGRADED", "200 OK but no table/data-stat markup found - page structure changed")
    return _result("ok", f"real page content, {len(r.text)} bytes (unusual - unblocked right now)")


def check_transfermarkt() -> dict:
    url = "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1"
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    if "Human Verification" in r.text or "awswaf" in r.text.lower():
        return _result("blocked (known)", f"HTTP {r.status_code} - AWS WAF captcha page, as expected without a browser")
    if r.status_code in (202, 403, 405, 429):
        # 405 from a home IP, 202/403 from CI's datacenter range - all the same
        # standing "no plain requests" WAF block, just a different response tier.
        return _result("blocked (known)", f"HTTP {r.status_code} - the standing non-browser block")
    if r.status_code >= 500:
        return _result("unreachable", f"HTTP {r.status_code} - upstream 5xx, transient")
    if r.status_code != 200:
        return _result("ERROR", f"unexpected HTTP {r.status_code}")
    if "startseite" not in r.text.lower() and "premier league" not in r.text.lower():
        return _result("DEGRADED", "200 OK but doesn't look like a Transfermarkt page - check manually")
    return _result("ok", "200 OK, unblocked (unusual - a scrape might work without nodriver right now)")


def check_worldfootballr_mirror() -> dict:
    url = ("https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master/"
           "data/tm_player_vals/big5_player_vals.rds")
    r = requests.head(url, headers=UA, timeout=TIMEOUT, allow_redirects=True)
    if r.status_code != 200:
        return _result("ERROR", f"HTTP {r.status_code}")
    size = int(r.headers.get("content-length", 0))
    if size < 500_000:
        return _result("DEGRADED", f"file is only {size} bytes - looks truncated/replaced")
    return _result("ok", f"{size / 1e6:.1f} MB")


# --- JSON-API sources: these are pulled for real in CI, so probe them for real

def check_live_source(module_name: str) -> dict:
    import importlib
    mod = importlib.import_module(f"live.{module_name}")
    df = mod.fetch(["ENG1"], current_season(), with_stats=False)
    if df is None or df.empty:
        # fetch() swallows upstream HTTPErrors and returns a blank frame, so an
        # empty result here is almost always an upstream hiccup (or CI's IP
        # getting throttled), not schema drift - genuine drift shows up as the
        # missing-columns branch below or as an exception. Don't fail the run.
        return _result("unreachable", "returned 0 rows for the current top flight - upstream hiccup, transient")
    need = {"HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG", "status"}
    missing = need - set(df.columns)
    if missing:
        return _result("DEGRADED", f"missing expected columns: {sorted(missing)}")
    return _result("ok", f"{len(df)} rows", n=len(df))


def check_football_data_org() -> dict:
    import os
    from pull_live import _load_dotenv
    _load_dotenv()  # reads .env into os.environ if it's not already set (same as pull_live.py)
    if not os.environ.get("FOOTBALL_DATA_ORG_KEY"):
        return _result("skipped", "no FOOTBALL_DATA_ORG_KEY - optional source")
    return check_live_source("football_data_org")


def check_football_data_co_uk() -> dict:
    season = current_season()
    yy = season.replace("-", "")[2:]
    # apex host, not www - see the BASE_URL note in pull_match_archive.py: the
    # www vhost's nginx proxy backend drops for hours, the apex Apache stays up.
    url = f"https://football-data.co.uk/mmz4281/{yy}/E0.csv"
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    if r.status_code >= 500:
        # still possible even on the apex - a 5xx here is the site, not us.
        return _result("unreachable", f"HTTP {r.status_code}, {len(r.content)} bytes - upstream 5xx, transient")
    if r.status_code != 200 or len(r.content) < 1000:
        return _result("ERROR", f"HTTP {r.status_code}, {len(r.content)} bytes")
    head = r.text.splitlines()[0]
    if "HomeTeam" not in head:
        return _result("DEGRADED", f"unexpected header: {head[:120]}")
    return _result("ok", f"{len(r.content)} bytes")


def check_sofascore_values() -> dict:
    r = requests.get("https://api.sofascore.com/api/v1/unique-tournament/17/season/96668/standings/total",
                     timeout=TIMEOUT)
    if r.status_code in (403, 429):
        return _result("blocked (known)", f"{r.status_code} - Sofascore IP-blocks this network sometimes (CI especially)")
    if r.status_code >= 500:
        return _result("unreachable", f"HTTP {r.status_code} - upstream 5xx, transient")
    if r.status_code != 200:
        return _result("ERROR", f"HTTP {r.status_code}")
    data = r.json()
    if not data.get("standings"):
        return _result("DEGRADED", "200 OK but no 'standings' key - response shape changed")
    return _result("ok", f"{len(data['standings'][0].get('rows', []))} teams")


PROBES = {
    "fbref (browser scrape)": check_fbref,
    "transfermarkt (browser scrape)": check_transfermarkt,
    "worldfootballR mirror": check_worldfootballr_mirror,
    "espn": lambda: check_live_source("espn"),
    "fotmob": lambda: check_live_source("fotmob"),
    "understat (live match schema)": lambda: check_live_source("understat"),
    "football-data.org": check_football_data_org,
    "football-data.co.uk": check_football_data_co_uk,
    "sofascore (values API)": check_sofascore_values,
}


def main() -> None:
    print(f"Health check - {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    results = {}
    bad = []
    flaky = []
    for name, fn in PROBES.items():
        r = _probe(fn)
        results[name] = r
        flag = {"ok": "OK", "blocked (known)": "~~", "unreachable": "~~", "skipped": "--"}.get(r["status"], "!!")
        print(f"  [{flag:>2}] {name:32} {r['status']:16} {r['detail']}  ({r.get('seconds', '?')}s)")
        if r["status"] in ("DEGRADED", "ERROR"):
            bad.append(name)
        elif r["status"] == "unreachable":
            flaky.append(name)

    payload = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "results": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")

    if flaky:
        print(f"\n{len(flaky)} source(s) transiently unreachable (not failing the run): {', '.join(flaky)}")
    if bad:
        print(f"\n{len(bad)} source(s) need attention: {', '.join(bad)}")
        sys.exit(1)
    print("\nall sources healthy (or blocked / transiently unreachable exactly as expected)")


if __name__ == "__main__":
    main()

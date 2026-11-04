"""JSON GET with a hard wall-clock cap, for the unofficial match-data APIs.

requests' own timeout only bounds each socket read, so a reply that trickles in
(or a server that holds the connection) can outlast a pull's --budget-min and
the workflow's step timeout. The request runs on a daemon thread and is
abandoned after `max_s` seconds instead.
"""

from __future__ import annotations

import threading

import requests


def run_capped(fn, max_s: float, what: str = "request"):
    """fn() on a daemon thread, abandoned after max_s seconds - for any HTTP
    client (requests, tls_requests) whose own timeout can't bound a whole call.
    Raises requests.Timeout on the cap, re-raises whatever fn raised."""
    box: dict = {}

    def run() -> None:
        try:
            box["v"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(max_s)
    if t.is_alive():
        raise requests.Timeout(f"{what}: no full reply after {max_s}s")
    if "e" in box:
        raise box["e"]
    return box["v"]


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             max_s: float = 45) -> object:
    """Raises requests.RequestException on an HTTP error, timeout, dropped
    connection or non-JSON reply - callers skip that item and retry next run."""
    def fetch():
        r = requests.get(url, params=params, headers=headers, timeout=(10, 20))
        r.raise_for_status()
        return r.json()

    try:
        return run_capped(fetch, max_s, url)
    except requests.RequestException:
        raise
    except Exception as e:  # noqa: BLE001 - bad JSON etc.
        raise requests.RequestException(f"{url}: {e}") from e

"""JSON GET with a hard wall-clock cap, for the unofficial match-data APIs.

requests' own timeout only bounds each socket read, so a reply that trickles in
(or a server that holds the connection) can outlast a pull's --budget-min and
the workflow's step timeout. The request runs on a daemon thread and is
abandoned after `max_s` seconds instead.
"""

from __future__ import annotations

import threading

import requests


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             max_s: float = 45) -> object:
    """Raises requests.RequestException on an HTTP error, timeout, dropped
    connection or non-JSON reply - callers skip that item and retry next run."""
    box: dict = {}

    def run() -> None:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=(10, 20))
            r.raise_for_status()
            box["v"] = r.json()
        except Exception as e:  # noqa: BLE001 - re-raised below as a RequestException
            box["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(max_s)
    if t.is_alive():
        raise requests.Timeout(f"{url}: no full reply after {max_s}s")
    if "e" in box:
        e = box["e"]
        raise e if isinstance(e, requests.RequestException) else requests.RequestException(f"{url}: {e}")
    return box["v"]

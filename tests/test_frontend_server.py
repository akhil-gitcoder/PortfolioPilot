"""Tests for the web backend: start the real server on a free port and call it like the page does."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from frontend.server import create_server


@pytest.fixture(scope="module")
def server():
    srv, app = create_server(port=0, n_evals=1500)       # port 0 = any free port, small budget = fast
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", app
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.status, r.read()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_context(server):
    base, app = server
    code, html = get(base + "/")
    assert code == 200 and b"PortfolioPilot" in html and b"/api/rebalance" in html
    code, ctx = get(base + "/api/context")
    ctx = json.loads(ctx)
    assert len(ctx["holdings"]) == 30 and abs(sum(h["pct"] for h in ctx["holdings"]) - 100) < 0.2
    assert ctx["sources"][0]["id"] == "bundled"


def test_evidence_reads_project_results(server):
    base, _ = server
    ev = json.loads(get(base + "/api/evidence")[1])
    assert ev["available"] and ev["wins"] == 30 and ev["seeds"] == 30
    assert {r["m"] for r in ev["rows"]} >= {"WeightedSum", "Hybrid+archive"}


def test_rebalance_returns_consistent_options(server):
    base, app = server
    ctx = json.loads(get(base + "/api/context")[1])
    code, res = post(base + "/api/rebalance", {"holdings": [h["pct"] for h in ctx["holdings"]], "source": "bundled"})
    assert code == 200 and set(res["byk"]) == {"3", "4", "5", "6"}
    for k, R in res["byk"].items():
        opts = R["opts"]
        assert 3 <= len(opts) <= int(k)
        vols = [o["vol"] for o in opts]
        assert vols == sorted(vols)
        assert opts[0]["label"] == "Safest" and opts[-1]["label"] == "Highest return"
        for o in opts:
            assert "{INR}" in o["text"]["vs_now"]
            assert abs(sum(t['sug'] for t in o['trades']) - 1.0) < 1e-3     # new weights add to 100%
            assert o["ret"] > R["cur"]["ret"] or o["vol"] < R["cur"]["vol"]
    # same request again is served from the cache and is identical
    code2, res2 = post(base + "/api/rebalance", {"holdings": [h["pct"] for h in ctx["holdings"]], "source": "bundled"})
    assert res2 == res


@pytest.mark.parametrize("body", [
    {"holdings": [1, 2, 3]},                                  # wrong length
    {"holdings": ["x"] * 30},                                 # not numbers
    {"holdings": [-5] + [0] * 29},                            # negative
    {"holdings": [0] * 30},                                   # nothing held
    {"holdings": [1] * 30, "source": "nonsense"},             # unknown source
    {"holdings": [1000] + [0] * 29},                          # more than 100%
])
def test_bad_input_is_rejected_cleanly(server, body):
    base, _ = server
    code, res = post(base + "/api/rebalance", body)
    assert code == 400 and res["error"]


def test_latest_without_download_is_a_clear_error(server):
    base, app = server
    if app.latest() is not None:
        pytest.skip("a downloaded price file exists on this machine")
    code, res = post(base + "/api/rebalance", {"holdings": [100 / 30] * 30, "source": "latest"})
    assert code == 400 and "Download" in res["error"]


def test_unknown_route_and_large_body(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + "/nope")
    assert e.value.code == 404
    code, res = post(base + "/api/rebalance", {"holdings": [1] * 30, "pad": "x" * 200_000})
    assert code == 413

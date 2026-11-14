import os
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from frontend import engine
from frontend.server import App

if os.environ.get("VERCEL"):
    engine.CACHE_DIR = Path("/tmp/portfoliopilot")
    engine.LATEST_PRICES = engine.CACHE_DIR / "prices_latest.csv"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100_000

portfolio = App()
PAGE = Path(__file__).resolve().parent / "frontend" / "static" / "index.html"


@app.after_request
def prevent_stale_responses(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def home():
    return send_file(PAGE)


@app.get("/api/context")
def context():
    return jsonify(portfolio.context())


@app.get("/api/evidence")
def evidence():
    return jsonify(portfolio.evidence())


@app.post("/api/rebalance")
def rebalance():
    body = request.get_json()
    if not isinstance(body, dict):
        return jsonify(error="Expected a JSON object."), 400

    if len(portfolio.cache) >= 20:
        portfolio.cache.clear()

    return jsonify(
        portfolio.rebalance(
            body.get("holdings"),
            body.get("source", "bundled"),
        )
    )


@app.post("/api/refresh")
def refresh():
    return jsonify(portfolio.refresh())


@app.errorhandler(ValueError)
def invalid_input(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(HTTPException)
def http_error(error):
    return jsonify(error=error.description), error.code


@app.errorhandler(Exception)
def server_error(error):
    app.logger.exception("PortfolioPilot request failed")
    return jsonify(error="The request failed. Check deployment logs."), 500

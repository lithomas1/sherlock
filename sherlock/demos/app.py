from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sherlock.demos.twitter as tw

public = Path(__file__).parent / "client/public"

app = FastAPI()
app.mount("/public", StaticFiles(directory=f"{public}"), name="pub")


@app.get("/verify")
def verify(claim: str):
    return {
        "agree": [
            ("the left something", 0.987),
            ("the right something", 0.912),
            ("another agree", 0.812),
        ],
        "disagree": [
            ("a based wiki article", 0.912),
            ("another based article", 0.71),
            ("yippee another", 0.61),
        ],
    }


@app.get("/claims")
def claims(username: str):
    return tw.get_claims(username)


@app.get("/")
def root():
    return FileResponse(f"{public / 'index.html'}")

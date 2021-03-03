from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sherlock.demos.twitter as tw
from sherlock.models import BaseModel, Verification
from sherlock.models.lstm import LSTMModel

public = Path(__file__).parent / "client/public"

app = FastAPI()
app.mount("/public", StaticFiles(directory=f"{public}"), name="pub")

@app.on_event("startup")
def create_verifier() -> None:
    # TODO: Load dataset
    app.verifier: BaseModel = LSTMModel().eval()
    app.verifier.sherlock_load()

@app.get("/verify", response_model=Verification)
def verify(claim: str):
    return app.verifier.verify(claim)


@app.get("/claims")
def claims(username: str):
    return tw.get_claims(username)


@app.get("/")
def root():
    return FileResponse(f"{public / 'index.html'}")

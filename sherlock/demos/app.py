from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

public = Path(__file__).parent / "public"

app = FastAPI()
app.mount("/public", StaticFiles(directory=f"{public}"), name="pub")


@app.get("/")
def root():
    return FileResponse(f"{public / 'index.html'}")

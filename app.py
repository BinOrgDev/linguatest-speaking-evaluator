import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel

app = FastAPI()
model = WhisperModel("base", device="cpu")
private_key = "60d9f8de-96d7-4163-84ee-2351db4f5e3f"


@app.post("/transcribe/{key}")
async def transcribe(key: str, file: UploadFile = File(...)):
    if key != private_key:
        raise HTTPException(status_code=403, detail="You don't have access to this resource")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    segments, _ = model.transcribe(tmp_path, beam_size=5)
    transcript = "".join([seg.text for seg in segments])
    return {"transcript": transcript}

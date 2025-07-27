import json
import os
import tempfile
import openai

from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel
from openai import OpenAI

from evaluation import EvaluationRequest
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


@app.post("/evaluate")
async def evaluate_speech(req: EvaluationRequest):
    system_prompt = """
    You are an IELTS speaking examiner and English teacher.

    Given a student's spoken answer (transcribed) to an IELTS-style question, do the following:

    1. Evaluate it using IELTS criteria (fluency, grammar, vocabulary, pronunciation).
    2. Score each component (1–9 scale) and compute an overall score.
    3. Analyze the answer word-by-word and return an array:
       - word
       - error (true/false)
       - issue (e.g., grammar, word choice) or null
       - correction (if applicable)
       - reason (brief explanation)
    4. Assess transcript accuracy (consider ASR errors).
    5. Provide a model Band 9 answer to the same question.

    ⚠️ Output must be a valid JSON ONLY in this exact format:

    {
      "band_score": {
        "overall": 6.5,
        "fluency": 6,
        "grammar": 7,
        "vocabulary": 6,
        "pronunciation": 7
      },
      "word_analysis": [
        {
          "word": "I",
          "error": false,
          "issue": null,
          "correction": null,
          "reason": null
        },
        {
          "word": "has",
          "error": true,
          "issue": "Verb agreement",
          "correction": "have",
          "reason": "Use 'have' with first person singular"
        }
      ],
      "correctness_of_transcript": "Mostly accurate with minor ASR issues.",
      "suggested_answer": "I have lived in my hometown all my life. It's a peaceful and historic place."
    }
        """

    user_prompt = f"""
    Question: {req.question}
    Transcript: {req.transcription}
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
        )
        content = response.choices[0].message.content.strip()

        content = response.choices[0].message.content.strip()

        if not content:
            return {"error": "Empty response from OpenAI"}

        # Clean code block formatting
        import re
        cleaned = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.DOTALL)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON from OpenAI: {str(e)}",
                "raw": content
            }
    except Exception as e:
        return {"error": str(e)}

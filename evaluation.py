from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    question: str
    transcription: str

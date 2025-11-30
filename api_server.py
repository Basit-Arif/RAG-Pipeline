from fastapi import FastAPI
from pydantic import BaseModel
from hybrid_qa import HybridQAPipeline

app = FastAPI()

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.post("/ask")
def ask(question: Question):
    return {"answer": HybridQAPipeline().ask(question.question).answer}
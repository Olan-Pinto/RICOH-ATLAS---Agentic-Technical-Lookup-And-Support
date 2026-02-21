# nodes/analyzer.py
from openai import OpenAI
from pydantic import BaseModel
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

class AnalyzerOutput(BaseModel):
    confidence_score: int    # 0-10
    sufficient: bool
    missing: str             
    reasoning: str

ANALYZER_PROMPT = """You are an evidence quality checker for a technical support agent.

Score the evidence from 0-10 using these anchors:
- 10: Evidence completely answers the question with no gaps
- 7:  Evidence covers the main answer but misses minor details
- 5:  Evidence is relevant but only partially answers the question
- 3:  Evidence is loosely related but does not answer the question
- 0:  Evidence is completely irrelevant or missing

Set sufficient = true if confidence_score >= 7, otherwise false.
If not sufficient, describe exactly what is missing."""

def run_analyzer(state: dict) -> dict:
    chunks_text = "\n\n".join(
        f"[{c['doc']} | {c['section']} | p.{c['page']}]\n{c['text']}"
        for c in state["retrieved_chunks"]
    )
    response = client.beta.chat.completions.parse(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": ANALYZER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n"
                    f"Intent: {state['intent']}\n\n"
                    f"Evidence:\n{chunks_text}"
                ),
            },
        ],
        response_format=AnalyzerOutput,
        temperature=0,
    )
    parsed: AnalyzerOutput = response.choices[0].message.parsed
    state["confidence_score"]    = parsed.confidence_score
    state["evidence_sufficient"] = parsed.sufficient
    state["missing_info"]        = parsed.missing
    return state

# verifier.py
from openai import OpenAI
from pydantic import BaseModel
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

class VerifierOutput(BaseModel):
    verified_answer:  str
    uncertainty_flag: bool
    flagged_claims:   list[str]  # claims that were removed or flagged

VERIFIER_PROMPT = """You are a strict claim verifier for a technical support system.

Given a draft answer and the source chunks it was based on:
1. Check every factual claim — is it supported by the chunks?
2. Remove or flag any claim that is NOT directly supported
3. If critical information is missing, add an explicit
   " Note: [topic] was not found in the available documentation."

Set uncertainty_flag = true if any claims were removed or information was missing."""

def run_verifier(state: dict) -> dict:
    chunks_text = "\n\n".join(
        f"[{c['doc']} | {c['section']} | p.{c['page']}]\n{c['text']}"
        for c in state["retrieved_chunks"]
    )
    response = client.beta.chat.completions.parse(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": VERIFIER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Draft answer:\n{state['draft_answer']}\n\n"
                    f"Source chunks:\n{chunks_text}"
                ),
            },
        ],
        response_format=VerifierOutput,
        temperature=0,
    )
    parsed: VerifierOutput = response.choices[0].message.parsed
    state["final_answer"]    = parsed.verified_answer
    state["uncertainty_flag"] = parsed.uncertainty_flag
    return state

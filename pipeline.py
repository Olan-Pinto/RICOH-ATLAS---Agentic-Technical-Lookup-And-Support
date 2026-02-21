# pipeline.py
from index_loader import load_index
from nodes.planner import run_planner
from nodes.retriever import run_retriever
from nodes.analyzer import run_analyzer
from nodes.query_refiner import run_query_refiner
from nodes.synthesizer import run_synthesizer
from verifier import run_verifier
from config import MAX_ITERATIONS


def run_pipeline_streaming(question: str, index):
    """
    Runs each agent node manually and yields state after each one
    so the UI can show live progress.
    """

    # --- Initial State --- 
    state = {
        "question":            question,
        "intent":              "",
        "subquestions":        [],
        "retrieved_chunks":    [],
        "evidence_sufficient": False,
        "confidence_score":    0,
        "missing_info":        "",
        "iteration_count":     0,
        "draft_answer":        "",
        "final_answer":        "",
        "uncertainty_flag":    False,
    }

    # ── NODE 1: PLANNER ──────────────────────────────────────────
    state = run_planner(state)
    yield {**state, "stage": "planner"}

    # ── NODE 2 + 3 + 4 LOOP: RETRIEVER → ANALYZER → REFINER ─────
    while True:

        # Node 2 — Retriever
        state = run_retriever(state, index)
        yield {
            **state,
            "stage":       "retriever",
            "chunk_count": len(state["retrieved_chunks"]),
        }

        # Node 3 — Analyzer
        state = run_analyzer(state)
        yield {
            **state,
            "stage":     "analyzer",
            "sufficient": state["evidence_sufficient"],
        }

        # Exit loop if evidence is good enough or max iterations hit
        if state["evidence_sufficient"] or state["iteration_count"] >= MAX_ITERATIONS:
            break

        # Node 4 — Query Refiner (only runs if evidence insufficient)
        state = run_query_refiner(state)
        yield {**state, "stage": "refiner"}

    # ── NODE 5: SYNTHESIZER ──────────────────────────────────────
    state = run_synthesizer(state)
    yield {**state, "stage": "synthesizer"}

    # ── POST-LOOP: VERIFIER ──────────────────────────────────────
    state = run_verifier(state)
    yield {**state, "stage": "verifier"}

"""
All LLM prompts used by the pipeline, in one place so they can be audited.

Design principles (from RAGalyst Nov 2025):
  - Split QA generation into TWO separate calls: question then answer.
  - Answer prompt is strictly grounding-only; if answer cannot be derived,
    return NOT_ANSWERABLE.
  - Each judge metric has its own prompt. We ask for a JSON response so
    parsing is deterministic.
"""


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION PROMPTS (Stage A — Qwen2.5-72B-Instruct-AWQ)
# ══════════════════════════════════════════════════════════════════════════════

QUESTION_GEN_SYSTEM = """You are a question-writing expert for a technical \
RAG evaluation dataset. You produce ONE well-formed question per call, in \
clean, formal technical English.

Strict rules:
1. The question must be answerable using ONLY the provided context.
2. The question must be self-contained: a reader who does not have the \
context should still understand what is being asked.
3. Do NOT reference figure numbers, document IDs, page numbers, or section \
numbers in the question unless those artifacts are themselves the topic.
4. Do NOT ask "what does the document say about X" — ask directly about X.
5. Do not write casual or informal English. Do not use abbreviations like \
"wat" or "iz". Use full technical terms.
6. The question must match the requested TYPE and PERSONA.

Respond ONLY with valid JSON in this exact format:
{"question": "<the question>"}
No preamble, no markdown, no code fences."""


QUESTION_GEN_USER_SINGLEHOP = """Context:
<<<
{context}
>>>

Persona: {persona_role}

Question type: {question_type}

Write one question per the rules. {style_hint}"""


QUESTION_GEN_USER_MULTIHOP = """You are given TWO or more related context \
chunks. Write ONE question that REQUIRES information from AT LEAST TWO of \
the chunks to answer correctly. A question answerable from a single chunk \
is INVALID.

Contexts:
{contexts_block}

Persona: {persona_role}

Question type: {question_type}

Write one question per the rules. {style_hint}"""


STYLE_HINTS = {
    "specific": (
        "The question should ask about a specific fact, requirement, value, "
        "mechanism, or definition present in the context."
    ),
    "abstract": (
        "The question should require synthesis or explanation of a concept "
        "rather than a single fact lookup, while still being grounded in the "
        "context."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────


ANSWER_GEN_SYSTEM = """You are an expert answer-writer for a technical RAG \
evaluation dataset. You produce ONE ground-truth answer grounded strictly \
in the provided context.

Strict rules:
1. Use ONLY the information in the provided context. Do NOT add information \
from general knowledge, even if it would be correct.
2. If the context does NOT contain enough information to answer the \
question fully, respond with exactly: {"answer": "NOT_ANSWERABLE"}
3. The answer should be complete but concise: 1–4 sentences for specific \
questions, up to 6 sentences for abstract ones.
4. Do NOT pad the answer by rephrasing the question.
5. Do NOT say "according to the context" or "as stated in section X". Just \
give the factual answer.
6. Write in clean, formal technical English.

Respond ONLY with valid JSON in this exact format:
{"answer": "<the answer>"}
No preamble, no markdown, no code fences."""


ANSWER_GEN_USER = """Context:
<<<
{context}
>>>

Question: {question}

Write the answer per the rules."""


# ══════════════════════════════════════════════════════════════════════════════
# JUDGE PROMPTS (Stage B — Qwen3-30B-A3B-Instruct-2507)
# ══════════════════════════════════════════════════════════════════════════════
#
# These follow the RAGalyst-validated prompt patterns. Each returns a small
# JSON object with a numeric score and a short rationale.
# ══════════════════════════════════════════════════════════════════════════════

ANSWERABILITY_SYSTEM = """You are a strict evaluator. You will be given a \
context and a question. Your job is to determine if the question is \
UNAMBIGUOUSLY ANSWERABLE using ONLY the given context.

Rules:
- If the context contains all information needed to answer the question \
without making any assumption or using any outside knowledge, return 1.
- If any key information is missing, the question is ambiguous, or the \
answer would require inference beyond what is stated, return 0.
- Do NOT use prior knowledge. Judge purely on whether the context supports \
a confident answer.

Respond ONLY with valid JSON in this format:
{"answerability": 0 or 1, "rationale": "<one short sentence>"}"""


ANSWERABILITY_USER = """Context:
<<<
{context}
>>>

Question: {question}

Is this question unambiguously answerable from ONLY this context?"""


# ──────────────────────────────────────────────────────────────────────────────


FAITHFULNESS_SYSTEM = """You are a strict evaluator measuring FAITHFULNESS: \
whether every factual claim in the given answer is supported by the given \
context.

Rules:
- Decompose the answer into individual factual claims (mentally).
- For each claim, check if it is directly supported by the context.
- A claim supported by the context counts as faithful. A claim that goes \
beyond what the context says, even if it is true in general, is unfaithful.
- Return the fraction of claims that are faithful, rounded to 2 decimals.
- 1.0 = every claim is supported. 0.0 = none is supported.

Respond ONLY with valid JSON in this format:
{"faithfulness": <float between 0 and 1>, "rationale": "<one short sentence>"}"""


FAITHFULNESS_USER = """Context:
<<<
{context}
>>>

Answer:
<<<
{answer}
>>>

What fraction of the factual claims in the answer are supported by the context?"""


# ──────────────────────────────────────────────────────────────────────────────


ANSWER_RELEVANCE_SYSTEM = """You are a strict evaluator measuring \
ANSWER RELEVANCE: whether the given answer directly and meaningfully \
addresses the given question.

Rules:
- An answer that directly addresses the question scores high.
- An answer that is only tangentially related, or that addresses a different \
question, scores low.
- Ignore whether the answer is correct — only judge whether it addresses \
the question asked.
- Score on a continuous scale from 0 (completely off-topic) to 1 (directly \
and fully addresses the question).

Respond ONLY with valid JSON in this format:
{"answer_relevance": <float between 0 and 1>, "rationale": "<one short sentence>"}"""


ANSWER_RELEVANCE_USER = """Question: {question}

Answer: {answer}

How directly does the answer address the question?"""


# ──────────────────────────────────────────────────────────────────────────────


QUESTION_SPECIFICITY_SYSTEM = """You are a strict evaluator measuring \
QUESTION SPECIFICITY: whether the question is self-contained and specific \
enough to be answered without the original context.

Rules:
- A specific, self-contained question (e.g. "What is the role of BswM in \
AUTOSAR mode management?") scores 1.
- A vague or context-dependent question (e.g. "What is this about?", \
"What does it do?", "What is mentioned in the document?") scores 0.
- A question that references figures, section numbers, or document IDs \
without context (e.g. "What does Figure 46 show?") scores 0.
- A question that an AUTOSAR engineer could understand and attempt to \
answer without seeing the source scores 1.

Respond ONLY with valid JSON in this format:
{"question_specificity": 0 or 1, "rationale": "<one short sentence>"}"""


QUESTION_SPECIFICITY_USER = """Question: {question}

Is this question specific and self-contained?"""

"""
AUTOSAR-Aware Question Generation for RAG Evaluation
Generates difficulty-stratified questions based on document content type and question complexity.

Document-side complexity:
  - Pure prose sections              → easiest
  - Mixed prose + requirement IDs    → medium
  - Tables / parameter lists         → hardest
  - Cross-document references        → hardest

Question-side complexity:
  - Lookup questions                 → easy
  - Parameter/value extraction       → medium
  - Relationship questions           → hard
  - Requirement tracing              → hardest
"""

import os
import re
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import PyPDF2
import pdfplumber
from tqdm import tqdm
import time
import hashlib
import numpy as np

# ==================================================
# OLLAMA CONFIGURATION
# ==================================================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "qwen2.5:72b")

print(f"🤖 Using Ollama Model : {OLLAMA_MODEL}")
print(f"🌐 Ollama URL         : {OLLAMA_BASE_URL}")

# ==================================================
# GENERAL CONFIGURATION
# ==================================================
MAX_RETRIES    = 3
RETRY_DELAY    = 2          # seconds
CHUNK_SIZE     = 10000      # characters — kept smaller so content-type detection is accurate
CHUNK_OVERLAP  = 1500

# Semantic dedup (optional — needs bge-m3 pulled in Ollama)
ENABLE_SEMANTIC_DEDUP        = os.environ.get("ENABLE_SEMANTIC_DEDUP", "false").lower() == "true"
SEMANTIC_SIMILARITY_THRESHOLD = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD", "0.85"))

# ==================================================
# QUESTION BUDGET CONFIGURATION
# How many questions of each difficulty to generate PER DOCUMENT.
# Override from the command line or edit here.
# ==================================================
DEFAULT_QUESTION_BUDGET = {
    "easy":    1,   # Lookup questions from pure prose
    "medium":  2,   # Parameter/value extraction
    "hard":    2,   # Relationship / multi-hop
    "hardest": 1,   # Requirement tracing / cross-document
}


# ──────────────────────────────────────────────────────────────────────────────
# CONTENT TYPE CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

class ContentTypeClassifier:
    """
    Classifies a text chunk into one of four AUTOSAR content categories:
      - prose              : plain explanatory text, no special structure
      - mixed_requirements : prose mixed with SWS/RS/SRS requirement IDs
      - parameter_table    : primarily parameter/configuration tables
      - cross_reference    : heavy use of cross-document references
    """

    # Requirement ID patterns common in AUTOSAR documents
    REQ_ID_PATTERN      = re.compile(
        r'\b(SWS|RS|SRS|ECUC|CONSTR|TPS|MOD|BSW|RTE|COM|NM|FIM|DEM|DCM|FEE|NVM|MEM|OS|WDGM|BSWM|PDUR|SOAD|DOIP|SD|ETHIF|TCPIP)_\w+\b'
    )
    # Cross-document reference patterns
    CROSS_REF_PATTERN   = re.compile(
        r'\b(AUTOSAR_SWS|AUTOSAR_RS|AUTOSAR_TPS|AUTOSAR_MOD|AUTOSAR_EXP|refer to|see \[|as defined in|according to)\b',
        re.IGNORECASE
    )
    # Table indicators from pdfplumber serialization
    TABLE_ROW_PATTERN   = re.compile(r'\|.*\|')
    # Parameter-list indicators
    PARAM_PATTERN       = re.compile(
        r'\b(uint8|uint16|uint32|sint8|sint16|sint32|boolean|float|EcucNumericalParamValue|EcucTextualParamValue|Multiplicity|upperMultiplicity|lowerMultiplicity|ValidValues|DefaultValue)\b',
        re.IGNORECASE
    )

    @classmethod
    def classify(cls, text: str) -> str:
        """Return content type label for this chunk."""
        total_chars  = max(len(text), 1)
        lines        = text.splitlines()
        total_lines  = max(len(lines), 1)

        req_ids      = len(cls.REQ_ID_PATTERN.findall(text))
        cross_refs   = len(cls.CROSS_REF_PATTERN.findall(text))
        table_rows   = sum(1 for l in lines if cls.TABLE_ROW_PATTERN.search(l))
        param_hits   = len(cls.PARAM_PATTERN.findall(text))

        table_ratio  = table_rows / total_lines
        req_density  = req_ids    / (total_chars / 1000)   # IDs per 1 k chars
        cross_density= cross_refs / (total_chars / 1000)

        # Decision rules (order matters)
        if cross_density >= 1.5 or cross_refs >= 5:
            return "cross_reference"
        if table_ratio >= 0.25 or param_hits >= 6:
            return "parameter_table"
        if req_density >= 2.0 or req_ids >= 4:
            return "mixed_requirements"
        return "prose"

    @classmethod
    def describe(cls, content_type: str) -> str:
        return {
            "prose":              "Pure explanatory prose — no formal requirement IDs or tables.",
            "mixed_requirements": "Prose mixed with formal AUTOSAR requirement IDs (SWS_, RS_, etc.).",
            "parameter_table":    "Primarily ECUC parameter definitions, value tables, or configuration lists.",
            "cross_reference":    "Heavy cross-document references to other AUTOSAR specifications.",
        }.get(content_type, "Unknown content type.")


# ──────────────────────────────────────────────────────────────────────────────
# TABLE SERIALIZER  (converts pdfplumber raw table → readable prose)
# ──────────────────────────────────────────────────────────────────────────────

class TableSerializer:
    """Converts raw table data into LLM-readable structured prose."""

    @staticmethod
    def to_markdown(table: List[List]) -> str:
        """Serialize a table as markdown (preferred for LLMs)."""
        if not table or not table[0]:
            return ""
        lines = []
        header = [str(c).strip() if c else "" for c in table[0]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in table[1:]:
            cells = [str(c).strip() if c else "" for c in row]
            # Pad row to header width
            while len(cells) < len(header):
                cells.append("")
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    @staticmethod
    def to_prose(table: List[List]) -> str:
        """
        For parameter tables, serialize as structured prose so the LLM
        understands column relationships even without visual layout.
        e.g.  Parameter 'BswModuleEntry': Type=EcucParamConfContainerDef,
              Multiplicity=1..*, ValidValues=N/A, DefaultValue=N/A
        """
        if not table or not table[0]:
            return ""
        header = [str(c).strip() if c else "Field" for c in table[0]]
        rows_text = []
        for row in table[1:]:
            cells = [str(c).strip() if c else "" for c in row]
            parts = [f"{h}={v}" for h, v in zip(header, cells) if v]
            if parts:
                rows_text.append("  • " + ", ".join(parts))
        return "\n".join(rows_text)


# ──────────────────────────────────────────────────────────────────────────────
# PDF CONTENT EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

class PDFContentExtractor:
    """Extract text, tables, and structure from AUTOSAR PDFs."""

    @staticmethod
    def intelligent_chunking(text: str,
                             max_chunk_size: int = CHUNK_SIZE,
                             overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
        """
        Structure-aware chunking.
        Tries to split at AUTOSAR section headings before falling back to
        page boundaries, then character count.
        """
        chunks = []

        # Split by pages first
        pages = text.split('[Page ')
        current_chunk = ""
        current_pages: List[int] = []

        for page in pages:
            if not page.strip():
                continue
            try:
                page_num     = int(page.split(']')[0])
                page_content = ']'.join(page.split(']')[1:])
            except (ValueError, IndexError):
                page_content = page
                page_num     = (current_pages[-1] + 1) if current_pages else 1

            if len(current_chunk) + len(page_content) > max_chunk_size and current_chunk:
                chunks.append({
                    'text':       current_chunk.strip(),
                    'pages':      current_pages.copy(),
                    'char_count': len(current_chunk),
                })
                # Overlap: last few sentences
                sentences    = current_chunk.split('.')
                overlap_text = '.'.join(sentences[-5:]) if len(sentences) > 5 else current_chunk[-overlap:]
                current_chunk = overlap_text + f"\n[Page {page_num}]{page_content}"
                current_pages = [page_num]
            else:
                current_chunk += f"\n[Page {page_num}]{page_content}"
                if page_num not in current_pages:
                    current_pages.append(page_num)

        if current_chunk.strip():
            chunks.append({
                'text':       current_chunk.strip(),
                'pages':      current_pages.copy(),
                'char_count': len(current_chunk),
            })
        return chunks

    @staticmethod
    def extract_with_pdfplumber(pdf_path: str) -> Optional[Dict[str, Any]]:
        """Extract text and tables; serialize tables as markdown + prose."""
        content = {
            "pages":     [],
            "full_text": "",
            "tables":    [],
            "metadata":  {},
            "chunks":    [],
        }
        try:
            with pdfplumber.open(pdf_path) as pdf:
                content["metadata"] = {
                    "num_pages": len(pdf.pages),
                    "filename":  os.path.basename(pdf_path),
                }
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_data = {"page_number": page_num, "text": "", "tables": []}

                    # Text
                    text = page.extract_text()
                    if text:
                        page_data["text"]     = text.strip()
                        content["full_text"] += f"\n[Page {page_num}]\n{text}\n"

                    # Tables — serialized as markdown AND prose for LLM readability
                    tables = page.extract_tables()
                    if tables:
                        for t_idx, table in enumerate(tables):
                            if not table:
                                continue
                            table_data = {
                                "page":     page_num,
                                "table_id": f"page{page_num}_table{t_idx+1}",
                                "data":     table,
                            }
                            page_data["tables"].append(table_data)
                            content["tables"].append(table_data)

                            # Add BOTH representations to full_text
                            md_table = TableSerializer.to_markdown(table)
                            prose_table = TableSerializer.to_prose(table)
                            content["full_text"] += (
                                f"\n[Table on Page {page_num}]\n"
                                f"{md_table}\n"
                                f"[Table Summary]\n{prose_table}\n"
                            )
                    content["pages"].append(page_data)

                content["chunks"] = PDFContentExtractor.intelligent_chunking(content["full_text"])

        except Exception as e:
            print(f"  ❌ pdfplumber error for {pdf_path}: {e}")
            return None
        return content

    @staticmethod
    def extract_with_pypdf2(pdf_path: str) -> Optional[str]:
        """Fallback extraction via PyPDF2."""
        try:
            text = ""
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num}]\n{page_text}\n"
            return text
        except Exception as e:
            print(f"  ❌ PyPDF2 error for {pdf_path}: {e}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# AUTOSAR PROMPT LIBRARY
# ──────────────────────────────────────────────────────────────────────────────

class AutosarPromptLibrary:
    """
    Builds fully explicit, long-form prompts for each difficulty × content-type
    combination. Designed so the LLM has zero ambiguity about what to produce.

    Each prompt contains:
      1. Domain context  — AUTOSAR conventions, normative language, module names
      2. Content-type briefing — what kind of text the model is reading
      3. Difficulty contract  — precise definition, good examples, bad examples,
                                answer format, self-validation checklist
      4. Output format spec  — field-by-field explanation with types and constraints
      5. Chain-of-thought gate — model must reason BEFORE writing JSON
      6. Final output gate    — strict JSON only after reasoning
    """

    # ──────────────────────────────────────────────────────────────────────────
    # SHARED BLOCKS — injected into every prompt
    # ──────────────────────────────────────────────────────────────────────────

    _DOMAIN_CONTEXT = """\
═══════════════════════════════════════════════════════════════════════════════
SECTION 1 — WHO YOU ARE AND WHAT THIS TASK IS
═══════════════════════════════════════════════════════════════════════════════

You are a senior AUTOSAR technical writer and systems architect with deep expertise
in the AUTOSAR Classic Platform. Your task is to generate evaluation questions and
ground-truth answers for a Retrieval-Augmented Generation (RAG) benchmark dataset.

Every question you generate will later be used to test whether a RAG system can
correctly retrieve and answer questions about AUTOSAR specifications. The quality
of the entire benchmark depends on the precision and correctness of what you
produce here.

═══════════════════════════════════════════════════════════════════════════════
SECTION 2 — AUTOSAR DOCUMENT CONVENTIONS YOU MUST APPLY
═══════════════════════════════════════════════════════════════════════════════

2.1  NORMATIVE LANGUAGE (RFC 2119 as used in AUTOSAR)
     ┌─────────────┬──────────────────────────────────────────────────────────┐
     │  Keyword    │  Meaning                                                 │
     ├─────────────┼──────────────────────────────────────────────────────────┤
     │  shall      │  Mandatory. Non-compliance is a specification violation. │
     │  shall not  │  Prohibited. Non-compliance is a specification violation.│
     │  should     │  Recommended but not mandatory.                          │
     │  should not │  Not recommended but not prohibited.                     │
     │  may        │  Optional. Implementer's choice.                         │
     └─────────────┴──────────────────────────────────────────────────────────┘
     When quoting requirement text in answers, always preserve the exact
     normative keyword. Do NOT paraphrase "shall" as "must" or "needs to".

2.2  REQUIREMENT ID PATTERNS
     Every normative statement in AUTOSAR has a unique ID. Patterns you will
     encounter:
       SWS_<Module>_<5-digit-number>   — Software Specification requirement
       RS_<Feature>_<5-digit-number>   — Requirements Specification item
       SRS_<Area>_<5-digit-number>     — Software Requirements Specification
       CONSTR_<number>                 — Design constraint
       TPS_<number>                    — Template specification item
     Examples: SWS_Com_00012, RS_BswM_00003, CONSTR_00456

2.3  ECUC PARAMETER TABLE STRUCTURE
     AUTOSAR configuration parameter tables always have these columns (though
     naming may vary slightly between documents):
       • Parameter Name / Short Name  — identifier used in code/tools
       • Category / Type              — e.g. EcucNumericalParamValue,
                                        EcucTextualParamValue, EcucBooleanParamValue
       • Multiplicity (lower..upper)  — e.g. 0..1, 1, 0..*
       • Valid Values / Range         — numeric range, enumeration, or regexp
       • Default Value                — value used when not configured
       • Description / Scope          — human-readable explanation
     When reading a table, always identify which column is which BEFORE
     extracting a value. Never confuse Range with Default Value.

2.4  CROSS-DOCUMENT REFERENCE PATTERNS
     Documents reference each other using bracket notation or inline phrases:
       [AUTOSAR_SWS_BSWGeneral]
       [AUTOSAR_RS_CommunicationStack]
       "as defined in AUTOSAR_SWS_NvM"
       "refer to [TPS_MMOD_00123]"
     A cross-reference means the behavior or definition lives in another
     document. Only answer questions about what THIS document says about
     the reference — do NOT invent content from the referenced document.

2.5  COMMON AUTOSAR MODULES (for context — use only what appears in the text)
     Communication : COM, PDU Router (PduR), CAN Interface (CanIf),
                     SoAd, DoIP, EthIf, TCPIP, SD
     Network Mgmt  : NM, CanNM, UdpNm, FrNm
     Diagnostics   : DEM (Diagnostic Event Manager), DCM (Diagnostic Comm. Mgr),
                     FiM (Function Inhibition Manager)
     Memory        : NvM, MemIf, Fee, Fls, Ea, Eep
     System        : OS, EcuM (ECU State Manager), BswM (BSW Mode Manager),
                     WdgM (Watchdog Manager), Det (Default Error Tracer)
     RTE / Composition: RTE, SchM (BSW Scheduler), Port Interface, SW-C

═══════════════════════════════════════════════════════════════════════════════
SECTION 3 — OUTPUT FORMAT (READ THIS CAREFULLY)
═══════════════════════════════════════════════════════════════════════════════

You must output a JSON array. Each element is one question object with EXACTLY
these fields — no extra fields, no missing fields:

  {
    "question"        : string  — The full, self-contained question text.
                                  A reader who has NOT seen the document must
                                  understand what is being asked. Include module
                                  names, parameter names, or requirement IDs
                                  where needed for clarity.

    "answer"          : string  — The complete ground-truth answer.
                                  Must be entirely derivable from the document
                                  content provided below. Must cite the specific
                                  source (page number, section heading, table
                                  name, or requirement ID). For parameter
                                  questions: state the exact value including
                                  units or data type. For requirement tracing:
                                  quote the normative keyword and requirement ID.

    "question_type"   : string  — MUST be exactly one of:
                                  "lookup" | "parameter_extraction" |
                                  "relationship" | "requirement_tracing"

    "difficulty"      : string  — MUST match the requested difficulty exactly:
                                  "easy" | "medium" | "hard" | "hardest"

    "content_type"    : string  — The content type of the source chunk.
                                  Will be filled in by the system; set to the
                                  value stated in Section 4 below.

    "reasoning_steps" : string  — For easy/medium: write "Direct lookup from
                                  [location]." For hard/hardest: write a
                                  numbered chain showing exactly how you moved
                                  from evidence piece 1 → evidence piece 2 →
                                  final answer. Minimum 2 steps for hard,
                                  minimum 3 steps for hardest.

    "requirement_ids" : array   — List of AUTOSAR requirement ID strings that
                                  are referenced in the question or answer.
                                  Use [] if none. Example: ["SWS_Com_00012"].
                                  NEVER invent requirement IDs not present in
                                  the document text.

    "page_reference"  : string  — "Page N" or "Pages N–M". Use the page
                                  numbers shown in the document content below.

    "evidence_snippets": array  — List of 1–3 SHORT verbatim quotes (under 20
                                  words each) from the document that directly
                                  support the answer. These must be EXACT quotes
                                  — copy the words character-for-character from
                                  the document text. Do NOT paraphrase here.
  }

CRITICAL OUTPUT RULES:
  • Return ONLY the JSON array. No preamble, no explanation, no markdown fences.
  • The array must contain EXACTLY the number of questions requested.
  • Every string value must be properly JSON-escaped.
  • Do NOT add comments inside the JSON (// comments are not valid JSON).
  • Do NOT add trailing commas after the last array element or object field.
"""

    _ABSOLUTE_PROHIBITIONS = """\
═══════════════════════════════════════════════════════════════════════════════
SECTION 5 — ABSOLUTE PROHIBITIONS (violations make the question invalid)
═══════════════════════════════════════════════════════════════════════════════

PROHIBITED QUESTION PATTERNS — never generate these regardless of content:

  ✗  Yes/No questions
     BAD:  "Does module X support feature Y?"
     GOOD: "What is module X's approach to feature Y?"

  ✗  Questions answerable with "N/A", empty cells, or absent information
     BAD:  "What is the default value of parameter X?" (when the table shows N/A)
     GOOD: Only ask about parameters where the table contains a real value.

  ✗  Questions that require knowledge from a different document
     BAD:  "According to AUTOSAR_SWS_BSWGeneral, what is the timeout for X?"
           (BSWGeneral is not the document in the chunk below)
     GOOD: Only ask about what THIS document's text explicitly states.

  ✗  Invented requirement IDs — never fabricate IDs not present in the text
     BAD:  "What does SWS_Com_99999 specify?" (if that ID is not in the text)
     GOOD: Only use requirement IDs that appear verbatim in the document content.

  ✗  Hypothetical questions not grounded in document text
     BAD:  "What would happen if parameter X exceeded its maximum value?"
           (unless the document explicitly describes this behavior)

  ✗  Ambiguous questions with multiple valid answers
     BAD:  "What are the requirements for module X?" (too broad, many answers)
     GOOD: "What are the initialization sequence requirements for module X
            as stated in section Y?"

  ✗  Questions that test general AUTOSAR knowledge, not THIS document
     BAD:  "What does the acronym ECU stand for?"
     GOOD: Questions where the answer is a specific fact from the provided text.

  ✗  Questions whose answer is just a module name with no further detail
     BAD:  "Which module handles diagnostic events?" / Answer: "DEM"
     GOOD: Ask something that requires understanding what the module does or
           how it behaves, with the answer drawn from the document text.

  ✗  Repeating or paraphrasing a question already generated in this batch
     Every question in your output must be about a DIFFERENT fact or concept.
"""

    _SELF_VALIDATION_GATE = """\
═══════════════════════════════════════════════════════════════════════════════
SECTION 6 — MANDATORY SELF-VALIDATION BEFORE WRITING JSON
═══════════════════════════════════════════════════════════════════════════════

Before writing your final JSON output, you MUST silently work through this
checklist for EACH question. Only include a question if ALL checks pass.

For EVERY question you plan to generate, verify:

  [ ] 1. ANSWERABILITY
          Can I point to the exact sentence, table row, or requirement ID in
          the document content above that contains the answer?
          → If NO: discard this question entirely.

  [ ] 2. SELF-CONTAINMENT
          If I give only the question text to someone who has NOT read the
          document, will they understand what is being asked without confusion?
          → If NO: rephrase to include the necessary context (module name,
            parameter name, section reference).

  [ ] 3. DIFFICULTY MATCH
          Does this question match the requested difficulty level?
          Easy    → answerable from one sentence in prose
          Medium  → requires reading a specific table cell correctly
          Hard    → requires combining two separate pieces of information
          Hardest → requires tracing from behavior to requirement ID or
                    from one requirement to another
          → If difficulty does not match: adjust the question or discard it.

  [ ] 4. PROHIBITION CHECK
          Does this question violate any prohibition in Section 5?
          → If YES: discard or rephrase.

  [ ] 5. ANSWER ACCURACY
          Is my answer 100% supported by the document text?
          Have I preserved normative keywords (shall/should/may) exactly?
          Have I copied values (numbers, ranges, types) exactly as written?
          → If ANY part of the answer comes from my general knowledge rather
            than the document text: remove that part or discard the question.

  [ ] 6. EVIDENCE SNIPPETS
          Are my evidence_snippets exact verbatim quotes from the document?
          Are they under 20 words each?
          → If I am paraphrasing: copy the exact words instead.

  [ ] 7. REQUIREMENT IDs
          Do all IDs in requirement_ids appear verbatim in the document text?
          → If any ID was inferred or invented: remove it from the list.

  [ ] 8. UNIQUENESS
          Is this question about a different fact than all other questions
          in this batch?
          → If it overlaps: discard it and choose a different fact.

Only after completing this checklist for all questions should you write the
JSON output. Do not show the checklist in your output — only show the JSON.
"""

    # ──────────────────────────────────────────────────────────────────────────
    # DIFFICULTY-SPECIFIC CONTRACTS
    # ──────────────────────────────────────────────────────────────────────────

    _EASY_CONTRACT = """\
═══════════════════════════════════════════════════════════════════════════════
SECTION 4 — DIFFICULTY CONTRACT: EASY (Lookup Questions)
═══════════════════════════════════════════════════════════════════════════════

DEFINITION:
  An easy question has a single, direct answer that appears in ONE place in
  the document. The reader needs only to find that one place and read it.
  No reasoning, no combining of facts, no table parsing required.
  Typically answered from pure prose: purpose statements, scope descriptions,
  introductory definitions, or single-sentence factual claims.

WHAT MAKES A GOOD EASY QUESTION:
  • Asks about the PURPOSE, ROLE, SCOPE, or DEFINITION of something
  • Asks about a COUNT or ENUMERATION that is directly stated
  • Asks about a NAMED PROPERTY that is defined in one sentence
  • The answer is 1–2 sentences maximum
  • Anyone reading the correct paragraph would immediately know the answer

CORRECT EXAMPLES (study these carefully):
  ✅ "What is the primary purpose of the BswM module as described in this
      specification?"
      → Answer is a direct purpose statement from the module introduction.

  ✅ "According to this document, how many operational modes does the EcuM
      module define for the ECU lifecycle?"
      → Answer is a count explicitly stated in the text.

  ✅ "What does the abbreviation 'PDU' stand for as used in this specification?"
      → Answer is a definition stated once in the text.

  ✅ "What is the defined scope of the COM module as stated in this document?"
      → Answer copies the scope statement from the introductory section.

INCORRECT EXAMPLES (never generate these as easy questions):
  ✗ "What are all the requirements that COM module shall fulfill?"
    → Too broad, requires synthesising multiple paragraphs. This is hard/hardest.

  ✗ "Does the NvM module support asynchronous operations?"
    → Yes/no question. Prohibited.

  ✗ "What is the valid range of parameter ComSignalLength?"
    → Requires reading a table. This is medium difficulty.

  ✗ "How does BswM interact with EcuM during shutdown?"
    → Requires combining information. This is hard difficulty.

ANSWER FORMAT FOR EASY:
  • 1–2 sentences
  • Must be a direct restatement of what the document says
  • Must include a page or section reference
  • Do NOT add interpretation or elaboration
  • reasoning_steps must say: "Direct lookup from [page/section reference]."

CONTENT-TYPE GUIDANCE FOR EASY:
  prose content            → Ideal. Find a purpose, definition, or scope statement.
  mixed_requirements       → Find a non-normative explanatory sentence. Avoid
                             quoting a "shall" statement for an easy question —
                             that belongs in hardest.
  parameter_table content  → Ask about the DESCRIPTION column of a parameter
                             (what is it for?), NOT about its range or type.
  cross_reference content  → Ask what the document states IS the reference,
                             not what the referenced document contains.
"""

    _MEDIUM_CONTRACT = """\
═══════════════════════════════════════════════════════════════════════════════
SECTION 4 — DIFFICULTY CONTRACT: MEDIUM (Parameter/Value Extraction)
═══════════════════════════════════════════════════════════════════════════════

DEFINITION:
  A medium question requires the reader to locate a specific VALUE, RANGE,
  TYPE, MULTIPLICITY, or CONSTRAINT, typically stored in a table or a formal
  parameter definition block. The answer is precise and exact — a wrong value
  is clearly wrong. The challenge is correctly reading the right row and the
  right column of a table, or finding the right field in a structured definition.

WHAT MAKES A GOOD MEDIUM QUESTION:
  • Names a specific parameter and asks for one of its formal properties
  • The answer is a precise value: a number, range, type name, or enum value
  • The question could only be answered by someone who read the right table row
  • Multiple similar-sounding parameters exist, so the reader must be precise
  • A wrong answer (e.g., confusing default value with valid range) is clearly
    distinguishable from the correct answer

CORRECT EXAMPLES (study these carefully):
  ✅ "What is the valid numerical range for the parameter 'ComTimeoutFactor'
      as defined in the COM ECUC configuration table?"
      → Answer: e.g. "0 to 65535, type uint16" — exact values from table.

  ✅ "What is the lower and upper multiplicity for the container
      'ComIPdu' in the COM configuration schema?"
      → Answer: e.g. "Lower: 0, Upper: * (unbounded)" — from Multiplicity column.

  ✅ "What data type category does the parameter 'NvMDatasetSelectionBits'
      belong to according to the ECUC parameter definition?"
      → Answer: e.g. "EcucNumericalParamValue" — from Type column.

  ✅ "What is the default value of the parameter 'WdgMSupervisedEntityId'?"
      → Answer: exact default value from DefaultValue column.

INCORRECT EXAMPLES (never generate these as medium questions):
  ✗ "What is the purpose of parameter ComSignalLength?"
    → That is easy (description column). For medium, ask about Range or Type.

  ✗ "What is the valid range for parameter X?" when the table shows N/A or empty
    → Prohibited. Only ask about parameters with real, non-empty values.

  ✗ "How does parameter ComTimeoutFactor affect the timeout behavior?"
    → That is hard (relationship between parameter and behavior).

  ✗ "What parameters does the COM module define?"
    → Too vague and broad. Name a specific parameter.

ANSWER FORMAT FOR MEDIUM:
  • Must state the exact value(s) as they appear in the table/definition
  • Must include the column name (Type / Range / Default / Multiplicity)
  • Must include the parameter name in the answer
  • Must cite the table location (page number or table heading)
  • 1–3 sentences maximum
  • reasoning_steps: "Located parameter [name] in table on [page]. Read the
    [column name] column value: [value]."

CRITICAL TABLE READING RULES:
  When a table is present in the document content below, you MUST:
  1. Identify the header row — the first row defines what each column means.
  2. Identify which row contains the parameter you are asking about.
  3. Read the CORRECT column — do not confuse Range with Default Value.
  4. If a cell is empty or says "N/A" or "–", do NOT ask about that property.
  5. If the table has merged cells, apply the merged header to all sub-rows.

CONTENT-TYPE GUIDANCE FOR MEDIUM:
  parameter_table content  → Ideal. One parameter per question. Use Range, Type,
                             Multiplicity, or DefaultValue columns.
  mixed_requirements       → Look for inline parameter constraints like
                             "the value shall be in range 0..255".
  prose content            → Only generate medium questions if the prose contains
                             an explicit "value is X" or "range is Y..Z" statement.
  cross_reference content  → Generally avoid medium questions unless explicit
                             parameter values are stated in this chunk.
"""

    _HARD_CONTRACT = """\
═══════════════════════════════════════════════════════════════════════════════
SECTION 4 — DIFFICULTY CONTRACT: HARD (Relationship / Multi-hop Questions)
═══════════════════════════════════════════════════════════════════════════════

DEFINITION:
  A hard question CANNOT be answered from a single location. It requires
  combining EXACTLY TWO pieces of information from DIFFERENT parts of the
  document — different paragraphs, different sections, or a table value
  combined with a prose explanation. The reader must reason about the
  relationship between those two pieces to arrive at the answer.

WHAT MAKES A GOOD HARD QUESTION:
  • Explicitly asks about a relationship, interaction, dependency, or sequence
  • The answer requires PIECE A (from location X) + PIECE B (from location Y)
  • If you only had piece A OR only piece B, you could not answer correctly
  • The relationship being asked about is explicitly described in the document
    (not something you infer from general AUTOSAR knowledge)
  • The answer is 2–4 sentences, explaining both pieces and how they connect

CORRECT EXAMPLES (study these carefully):
  ✅ "According to this document, what must happen to the watchdog trigger
      interval if the supervised entity transitions from ALIVE to FAILED state,
      and which module is responsible for performing that action?"
      → Piece A: what happens to the interval (from one paragraph)
      → Piece B: which module is responsible (from another paragraph/table)

  ✅ "The document states that parameter 'NvMBlockUseCrc' must be enabled for
      a certain feature to work. What feature is this, and what error is
      reported if the CRC check fails?"
      → Piece A: the feature that requires CRC (from parameter description)
      → Piece B: the specific error reported on failure (from another section)

  ✅ "How does the BswM module determine which mode rules to evaluate when
      it receives a mode indication from the RTE, and what is the maximum
      number of mode rules permitted in this configuration?"
      → Piece A: how BswM evaluates rules (from behavioral description)
      → Piece B: the maximum number (from a parameter or constraint)

  ✅ "What initialization step must be completed before the COM module can
      transmit I-PDUs, and which requirement mandates this ordering?"
      → Piece A: the required initialization step (from prose)
      → Piece B: the requirement ID that mandates it (from requirement block)

INCORRECT EXAMPLES (never generate these as hard questions):
  ✗ "What is the purpose of the DEM module and what is the range of
      parameter DemEventId?"
    → This is two separate easy/medium questions joined with "and", NOT a
      genuine relationship. Hard questions require the two pieces to be
      LOGICALLY CONNECTED, not just listed together.

  ✗ "How does BswM interact with EcuM?"
    → Too vague. A hard question must specify the exact scenario or condition
      described in the document.

  ✗ "What are all the modules that COM depends on?"
    → List question with no relationship reasoning. This is easy/medium.

  ✗ "If parameter X exceeds its range, what happens?"
    → Hypothetical unless the document explicitly describes this behavior.

ANSWER FORMAT FOR HARD:
  • 2–4 sentences
  • Must explicitly identify BOTH pieces of evidence and WHERE each comes from
  • Must explain the logical connection between them
  • reasoning_steps must be a numbered list of at least 2 steps:
    "1. From [location], established that [Piece A].
     2. From [location], established that [Piece B].
     3. Combined: [conclusion]."

CONTENT-TYPE GUIDANCE FOR HARD:
  mixed_requirements  → Ideal. Connect a requirement ID to a behavioral
                        consequence described in adjacent prose.
  parameter_table     → Connect a parameter constraint to a behavioral
                        description elsewhere in the chunk.
  cross_reference     → Connect a reference to another document with the
                        reason for that reference stated in this chunk.
  prose               → Connect a precondition described in one paragraph
                        to a consequence described in another.
"""

    _HARDEST_CONTRACT = """\
═══════════════════════════════════════════════════════════════════════════════
SECTION 4 — DIFFICULTY CONTRACT: HARDEST (Requirement Tracing)
═══════════════════════════════════════════════════════════════════════════════

DEFINITION:
  A hardest question requires understanding the FORMAL REQUIREMENT STRUCTURE
  of AUTOSAR documents. This means one of:
    (A) BEHAVIOR → REQUIREMENT: Given a described behavior or constraint,
        identify which specific requirement ID (SWS_, RS_, CONSTR_, etc.)
        mandates it, and what normative keyword (shall/should/may) it uses.
    (B) REQUIREMENT → REQUIREMENT: Given a lower-level SWS requirement,
        identify which higher-level RS or SRS requirement it traces to,
        or vice versa.
    (C) CROSS-DOCUMENT TRACING: A requirement in this document references
        another AUTOSAR specification — identify what constraint or behavior
        that external reference imposes on the implementation as described
        in THIS document.

  These questions are hardest because they require:
    1. Understanding that AUTOSAR requirements are normative and hierarchical
    2. Reading a requirement ID correctly and matching it to the right text
    3. Distinguishing between "shall" (mandatory) and "should" (recommended)
    4. Knowing that a cross-reference means a dependency, not just a mention

WHAT MAKES A GOOD HARDEST QUESTION:
  • Always involves at least one AUTOSAR requirement ID (SWS_, RS_, etc.)
  • Tests whether the reader understands NORMATIVE vs. INFORMATIVE content
  • Cannot be answered without understanding the requirement structure
  • The answer must cite the exact requirement ID AND the normative keyword
  • Tracing questions must follow the actual hierarchy present in the text

CORRECT EXAMPLES — TYPE A: Behavior → Requirement:
  ✅ "The COM module is required to stop transmission of all I-PDUs upon
      detection of a bus-off condition. Which requirement ID mandates this
      behavior, and is this obligation normative (shall) or a recommendation
      (should)?"
      → Answer must identify the specific SWS_ ID and quote "shall" or "should".

  ✅ "This document specifies that the WdgM module shall lock the global
      status to WDGM_GLOBAL_STATUS_STOPPED under a specific error condition.
      What is the requirement ID for this obligation, and what is that
      error condition?"
      → Answer: [SWS_WdgM_XXXXX] "shall" + the specific error condition named.

CORRECT EXAMPLES — TYPE B: Requirement → Requirement:
  ✅ "Requirement SWS_BswM_00045 defines a specific arbitration behavior.
      According to this document, which higher-level RS requirement does
      SWS_BswM_00045 implement?"
      → Answer traces the SWS requirement to an RS_ item in the traceability
        table or annotation present in the document.

  ✅ "Which SWS requirement in this document implements the RS-level
      requirement RS_BswM_00003, and what behavior does it specify?"
      → Answer identifies the SWS_ ID that traces to the RS_ ID.

CORRECT EXAMPLES — TYPE C: Cross-Document Tracing:
  ✅ "This specification states that the initialization sequence for module X
      shall follow [AUTOSAR_SWS_BSWGeneral]. According to this document,
      what specific initialization constraint does that reference impose,
      and which requirement ID in this document references it?"
      → Answer: what THIS document says about the referenced constraint +
        the requirement ID that contains the reference.

INCORRECT EXAMPLES (never generate these as hardest questions):
  ✗ "What does SWS_Com_00012 state?"
    → This is just a lookup (easy) if the answer is a single sentence.
      A hardest question must require TRACING or NORMATIVE ANALYSIS.

  ✗ "Which requirements does this document contain?"
    → Too broad. Must ask about a specific behavior, constraint, or trace.

  ✗ "What is the normative keyword in requirement SWS_X_00001?"
    → Too trivial if the document just says "shall" and you copy it.
      The question must require understanding WHY it is normative and
      what behavior it mandates.

  ✗ Inventing a requirement ID not present in the document text.
    → Every requirement ID in your question AND answer MUST appear
      verbatim in the document content provided below.

ANSWER FORMAT FOR HARDEST:
  • 2–4 sentences
  • MUST include the exact requirement ID(s)
  • MUST quote the exact normative keyword (shall/should/may) from the text
  • MUST explain what behavior or constraint the requirement mandates
  • For tracing questions: MUST identify both ends of the trace chain
  • reasoning_steps must be a numbered list of at least 3 steps:
    "1. Identified behavior/constraint: [description].
     2. Located requirement text: '[exact normative statement with keyword]'.
     3. Identified requirement ID: [SWS_X_00001].
     4. [For tracing] Traced to higher-level: [RS_X_00001]."

CONTENT-TYPE GUIDANCE FOR HARDEST:
  mixed_requirements  → Ideal. The requirement IDs are present. Ask which ID
                        mandates a specific "shall" behavior.
  cross_reference     → Ask about the constraint the cross-reference imposes
                        as described in this document, and which requirement
                        ID contains the reference.
  parameter_table     → Ask which requirement mandates a specific parameter
                        constraint (if a requirement ID appears in the table
                        or adjacent text).
  prose               → Only viable if the prose contains explicit requirement
                        IDs. If no IDs are present, do not force a hardest
                        question from pure prose — report inability instead.
"""

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ──────────────────────────────────────────────────────────────────────────

    _DIFFICULTY_CONTRACTS = {
        "easy":    _EASY_CONTRACT,
        "medium":  _MEDIUM_CONTRACT,
        "hard":    _HARD_CONTRACT,
        "hardest": _HARDEST_CONTRACT,
    }

    _QUESTION_TYPES = {
        "easy":    "lookup",
        "medium":  "parameter_extraction",
        "hard":    "relationship",
        "hardest": "requirement_tracing",
    }

    @classmethod
    def build_prompt(cls,
                     chunk_text: str,
                     difficulty: str,
                     num_questions: int,
                     content_type: str,
                     pages: List[int]) -> str:

        page_info    = f"Pages {min(pages)}–{max(pages)}" if len(pages) > 1 else f"Page {pages[0]}"
        content_desc = ContentTypeClassifier.describe(content_type)
        q_type       = cls._QUESTION_TYPES[difficulty]
        contract     = cls._DIFFICULTY_CONTRACTS[difficulty]

        # Hardest-specific warning when content type has no requirement IDs
        hardest_warning = ""
        if difficulty == "hardest" and content_type == "prose":
            hardest_warning = """\
⚠️  IMPORTANT: The chunk below is classified as pure prose with no detected
    requirement IDs. If you cannot find any AUTOSAR requirement IDs (SWS_, RS_,
    CONSTR_, etc.) in the document content below, you MUST output an empty JSON
    array [] rather than generating a question with invented requirement IDs.
    An empty array is acceptable and correct in this situation.
"""

        prompt = f"""{cls._DOMAIN_CONTEXT}
{contract}
{cls._ABSOLUTE_PROHIBITIONS}
{cls._SELF_VALIDATION_GATE}
═══════════════════════════════════════════════════════════════════════════════
SECTION 7 — YOUR SPECIFIC TASK FOR THIS CALL
═══════════════════════════════════════════════════════════════════════════════

Generate EXACTLY {num_questions} question(s) with:
  • difficulty      = "{difficulty}"
  • question_type   = "{q_type}"
  • content_type    = "{content_type}"  ({content_desc})
  • page_reference  = "{page_info}"

{hardest_warning}
The document content you must work from is below. Do NOT use any information
from outside this content block to generate questions or answers.

───────────────────────────────────────────────────────────────────────────────
DOCUMENT CONTENT ({page_info}):
───────────────────────────────────────────────────────────────────────────────
{chunk_text}
───────────────────────────────────────────────────────────────────────────────

Now:
  STEP 1 — Read the document content above carefully and completely.
  STEP 2 — Identify {num_questions} distinct fact(s) or relationship(s) that
            match the "{difficulty}" difficulty contract in Section 4.
  STEP 3 — For each candidate question, run the self-validation checklist
            from Section 6 silently. Discard any that fail.
  STEP 4 — If you cannot find enough valid questions after validation, output
            fewer than {num_questions} rather than generating invalid questions.
            Quality over quantity — one valid question is better than three
            invalid ones.
  STEP 5 — Output ONLY the final JSON array. No explanation. No preamble.
            No markdown code fences. Just the raw JSON array starting with [
            and ending with ].
"""
        return prompt

    @classmethod
    def _content_type_instructions(cls, content_type: str, difficulty: str) -> str:
        """Legacy helper kept for compatibility — logic is now inside contracts."""
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# SEMANTIC DUPLICATE DETECTOR  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

class SemanticDuplicateDetector:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, threshold: float = 0.85):
        self.base_url  = base_url
        self.threshold = threshold
        self.question_embeddings: List[Tuple[str, np.ndarray]] = []
        self.enabled   = ENABLE_SEMANTIC_DEDUP

        if self.enabled:
            try:
                self._verify_ollama()
                print(f"✅ Semantic deduplication enabled (threshold: {threshold})")
            except Exception as e:
                print(f"⚠️  Semantic deduplication disabled: {e}")
                self.enabled = False

    def _verify_ollama(self):
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        if "bge-m3:latest" not in models:
            raise RuntimeError("bge-m3:latest model not found in Ollama")

    def get_embedding(self, text: str) -> np.ndarray:
        if not self.enabled:
            return np.array([])
        try:
            r = requests.post(f"{self.base_url}/api/embeddings",
                              json={"model": "bge-m3:latest", "prompt": text}, timeout=30)
            r.raise_for_status()
            return np.array(r.json().get("embedding", []), dtype=np.float32)
        except Exception:
            return np.array([])

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if len(v1) == 0 or len(v2) == 0:
            return 0.0
        return float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))

    def is_duplicate(self, question: str) -> Tuple[bool, float]:
        if not self.enabled or not self.question_embeddings:
            return False, 0.0
        emb = self.get_embedding(question)
        if len(emb) == 0:
            return False, 0.0
        max_sim = max(self.cosine_similarity(emb, e) for _, e in self.question_embeddings)
        return max_sim >= self.threshold, max_sim

    def add_question(self, question: str):
        if not self.enabled:
            return
        emb = self.get_embedding(question)
        if len(emb) > 0:
            self.question_embeddings.append((question, emb))


# ──────────────────────────────────────────────────────────────────────────────
# QUESTION GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class QuestionGenerator:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model             = model
        self.base_url          = base_url
        self.generated_hashes  = set()
        self.semantic_detector = SemanticDuplicateDetector(
            base_url=OLLAMA_BASE_URL, threshold=SEMANTIC_SIMILARITY_THRESHOLD
        )
        self._verify_ollama()

    def _verify_ollama(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            available = [m["name"] for m in r.json().get("models", [])]
            if self.model not in available:
                raise RuntimeError(f"Model {self.model} not available. Pull it with: ollama pull {self.model}")
            print(f"✅ Ollama model verified: {self.model}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama not accessible: {e}")

    def call_llm(self, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 3500, retry_count: int = 0) -> str:
        """
        Temperature 0.2 — maximum precision and groundedness.
        Higher max_tokens to handle long chain-of-thought + detailed answers.
        """
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
            if r.status_code == 404:
                raise Exception(f"Model {self.model} not found.")
            if r.status_code != 200:
                if retry_count < MAX_RETRIES:
                    wait = RETRY_DELAY * (2 ** retry_count)
                    time.sleep(wait)
                    return self.call_llm(prompt, temperature, max_tokens, retry_count + 1)
                raise Exception(f"API Error {r.status_code}: {r.text[:200]}")
            return r.json().get("response", "")
        except requests.Timeout:
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                return self.call_llm(prompt, temperature, max_tokens, retry_count + 1)
            raise Exception("Request timeout after max retries")
        except Exception as e:
            if retry_count < MAX_RETRIES and "Model" not in str(e):
                time.sleep(RETRY_DELAY)
                return self.call_llm(prompt, temperature, max_tokens, retry_count + 1)
            raise

    def parse_json_response(self, response: str) -> List[Dict]:
        response = response.strip()
        # Strip markdown fences
        for fence in ("```json", "```"):
            if response.startswith(fence):
                response = response[len(fence):]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        start = response.find('[')
        end   = response.rfind(']')
        if start != -1 and end != -1:
            response = response[start:end+1]

        try:
            data = json.loads(response)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON parse error: {e} | Preview: {response[:200]}")
            return []

    def _question_hash(self, question: str) -> str:
        normalized = ''.join(c for c in question.lower() if c.isalnum() or c.isspace())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _is_duplicate(self, question: str) -> bool:
        h = self._question_hash(question)
        if h in self.generated_hashes:
            return True
        self.generated_hashes.add(h)
        return False

    def verify_question(self, q: Dict, context: str, difficulty: str) -> Tuple[bool, str]:
        """Validate a generated question against quality criteria."""
        required = ['question', 'answer', 'question_type', 'difficulty']
        for field in required:
            if not q.get(field):
                return False, f"Missing field: {field}"

        q_text = q['question'].strip()
        a_text = q['answer'].strip()

        if len(q_text) < 15:
            return False, "Question too short"
        if len(q_text) > 400:
            return False, "Question too long"
        if len(a_text) < 10:
            return False, "Answer too short"
        if q.get('difficulty') != difficulty:
            return False, f"Difficulty mismatch (expected {difficulty}, got {q.get('difficulty')})"
        if self._is_duplicate(q_text):
            return False, "Duplicate question"

        # Semantic duplicate check
        if self.semantic_detector.enabled:
            is_dup, sim = self.semantic_detector.is_duplicate(q_text)
            if is_dup:
                return False, f"Semantic duplicate (sim={sim:.3f})"

        # Grounding check — relaxed for parameter questions (short precise answers)
        if difficulty in ("easy", "hard", "hardest") and len(a_text.split()) > 5:
            answer_words  = set(a_text.lower().split())
            context_words = set(context.lower().split())
            overlap       = len(answer_words & context_words) / max(len(answer_words), 1)
            if overlap < 0.25:
                return False, "Answer not grounded in context"

        self.semantic_detector.add_question(q_text)
        return True, "Valid"

    def generate_for_difficulty(self,
                                content: Dict[str, Any],
                                doc_id: str,
                                difficulty: str,
                                num_questions: int) -> List[Dict]:
        """
        Generate `num_questions` questions of a specific difficulty level.
        Distributes requests across chunks, preferring chunks whose content type
        matches the difficulty's natural content source.
        """
        if num_questions <= 0:
            return []

        chunks = content.get("chunks", [])
        if not chunks:
            return []

        # Score chunks by suitability for this difficulty
        scored_chunks = []
        for chunk in chunks:
            ct    = ContentTypeClassifier.classify(chunk['text'])
            score = self._chunk_suitability_score(ct, difficulty)
            scored_chunks.append((score, ct, chunk))

        # Sort best first
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        all_questions: List[Dict] = []
        remaining     = num_questions
        chunk_idx     = 0

        while remaining > 0 and chunk_idx < len(scored_chunks):
            score, content_type, chunk = scored_chunks[chunk_idx]
            chunk_idx += 1

            # Request slightly more than needed to account for rejections
            request_n = min(remaining + 1, 3)
            pages     = chunk['pages']

            print(f"    [{difficulty.upper()}] Chunk content_type={content_type} "
                  f"(suitability={score}) — requesting {request_n} question(s)")

            prompt = AutosarPromptLibrary.build_prompt(
                chunk_text   = chunk['text'],
                difficulty   = difficulty,
                num_questions= request_n,
                content_type = content_type,
                pages        = pages,
            )

            try:
                response  = self.call_llm(prompt, temperature=0.2, max_tokens=3500)
                questions = self.parse_json_response(response)

                for q in questions:
                    if remaining <= 0:
                        break
                    is_valid, reason = self.verify_question(q, chunk['text'], difficulty)
                    if is_valid:
                        # Enrich metadata
                        q['content_type']     = content_type
                        q['difficulty']       = difficulty
                        q['source_document']  = content['metadata']['filename']
                        q.setdefault('requirement_ids',  [])
                        q.setdefault('reasoning_steps',  "direct lookup")
                        q.setdefault('evidence_snippets',[])
                        all_questions.append(q)
                        remaining -= 1
                    else:
                        print(f"    ⚠️  Rejected ({reason})")

            except Exception as e:
                print(f"    ❌ Generation error: {e}")

        return all_questions

    @staticmethod
    def _chunk_suitability_score(content_type: str, difficulty: str) -> int:
        """
        Returns a score (higher = better) indicating how well a content type
        serves a given difficulty level.
        """
        matrix = {
            # content_type          easy  medium  hard  hardest
            "prose":               [4,    1,      2,    1],
            "mixed_requirements":  [3,    2,      3,    4],
            "parameter_table":     [1,    4,      3,    3],
            "cross_reference":     [1,    2,      3,    4],
        }
        order = ["easy", "medium", "hard", "hardest"]
        idx   = order.index(difficulty) if difficulty in order else 0
        return matrix.get(content_type, [1, 1, 1, 1])[idx]

    def generate_questions_from_document(self,
                                         content: Dict[str, Any],
                                         doc_id: str,
                                         question_budget: Dict[str, int]) -> List[Dict]:
        """Generate questions across all difficulty levels for one document."""
        all_questions = []
        question_idx  = 1

        for difficulty in ["easy", "medium", "hard", "hardest"]:
            n = question_budget.get(difficulty, 0)
            if n == 0:
                continue
            print(f"\n  🎯 Generating {n} {difficulty.upper()} question(s)...")
            qs = self.generate_for_difficulty(content, doc_id, difficulty, n)
            for q in qs:
                q['id'] = f"{doc_id}_q{question_idx:03d}"
                question_idx += 1
            print(f"  ✅ Got {len(qs)}/{n} {difficulty.upper()} question(s)")
            all_questions.extend(qs)

        return all_questions


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_questions(questions: List[Dict]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "by_difficulty":    {},
        "by_question_type": {},
        "by_content_type":  {},
        "avg_answer_length": 0,
        "questions_with_req_ids": 0,
        "questions_with_evidence": 0,
    }
    if not questions:
        return stats

    lengths = []
    for q in questions:
        for key, field in [
            ("by_difficulty",    "difficulty"),
            ("by_question_type", "question_type"),
            ("by_content_type",  "content_type"),
        ]:
            val = q.get(field, "unknown")
            stats[key][val] = stats[key].get(val, 0) + 1

        if q.get('answer'):
            lengths.append(len(q['answer']))
        if q.get('requirement_ids'):
            stats["questions_with_req_ids"] += 1
        if q.get('evidence_snippets'):
            stats["questions_with_evidence"] += 1

    stats["avg_answer_length"] = round(sum(lengths) / len(lengths)) if lengths else 0
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def process_dataset(dataset_path: str,
                    output_path:  str,
                    question_budget: Dict[str, int] = None):

    if question_budget is None:
        question_budget = DEFAULT_QUESTION_BUDGET

    total_per_doc = sum(question_budget.values())

    print("=" * 80)
    print("AUTOSAR RAG EVALUATION — QUESTION GENERATOR")
    print("=" * 80)
    print(f"Dataset Path    : {dataset_path}")
    print(f"Output          : {output_path}")
    print(f"LLM Model       : {OLLAMA_MODEL}")
    print(f"Question Budget : {question_budget}")
    print(f"Total per doc   : {total_per_doc}")
    print("=" * 80)

    extractor = PDFContentExtractor()
    generator = QuestionGenerator(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    pdf_files = list(Path(dataset_path).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {dataset_path}")
        return

    print(f"📚 Found {len(pdf_files)} PDF file(s)\n")

    all_questions: List[Dict] = []
    failed_docs:   List[Tuple] = []
    doc_count = 0

    for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
        doc_id = f"doc{idx+1:03d}"
        print(f"\n📖 Processing: {pdf_path.name}")

        try:
            print("  📄 Extracting content (text + tables)...")
            content = extractor.extract_with_pdfplumber(str(pdf_path))

            if not content or not content["full_text"].strip():
                print("  ⚠️  pdfplumber failed, trying PyPDF2...")
                raw = extractor.extract_with_pypdf2(str(pdf_path))
                if not raw:
                    print(f"  ❌ Skipping {pdf_path.name} — no content")
                    failed_docs.append((pdf_path.name, "No content extracted"))
                    continue
                content = {
                    "full_text": raw,
                    "metadata":  {"filename": pdf_path.name, "num_pages": "unknown"},
                    "pages":     [],
                    "tables":    [],
                    "chunks":    PDFContentExtractor.intelligent_chunking(raw),
                }

            n_pages  = content['metadata']['num_pages']
            n_chunks = len(content.get('chunks', []))
            n_tables = len(content.get('tables', []))
            print(f"  ✅ {len(content['full_text'])} chars, {n_pages} pages, "
                  f"{n_chunks} chunks, {n_tables} tables")

            # Show content type distribution across chunks
            ct_counts: Dict[str, int] = {}
            for ch in content.get('chunks', []):
                ct = ContentTypeClassifier.classify(ch['text'])
                ct_counts[ct] = ct_counts.get(ct, 0) + 1
            print(f"  📊 Content types: {ct_counts}")

            questions = generator.generate_questions_from_document(
                content, doc_id, question_budget
            )

            if questions:
                all_questions.extend(questions)
                doc_count += 1
                print(f"  ✅ {len(questions)} questions generated for this document")
            else:
                print("  ⚠️  No valid questions generated")
                failed_docs.append((pdf_path.name, "No valid questions"))

        except Exception as e:
            print(f"  ❌ Error: {str(e)[:120]}")
            failed_docs.append((pdf_path.name, str(e)[:120]))

    # ── Final output ──────────────────────────────────────────────────────────
    stats = analyze_questions(all_questions)

    output_data = {
        "dataset_info": {
            "total_documents":   doc_count,
            "total_questions":   len(all_questions),
            "failed_documents":  len(failed_docs),
            "generation_date":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_used":        f"ollama-{OLLAMA_MODEL}",
            "question_budget":   question_budget,
            "dataset_path":      dataset_path,
            "chunk_size":        CHUNK_SIZE,
            "chunk_overlap":     CHUNK_OVERLAP,
            "statistics":        stats,
        },
        "questions":         all_questions,
        "failed_documents":  [{"filename": n, "reason": r} for n, r in failed_docs],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✅ GENERATION COMPLETE")
    print("=" * 80)
    print(f"Documents processed : {doc_count}")
    print(f"Total questions     : {len(all_questions)}")
    print(f"Failed documents    : {len(failed_docs)}")
    print(f"Output saved to     : {output_path}")

    print("\n📊 QUESTION STATISTICS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Sample output per difficulty
    print("\n🔍 SAMPLE QUESTIONS BY DIFFICULTY:")
    print("-" * 80)
    for diff in ["easy", "medium", "hard", "hardest"]:
        matches = [q for q in all_questions if q.get('difficulty') == diff]
        if matches:
            q = matches[0]
            print(f"\n[{diff.upper()}] type={q.get('question_type')} "
                  f"content={q.get('content_type')}")
            print(f"  Q: {q['question']}")
            print(f"  A: {q['answer'][:200]}...")
            if q.get('requirement_ids'):
                print(f"  REQ IDs: {q['requirement_ids']}")

    if failed_docs:
        print("\n⚠️  FAILED DOCUMENTS:")
        for name, reason in failed_docs:
            print(f"  • {name}: {reason}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AUTOSAR RAG Evaluation Question Generator"
    )
    parser.add_argument("--dataset",   default="/home/olj3kor/praveen/RAG_work/DATASET/",
                        help="Path to folder containing AUTOSAR PDF files")
    parser.add_argument("--output",    default="/home/olj3kor/praveen/RAG_work/evaluation_questions_autosar.json",
                        help="Output JSON file path")
    parser.add_argument("--easy",      type=int, default=DEFAULT_QUESTION_BUDGET["easy"],
                        help="Number of EASY (lookup) questions per document")
    parser.add_argument("--medium",    type=int, default=DEFAULT_QUESTION_BUDGET["medium"],
                        help="Number of MEDIUM (parameter extraction) questions per document")
    parser.add_argument("--hard",      type=int, default=DEFAULT_QUESTION_BUDGET["hard"],
                        help="Number of HARD (relationship/multi-hop) questions per document")
    parser.add_argument("--hardest",   type=int, default=DEFAULT_QUESTION_BUDGET["hardest"],
                        help="Number of HARDEST (requirement tracing) questions per document")
    args = parser.parse_args()

    # Verify Ollama
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        print(f"✅ Ollama running at {OLLAMA_BASE_URL}\n")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        exit(1)

    budget = {
        "easy":    args.easy,
        "medium":  args.medium,
        "hard":    args.hard,
        "hardest": args.hardest,
    }

    process_dataset(
        dataset_path    = args.dataset,
        output_path     = args.output,
        question_budget = budget,
    )
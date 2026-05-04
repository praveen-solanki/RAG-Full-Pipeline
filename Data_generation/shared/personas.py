"""
Personas used during candidate generation.

IMPORTANT: We do NOT use RAGAS's auto-generated personas because one of them
reliably turns out to be a "casual/lazy typer" which is what produced the
`wat iz ISO/WD 26262-1` queries in the v0 dataset (35% of single-hop queries
were noisy because of this).

All personas here write in clean technical English. If you want noisy/typo
robustness testing, generate a SEPARATE split with a `noisy_user` persona
whose role explicitly mentions misspellings, and keep that out of the main
gold set.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    role_description: str  # what kind of questions they ask, their domain focus


AUTOSAR_PERSONAS: list[Persona] = [
    Persona(
        name="autosar_integrator",
        role_description=(
            "An AUTOSAR integration engineer responsible for composing and "
            "configuring BSW modules. They ask precise, technical questions "
            "about module behavior, configuration parameters, interfaces "
            "between stacks (COM, CAN, Mode Management, Diagnostics), and "
            "how specific requirements interact. They write in clean, formal "
            "technical English using the document's own terminology."
        ),
    ),
    Persona(
        name="tier1_developer",
        role_description=(
            "A Tier-1 supplier software developer implementing an AUTOSAR "
            "module. They ask about implementation details, API contracts, "
            "runtime behavior, error handling, and how to satisfy specific "
            "SWS requirements. They write in clean, precise technical "
            "English and reference specific module names and requirement IDs."
        ),
    ),
    Persona(
        name="functional_safety_engineer",
        role_description=(
            "A functional safety engineer familiar with ISO 26262 working "
            "on automotive software. They ask about safety-related aspects "
            "of AUTOSAR: fault detection, ASIL decomposition, safety "
            "mechanisms, error handling, freedom from interference, and how "
            "these interact with specific BSW modules. They write in "
            "precise technical English with correct use of functional "
            "safety terminology."
        ),
    ),
    Persona(
        name="application_sw_engineer",
        role_description=(
            "An automotive application software engineer who writes SWCs "
            "and consumes AUTOSAR services. They ask about application "
            "interfaces, RTE behavior, port configurations, mode switching "
            "seen from the application side, and how system-level concepts "
            "apply to their domain (chassis, body, powertrain). They write "
            "in clean, professional English."
        ),
    ),
]


def persona_by_name(name: str) -> Persona:
    for p in AUTOSAR_PERSONAS:
        if p.name == name:
            return p
    raise KeyError(f"Unknown persona: {name}")

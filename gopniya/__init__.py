"""Gopniya: de-identify Indian clinical notes before they reach a cloud LLM."""

from .agent import AgentResult, RedactionAgent, TASKS
from .detect import ENTITY_TYPES, Span, detect
from .vault import Vault

__all__ = ["AgentResult", "RedactionAgent", "TASKS", "ENTITY_TYPES", "Span", "detect", "Vault"]
__version__ = "0.1.0"

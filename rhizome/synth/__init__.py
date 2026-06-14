"""Rhizome RAFT data synthesis — vault pages → (Q + golden + distractors + cited answer).

Deterministic engine (core.py) controls structure; generators (generator.py) only propose Q/A.
Orchestrated by pipeline.run_synth. See docs/raft-synth-detail.eraser.
"""
from .pipeline import run_synth
from .generator import Generator, StubGenerator, ClaudeMCPGenerator, QAPair
from .core import Chunk, chunk_vault

__all__ = ["run_synth", "Generator", "StubGenerator", "ClaudeMCPGenerator",
           "QAPair", "Chunk", "chunk_vault"]

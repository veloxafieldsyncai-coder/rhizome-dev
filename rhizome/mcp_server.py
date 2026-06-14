"""Rhizome MCP server: exposes rhizome as tools any frontier AI or local model can drive.

Run (stdio transport):  python -m rhizome.mcp_server
Requires the `mcp` package (pip install mcp). The tool logic lives in server_tools.py.

The point: a user does research with whatever AI they bring. That AI (or a local model)
drives rhizome through these tools to keep the vault current, generate Q/A, and build the
training set. Local-first: point a LOCAL model at it and no data leaves the box (NDA-safe);
point a FRONTIER model at it for more power when the data is not sensitive.
"""
from mcp.server.fastmcp import FastMCP

from . import server_tools as t

mcp = FastMCP("rhizome")


@mcp.tool()
def vault_status() -> dict:
    """Counts of vault pages and chunks by layer (research vs brainstorm), and how many Q/A are queued."""
    return t.vault_status()


@mcp.tool()
def list_chunks(layer: str = "", limit: int = 50) -> list:
    """List vault chunks to generate Q/A for. layer filter: 'research' or 'brainstorm' (empty = both). Returns each chunk's id, title, layer, and text."""
    return t.list_chunks(layer or None, limit)


@mcp.tool()
def submit_qa(question: str, answer: str, kind: str, golden_ids: list) -> dict:
    """Submit one Q/A pair grounded in vault chunk(s).
    kind: 'research' (assert + quote source VERBATIM in ##begin_quote##..##end_quote##) |
          'note' (attribute to the user, never assert) |
          'ungrounded' (refuse to assert from notes alone) |
          'cross' (reason over a note while grounding facts in a verbatim research quote).
    golden_ids: the chunk id(s) the answer is grounded in (note id first for cross)."""
    return t.submit_qa(question, answer, kind, golden_ids)


@mcp.tool()
def build_dataset(k_distractors: int = 4, p_golden: float = 0.8,
                  eval_frac: float = 0.1, seed: int = 42) -> dict:
    """Assemble all submitted Q/A into train.jsonl + eval.jsonl through the deterministic engine (tagging, asymmetric P split, citation QC, page split). Returns stats."""
    return t.build_dataset(k_distractors, p_golden, eval_frac, seed)


@mcp.tool()
def clear_qa() -> dict:
    """Clear the queued Q/A submissions (start a fresh dataset)."""
    return t.clear_qa()


@mcp.tool()
def add_page(rel_path: str, title: str, layer: str, body: str,
             category: str = "concepts", tags: list = [], summary: str = "") -> dict:
    """Add or overwrite a vault page. layer: 'research' (vetted truth) or 'brainstorm' (the user's provisional notes). rel_path is relative to the vault root, e.g. 'references/foo.md'."""
    return t.add_page(rel_path, title, layer, body, category, list(tags), summary)


@mcp.tool()
def training_command(adapter: str = "adapter-domain") -> dict:
    """Return the exact command (cwd, env, cmd) to RAFT-train the local 30B on the built dataset."""
    return t.training_command(adapter)


if __name__ == "__main__":
    mcp.run()

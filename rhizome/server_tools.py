"""Logic behind the rhizome MCP tools.

Plain functions, testable without the MCP runtime; mcp_server.py wraps them as tools.
This is how any frontier AI (or a local model, or an agent like Claude right now) drives
rhizome: inspect the vault, submit Q/A, build the RAFT dataset, add pages, get the
training command.

Local-first by design: Q/A can be proposed by a LOCAL model so no data leaves the box
(private, NDA-safe), or by a FRONTIER model for more power when the data is not sensitive.
The deterministic engine that assembles the dataset is identical either way.
"""
from __future__ import annotations
import datetime
import json
import os
from pathlib import Path

from .synth.core import chunk_vault
from .synth.generator import QAPair
from .synth.pipeline import assemble_dataset

REPO = Path(__file__).resolve().parent.parent
_KINDS = ("research", "note", "ungrounded", "cross")


def _vault() -> str:
    return os.environ.get("RHIZOME_VAULT", str(REPO / "vault"))


def _data() -> Path:
    return Path(os.environ.get("RHIZOME_DATA", str(REPO / "data")))


def _store() -> Path:
    return _data() / "qa_store.jsonl"


def _count_store() -> int:
    p = _store()
    return sum(1 for ln in p.read_text().splitlines() if ln.strip()) if p.exists() else 0


def _load_store() -> list[QAPair]:
    p = _store()
    if not p.exists():
        return []
    return [QAPair(d["question"], d["answer"], d["golden_ids"], d["kind"])
            for d in (json.loads(ln) for ln in p.read_text().splitlines() if ln.strip())]


# ── tools ─────────────────────────────────────────────────────────────────────


def vault_status() -> dict:
    chunks = chunk_vault(_vault())
    return {
        "vault": _vault(),
        "pages": len({c.page_path for c in chunks}),
        "chunks": len(chunks),
        "research_chunks": sum(c.layer == "research" for c in chunks),
        "brainstorm_chunks": sum(c.layer == "brainstorm" for c in chunks),
        "submitted_qa": _count_store(),
    }


def list_chunks(layer: str | None = None, limit: int = 50) -> list[dict]:
    chunks = chunk_vault(_vault())
    if layer:
        chunks = [c for c in chunks if c.layer == layer]
    return [{"id": c.id, "title": c.title, "heading": c.heading, "layer": c.layer,
             "text": c.text[:1500]} for c in chunks[:limit]]


def submit_qa(question: str, answer: str, kind: str, golden_ids: list[str]) -> dict:
    if kind not in _KINDS:
        return {"error": f"kind must be one of {_KINDS}"}
    _data().mkdir(parents=True, exist_ok=True)
    rec = {"question": question, "answer": answer, "kind": kind, "golden_ids": golden_ids}
    with open(_store(), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"ok": True, "submitted_total": _count_store()}


def clear_qa() -> dict:
    if _store().exists():
        _store().write_text("")
    return {"ok": True}


def build_dataset(k_distractors: int = 4, p_golden: float = 0.8,
                  eval_frac: float = 0.1, seed: int = 42) -> dict:
    qa = _load_store()
    if not qa:
        return {"error": "no submitted Q/A; call submit_qa first"}
    chunks = chunk_vault(_vault())
    return assemble_dataset(chunks, qa, str(_data()), k_distractors=k_distractors,
                            p_golden=p_golden, eval_frac=eval_frac, seed=seed)


def add_page(rel_path: str, title: str, layer: str, body: str,
             category: str = "concepts", tags: list[str] | None = None,
             summary: str = "") -> dict:
    if layer not in ("research", "brainstorm"):
        return {"error": "layer must be research or brainstorm"}
    today = datetime.date.today().isoformat()
    p = Path(_vault()) / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    fm = (f"---\ntitle: {title}\ncategory: {category}\nlayer: {layer}\n"
          f"tags: [{', '.join(tags or [])}]\nsummary: {summary}\nlifecycle: active\n"
          f"created: {today}\nupdated: {today}\n---\n\n")
    p.write_text(fm + body.rstrip() + "\n")
    return {"ok": True, "path": str(p.relative_to(REPO))}


def training_command(adapter: str = "adapter-domain") -> dict:
    """The exact command to RAFT-train the local 30B on the built dataset."""
    return {
        "cwd": str(REPO / "spikes" / "amd-lora"),
        "env": {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1", "HSA_ENABLE_SDMA": "0",
                "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
                "HF_DEACTIVATE_ASYNC_LOAD": "1", "HF_HOME": "/srv/project/cache/hf"},
        "cmd": (f".venv/bin/python train_raft_lora.py --model Qwen/Qwen3-30B-A3B "
                f"--data {_data()}/train.jsonl --out {adapter}"),
    }

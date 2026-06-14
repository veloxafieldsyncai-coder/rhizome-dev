"""Model selection for the Rhizome local agent.

The agent talks to local models by *role* (general / physics / coding / ...), not by hardcoded
name. This module is the single source of truth for which model fills which role, persisted in
`config/models.json`. It is deliberately NOT limited to any particular models:

  - any Ollama tag works (installed or not — we offer to pull it),
  - roles are free-form (add, rename, or remove them),
  - a `provider` field per role leaves the door open to non-Ollama backends later.

Pure stdlib, no third-party dependencies (matching the fork's zero-dep posture).

Usage:
    python -m rhizome.models show                 # current role -> model map
    python -m rhizome.models list                 # installed models + suggestions
    python -m rhizome.models get physics          # print the model for a role (for scripts)
    python -m rhizome.models set physics qwq:32b  # assign any model to any role (creates role if new)
    python -m rhizome.models choose               # interactive picker for every role
    python -m rhizome.models choose physics       # interactive picker for one role
    python -m rhizome.models pull qwq:32b         # pull a model via Ollama
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# config/models.json lives next to this package's parent (the repo root for rhizome-dev).
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.json"
OLLAMA_HOST = "http://localhost:11434"

# Suggestions are *hints only* — the picker always lets you choose any installed model or type
# any tag. Nothing here constrains what you can use. Extend or ignore freely.
SUGGESTIONS: dict[str, list[tuple[str, str]]] = {
    "general": [
        ("qwen3:30b-a3b", "MoE ~3B active, ~50 tok/s, thinking mode — Strix-Halo sweet spot"),
        ("llama3.3:70b", "dense 70B, more world knowledge, slower on iGPU"),
        ("qwen3:14b", "lighter and faster, less capable"),
    ],
    "physics": [
        ("qwq:32b", "dense reasoner, strong math/physics, shows its work"),
        ("qwen3:32b", "thinking toggle, strong math + more general"),
        ("deepseek-r1:32b", "R1-distilled reasoning, competition-math strength"),
    ],
    "coding": [
        ("qwen3-coder:30b", "Apache-2.0 MoE, 256K context, agentic/tool use"),
        ("deepseek-coder-v2:16b", "lighter, strong Python/JS"),
        ("codestral", "Mistral's code model"),
    ],
}


# ── config I/O ────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"default_provider": "ollama", "roles": {}}
    with CONFIG_PATH.open() as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def get_role(cfg: dict, role: str) -> dict | None:
    return cfg.get("roles", {}).get(role)


def set_role(cfg: dict, role: str, model: str, provider: str | None = None, note: str = "") -> None:
    roles = cfg.setdefault("roles", {})
    existing = roles.get(role, {})
    roles[role] = {
        "model": model,
        "provider": provider or existing.get("provider") or cfg.get("default_provider", "ollama"),
        "note": note or existing.get("note", ""),
    }


# ── ollama introspection ──────────────────────────────────────────────────────

def installed_models() -> list[str]:
    """Return every locally available Ollama model tag (whatever they are)."""
    # Prefer the HTTP API; fall back to the CLI; never raise.
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
            data = json.load(r)
        return sorted(m["name"] for m in data.get("models", []))
    except (urllib.error.URLError, OSError, ValueError):
        pass
    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
        names = []
        for line in out.splitlines()[1:]:  # skip header
            if line.strip():
                names.append(line.split()[0])
        return sorted(names)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []


def pull(model: str) -> int:
    """Pull a model via Ollama, streaming progress to the terminal."""
    print(f"Pulling {model} …")
    try:
        return subprocess.run(["ollama", "pull", model]).returncode
    except FileNotFoundError:
        print("ollama not found on PATH.", file=sys.stderr)
        return 1


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_show(_args) -> int:
    cfg = load_config()
    roles = cfg.get("roles", {})
    if not roles:
        print("No roles configured. Run `python -m rhizome.models choose`.")
        return 0
    have = set(installed_models())
    width = max(len(r) for r in roles)
    print("Role".ljust(width), " Model")
    print("-" * width, "------")
    for role, spec in roles.items():
        model = spec.get("model", "")
        mark = "✓" if model in have else "·"  # ✓ installed, · not yet pulled
        prov = spec.get("provider", "ollama")
        note = spec.get("note", "")
        line = f"{role.ljust(width)}  {mark} {model}"
        if prov != "ollama":
            line += f"  [{prov}]"
        if note:
            line += f"   — {note}"
        print(line)
    return 0


def cmd_list(_args) -> int:
    have = installed_models()
    print("Installed models (any of these can fill any role):")
    if have:
        for m in have:
            print(f"  ✓ {m}")
    else:
        print("  (none installed yet — `ollama pull <tag>` or use `choose`)")
    print("\nSuggestions by role (hints only — you are not limited to these):")
    for role, opts in SUGGESTIONS.items():
        print(f"  {role}:")
        for tag, desc in opts:
            mark = "✓" if tag in have else " "
            print(f"    [{mark}] {tag:24s} {desc}")
    return 0


def cmd_get(args) -> int:
    """Print just the model tag for a role — for the agent/scripts to consume."""
    spec = get_role(load_config(), args.role)
    if not spec:
        print(f"(no model set for role '{args.role}')", file=sys.stderr)
        return 1
    print(spec["model"])
    return 0


def cmd_set(args) -> int:
    cfg = load_config()
    set_role(cfg, args.role, args.model, provider=args.provider, note=args.note or "")
    save_config(cfg)
    print(f"Set {args.role} -> {args.model}"
          + (f" [{args.provider}]" if args.provider else ""))
    if args.provider in (None, "ollama") and args.model not in installed_models():
        print(f"Note: {args.model} is not installed. Pull it with: ollama pull {args.model}")
    return 0


def _choose_one(cfg: dict, role: str, have: list[str]) -> None:
    """Interactive pick for a single role. Any installed model, any suggestion, or any custom tag."""
    current = get_role(cfg, role)
    cur_model = current.get("model") if current else None
    print(f"\n=== role: {role} ===")
    if cur_model:
        print(f"current: {cur_model}")

    # Build a numbered menu: installed models first, then suggestions not already installed.
    menu: list[tuple[str, str]] = []
    for m in have:
        menu.append((m, "installed"))
    for tag, desc in SUGGESTIONS.get(role, []):
        if tag not in have:
            menu.append((tag, f"suggestion — {desc} (not installed)"))

    for i, (tag, desc) in enumerate(menu, 1):
        flag = "←" if tag == cur_model else " "
        print(f"  {i:2d}) {flag} {tag:26s} {desc}")
    print("   c) enter ANY custom model tag")
    print("   s) skip (keep current)")

    choice = input("pick a number, 'c', or 's': ").strip().lower()
    if choice in ("", "s"):
        return
    if choice == "c":
        tag = input("model tag (e.g. mixtral:8x22b, llama3.3:70b, my-model:latest): ").strip()
        if not tag:
            return
        provider = input("provider [ollama]: ").strip() or "ollama"
    else:
        try:
            tag = menu[int(choice) - 1][0]
        except (ValueError, IndexError):
            print("  (not a valid choice; skipping)")
            return
        provider = current.get("provider", "ollama") if current else "ollama"

    set_role(cfg, role, tag, provider=provider)

    # Offer to pull if it's an Ollama model we don't have yet.
    if provider == "ollama" and tag not in have:
        if input(f"  {tag} isn't installed. Pull now? [y/N]: ").strip().lower() == "y":
            pull(tag)


def cmd_choose(args) -> int:
    cfg = load_config()
    have = installed_models()
    roles = [args.role] if args.role else list(cfg.get("roles", {}) or SUGGESTIONS.keys())
    if args.role and args.role not in cfg.get("roles", {}):
        print(f"(new role '{args.role}')")
    for role in roles:
        _choose_one(cfg, role, have)
    save_config(cfg)
    print("\nSaved. Current map:")
    return cmd_show(args)


def cmd_pull(args) -> int:
    return pull(args.model)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rhizome.models", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="show current role -> model map").set_defaults(func=cmd_show)
    sub.add_parser("list", help="list installed models + suggestions").set_defaults(func=cmd_list)

    g = sub.add_parser("get", help="print the model tag for a role")
    g.add_argument("role")
    g.set_defaults(func=cmd_get)

    s = sub.add_parser("set", help="assign any model to any role (creates the role if new)")
    s.add_argument("role")
    s.add_argument("model")
    s.add_argument("--provider", default=None, help="backend (default: ollama)")
    s.add_argument("--note", default=None, help="freeform note")
    s.set_defaults(func=cmd_set)

    c = sub.add_parser("choose", help="interactive picker (all roles, or one)")
    c.add_argument("role", nargs="?", default=None)
    c.set_defaults(func=cmd_choose)

    pl = sub.add_parser("pull", help="pull a model via ollama")
    pl.add_argument("model")
    pl.set_defaults(func=cmd_pull)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

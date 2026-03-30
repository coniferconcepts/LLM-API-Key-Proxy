# Mirrowel Fork Policy

This repository is treated as a permanent Conifer Concepts fork for the OpenCode router stack.

## Policy

- We do **not** treat upstream Mirrowel as the primary source of truth for runtime behavior.
- We do **not** plan to submit routine changes upstream.
- Local behavior may intentionally diverge to match the OpenCode router product architecture.

## Working Rules

- Use fork-specific branches for local runtime changes.
- Commit local fork behavior here instead of leaving submodule changes dirty in the parent repo.
- Pull from upstream only selectively and only when there is a clear benefit.
- Prefer cherry-picking or controlled merges into fork branches rather than trying to stay upstream-clean.

## Current Intentional Divergences

- Upstream-port guidance is tailored to the OpenCode router topology.
- Provider/runtime messaging may omit unsupported or intentionally disabled lanes.

## Operational Model

- Parent repo: `opencode-router`
- Submodule fork remote: `coniferconcepts/LLM-API-Key-Proxy`
- Recommended branch for local product evolution: `fork/opencode-runtime`

# AI Engineering Learning Journal

Working notes, from-scratch implementations, exercises, and paper reading as I work through foundations → transformers → LLM engineering, June through November 2026.

This is the workshop. The polished output of this journey `would` live in my capstone repo:
[`indian-legal-llm-stack`](https://github.com/<your-username>/indian-legal-llm-stack).

---

## What this repo is for

A daily-ish journal of learning, deliberately kept separate from the capstone so I can:

- Commit messy, experimental, and broken code without polishing
- Track honest progress over 24 weeks instead of presenting a finished product
- Keep notes, ablations, and exercises in one searchable place
- Build a public record of the learning practice itself

If you're a recruiter or collaborator looking at my work, the capstone repo is the showcase. This repo is the receipts.

---
Directories appear as work happens — not all exist on day one.

## To be Structure
```
.
├── foundations/             # NumPy from scratch, math warmups, MLP on MNIST
├── karpathy-zero-to-hero/   # micrograd, makemore, nanoGPT exercises
├── papers/                  # Notes and implementations of papers I'm reading
│   ├── attention-is-all-you-need/   # Toy transformer on Tiny Shakespeare
│   ├── rag-lewis-2020/              # Notes + minimal RAG sketch
│   └── _reading-log.md              # Running list of papers at all engagement levels
├── quizzes/                 # Quiz scores, what I missed, retry notes
├── notes/                   # Concept notes, 3Blue1Brown summaries, Anki exports
└── ablations/               # Deliberate breakage experiments to build intuition
```
---

## How I'm using this

- **Daily**: commit whatever I worked on, even if incomplete. No polish requirement.
- **Weekly**: a brief `notes/week-NN.md` summarizing what I learned and what confused me.
- **Monthly**: review the `_reading-log.md` to see whether I'm going deep enough on papers or just skimming.
---

## Tech setup

- Python 3.11 via uv
- PyTorch, NumPy, matplotlib, jupyter
- No fixed environment — each subfolder has its own `pyproject.toml` if it needs one

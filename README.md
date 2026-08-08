# ml-journey

A 24-week, build-first path from shipping LLM applications to understanding and training the models underneath them — documented week by week, in public.

**Currently on: Week 4 complete — logistic regression, classification, regularization**

---

## Why this repo exists

I run [JustAutomateX](https://github.com/), a solo AI-automation agency for Indian SMBs. I ship production systems — n8n orchestration, LLM pipelines, WhatsApp Cloud API integrations, RAG on Supabase pgvector — for real paying clients.

But I was calling models, not understanding them. This repo is me closing that gap deliberately: from NumPy fundamentals through to building a GPT from scratch and fine-tuning open models.

Every week gets notes in my own words, drills implemented from scratch before reaching for a library, and — at each phase boundary — a mini-project framed as a real business problem.

---

## Roadmap

### Phase 0 — Foundations (Week 1) ✅

- [x] Environment: local Python + JupyterLab, Colab with GPU
- [x] NumPy — vectorization, broadcasting, boolean masks, axis behaviour
- [x] Pandas — loading, filtering, groupby, missing data
- [x] Kaggle Intro to ML — train/test split, under/overfitting, random forests

### Phase 1 — Classical ML (Weeks 2–8) 🔄

Andrew Ng, Machine Learning Specialization

- [x] Linear regression model
- [x] Cost function (squared error)
- [x] Gradient descent
- [x] Multiple linear regression, vectorization
- [x] Feature scaling and feature engineering
- [x] Logistic regression and classification
- [x] Overfitting and regularization
- [x] **Mini-project:** [lead scoring with logistic regression](mini-project-lead-scoring/) — from-scratch on UCI Bank Marketing (41k rows, F1=0.47 at tuned threshold)

### Phase 2 — Math foundations (parallel with Phase 1) 🔄

- [x] Linear algebra — vectors, matrices, transformations (3B1B ch. 1–5)
- [x] Calculus — derivatives, gradients, the chain rule (3B1B ch. 1–5)
- [ ] Probability and statistics

### Phase 3 — Deep learning (Weeks 9–16)

Karpathy, Neural Networks: Zero to Hero

- [ ] micrograd — autograd and backpropagation from scratch
- [ ] makemore — MLPs, activations, gradients, BatchNorm
- [ ] Build a GPT from scratch
- [ ] **Mini-project:** character-level language model

### Phase 4 — Production ML (Weeks 17–20)

- [ ] fast.ai Practical Deep Learning
- [ ] Kaggle competition entry
- [ ] Model deployment behind an API
- [ ] Evaluation done properly — precision, recall, imbalanced data
- [ ] **Mini-project:** a deployed model with an endpoint

### Phase 5 — LLMs and fine-tuning (Weeks 21–24)

Hugging Face LLM Course

- [ ] Transformers and the modern open-model workflow
- [ ] Fine-tuning with LoRA / QLoRA
- [ ] RAG evaluation and retrieval quality
- [ ] Structured extraction and LLM evals
- [ ] **Mini-project:** a fine-tuned model inside a live client workflow

---

## Week index

| Week | Topic | Notes | Status |
|---|---|---|---|
| 01 | Foundations — NumPy, Pandas, intro ML | [notes](week-01-foundations/notes.md) | ✅ Complete |
| 02 | Linear regression and the cost function | [notes](week-02-linear-regression/notes.md) | ✅ Complete |
| 03 | Multiple features, scaling, engineering | [notes](week-03-multiple-features/notes.md) | ✅ Complete |
| 04 | Logistic regression, classification, regularization | [notes](week-04-logistic-regression/notes.md) | ✅ Complete |

---

## How this repo is organised

```
week-NN-topic/
  notes.md      concepts in my own words, intuition before math
  *.py          implementations written from scratch, no libraries
  *.ipynb       exploration and drills
```

**Rule I follow:** implement it in raw NumPy first, understand it, *then* use the library version.

---

## Stack

Python · NumPy · Pandas · scikit-learn · Matplotlib · Jupyter

Started July 2026.

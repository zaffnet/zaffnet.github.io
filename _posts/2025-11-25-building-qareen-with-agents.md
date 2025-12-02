---
layout: post
title: "Building qareen: My Experience with Multi-Agent Coding"
desc: "Reflections on building a multimodal few-shot framework using a swarm of AI coding agents."
keywords: "AI Agents, Coding Assistants, qareen, Multimodal AI, LLM-as-a-Judge"
date:   2025-11-23 00:00:00 +0000
lastmod: 2025-11-23 00:00:00 +0000
comments: true
permalink: building-qareen-agents
---

I recently released [`qareen`](https://github.com/zaffnet/qareen), a framework designed to solve a specific problem in LLM evaluations: balancing relevance and diversity in few-shot examples. It extends Maximum Marginal Relevance (MMR) to multimodal tasks, helping LLM-as-a-Judge workflows avoid position bias and redundancy—because nobody wants a judging panel that just nods along like the Minions.

![Three agents handing artifacts to a human orchestrator](/assets/img/blog/qareen-swarm.svg)

## From Coder to Orchestrator

Building `qareen` with multiple agents felt less like writing every line myself and more like directing an Ocean's Eleven montage. One agent stitched together retrieval code, another riffed on evaluation notebooks, and a third made sure the UI actually booted. My job? Set the boundaries, review architecture, and keep the squad from stepping on each other's toes (or overwriting each other's `pytest` fixtures).

The speed of iteration was wild. I could test different ways to blend text and image signals—Weighted Linear Combination versus Reciprocal Rank Fusion (RRF), with a sprinkle of cosine similarity—before my coffee cooled. If an approach flopped, I pivoted faster than a jazz drummer dodging an off-beat cymbal.

## Quick tour: wiring an experiment

Here's a bite-sized sketch of how I ran a mini experiment while agents filled in the boilerplate:

```python
from qareen.rerank import mmr_multimodal
from qareen.scoring import reciprocal_rank_fusion

# Text + image scores from retrievers (courtesy of Agent #1)
text_scores = {"clip": 0.76, "siglip": 0.81}
image_scores = {"vision": 0.69}

# A/B test fusion strategies (courtesy of Agent #2)
rrf = reciprocal_rank_fusion([text_scores, image_scores], k=60)
mmr = mmr_multimodal(text_scores, image_scores, lambda_weight=0.3)

print("RRF winner:", max(rrf, key=rrf.get))
print("MMR winner:", max(mmr, key=mmr.get))
```

The orchestration layer (a.k.a. me) was mostly about guardrails: defining interfaces, keeping contexts tight, and reviewing diffs like a hawk with a latte.

## Lessons from the swarm

* **Context is King.** Agents are powerful, but only if you feed them clear interfaces and modular goals. Think Lego bricks, not mystery meatloaf.
* **Review over authoring.** I spent less time typing and more time catching hallucinations or subtle logic hiccups. The human-in-the-loop is still the QA boss.
* **Prototype like a DJ.** Spinning up a quick Gradio UI to visualize modality weights or alpha values took minutes. That feedback loop is pure creative fuel.

## A tiny retrospective scoreboard

| Task | Human time before | Human time with agents |
| --- | --- | --- |
| Write retrieval + rerank scaffold | ~3 hours | ~45 minutes |
| Build demo UI | ~2 hours | ~30 minutes |
| Cleanup + docs | ~1.5 hours | ~45 minutes |

## Conclusion

`qareen` is open source, and I invite you to check it out. It's a testament not just to the power of multimodal retrieval, but to a new way of building software—where human creativity is amplified by a playful swarm of digital assistants. Next time, I might even give them Ocean's Eleven-style code names; Agent Clooney does have a nice ring to it.

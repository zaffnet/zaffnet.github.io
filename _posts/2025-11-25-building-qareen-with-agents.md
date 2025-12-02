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

![Whiteboard with sticky notes and laptops](/assets/img/qareen-lab.svg)

I recently shipped [`qareen`](https://github.com/zaffnet/qareen), a framework for balancing relevance and diversity in few-shot examples. Under the hood it extends Maximum Marginal Relevance (MMR) to multimodal retrieval so LLM-as-a-Judge workflows dodge position bias and avoid spitting out the same four examples on repeat. The fun twist? I built it with a *swarm* of coding agents. Think "Ocean's Eleven," except everyone shows up on time and nobody steals the servers.

### From Coder to Orchestrator

Working with multiple agents turned me from keyboard jockey into band conductor. Instead of hand-writing every line, I focused on boundaries, architecture, and making sure the agents didn’t remix each other’s branches into free jazz. The pace was wild: testing Weighted Linear Combination vs. Reciprocal Rank Fusion (RRF) felt like speed-running a Kaggle comp with a pit crew.

A quick sketch of the ranking core that the agents and I iterated on:

```python
from qareen.retrievers import MultimodalMMR

ranker = MultimodalMMR(
    alpha=0.25,  # diversity vs. relevance
    weights={"text": 0.55, "vision": 0.45},
)

ranked = ranker.rank(query, candidates)
for item, score in ranked[:3]:
    print(f"{item.id}: {score:.3f}")
```

### Learnings from the Swarm

* **Context is King.** Agents are great at execution, but only if you feed them a clear interface and tight contracts. I spent more time writing docstrings than a courtroom drama binge.
* **Review over authoring.** My primary job became catching subtle logic tangles or hallucinated imports. The human-in-the-loop role felt closer to showrunner than scriptwriter.
* **Rapid prototyping.** Need a Gradio UI to visualize modality weights? The agents would draft one before my coffee cooled. Immediate feedback loops make research feel like a montage.

### Tiny Experiment Log

| Attempt | Blend | Takeaway |
| --- | --- | --- |
| Alpha 0.1 | Text-heavy | High relevance, but repeats city like a broken record. |
| Alpha 0.35 | Balanced | Sweet spot—diverse but still on-topic. |
| Alpha 0.6 | Diversity-max | Novel, yet sometimes invited the wrong guests to the party. |

### Qareen in the Real World

When I combined text and image cues for evaluation datasets, the agents helped me trace down edge cases like "GIFs with baked-in captions" that skew similarity scores. We ended up adding lightweight normalization steps before fusion. It’s the sort of sanding you forget until a swarm of assistants keeps finding splinters.

If you want to tinker with the same stack, clone the repo and try the demo notebook. Worst case, you end up with a weekend project; best case, you get your own agent ensemble and start humming the Avengers theme while CI runs.

### Conclusion

`qareen` is open source—come poke around, file issues, and tell me which fusion trick worked for you. It’s a story about multimodal retrieval, but also about a new flavor of engineering where humans set the vision and a fleet of helpful bots keeps the tempo.

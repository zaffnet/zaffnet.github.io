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

A few weeks ago I shipped [`qareen`](https://github.com/zaffnet/qareen), a framework for picking better few-shot examples. The algorithm is useful, but honestly? The interesting part was *how* I built it: by conducting a small orchestra of AI coding agents.

Imagine pair programming, except your pair is five different AIs with strong opinions about code style and a mysterious tendency to import libraries that don't exist.

### The gist of qareen

When you prompt an LLM with examples (few-shot learning), the examples matter. A lot. Give it five variations of the same thing and it'll parrot those patterns. Give it carefully selected, *diverse* examples and it generalizes better.

`qareen` picks examples that are relevant to your task but different enough from each other to actually teach something. It blends text and image signals—like making a mixtape where every song is both thematically appropriate *and* introduces something new.

![Hand-drawn sketch of the agent workflow](/assets/img/qareen-agents.svg)

### My new job: reviewer in chief

The biggest shift wasn't technical—it was how I spent my time. Before agents, I wrote code. Now I mostly review it. I went from keyboard-forward developer to someone who spends more time saying "wait, why would you do it that way?" to a robot.

This sounds like a downgrade until you realize: the iteration speed is wild. What used to be "set up an experiment, go get coffee, come back, realize you made a typo" became "propose five experiments, agents run all of them, pick the winner by lunch."

```python
from qareen.sampler import rr_rank, normalize_scores

def rerank_candidates(text_scores, image_scores, alpha=0.6):
    text = normalize_scores(text_scores)
    image = normalize_scores(image_scores)
    # Blend signals. Alpha was tuned through more experiments
    # than I want to admit.
    return rr_rank(text, image, weight=alpha)
```

### Patterns that actually worked

I tried a lot of things. Most didn't work. Here's what survived:

![Hand-drawn patterns that worked in building qareen](/assets/img/qareen-agent-patterns.svg)

**Planner → Builders.** One agent breaks issues into small, specific tasks. Others pick them up and execute. This sounds obvious, but getting the granularity right took some trial and error. Too big and agents get confused. Too small and you're managing a to-do list the length of a CVS receipt.

**Critic in the loop.** Before code hits my screen, a critic agent runs linting and tests. It's like having a very literal-minded coworker who catches the obvious stuff so I can focus on the subtle stuff.

**Keep logs of everything.** Every agent run dumps its commands and outputs to a log. When something breaks—and it will—you can replay the sequence to figure out what changed. Think git blame, but for agent decisions.

**Gradio for instant feedback.** I wired up a quick UI with sliders for the ranking weights. Being able to *see* the reranker's decisions made tuning dramatically faster than staring at JSON outputs.

### What I'd do differently

A few lessons learned the hard way:

* **Vague tasks = creative interpretations.** When I said "improve the reranker," one agent decided the way to do that was to rewrite the entire module in a different framework. Specific acceptance criteria are your friend.

* **Don't trust imports.** Agents will confidently import libraries that don't exist, or that exist but do something completely different. The critic pass caught most of these, but not all.

* **Human taste still matters.** Agents can optimize for metrics, but metrics don't always capture "does this actually feel right?" I kept a human checkpoint before anything shipped.

### The takeaway

Multi-agent coding isn't a toy or a demo—it's a genuinely different way to work. Not faster in *every* way (debugging agent confusion takes time), but faster in enough ways that the overall velocity goes up. You trade writing code for reviewing it, and if you're okay with that shift, it's pretty great.

`qareen` is open source if you want to try the framework. And if you build something with a swarm of agents, I'd love to hear which patterns worked for you—and which ones went hilariously wrong.

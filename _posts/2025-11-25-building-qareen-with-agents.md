---
layout: post
title: "Building qareen: My Experience with Multi-Agent Coding"
desc: "Reflections on building a multimodal few-shot framework using a swarm of AI coding agents."
keywords: "AI Agents, Coding Assistants, qareen, Multimodal AI, LLM-as-a-Judge"
date:   2025-11-25 00:00:00 +0000
lastmod: 2025-11-25 00:00:00 +0000
comments: true
permalink: building-qareen-agents
---

I recently released [`qareen`](https://github.com/zaffnet/qareen), a framework designed to solve a specific problem in LLM evaluations: balancing relevance and diversity in few-shot examples. It extends Maximum Marginal Relevance (MMR) to multimodal tasks, helping LLM-as-a-Judge workflows avoid position bias and redundancy.

But the most interesting part of building `qareen` wasn't just the algorithm itself—it was *how* I built it. I used a swarm of AI coding agents to accelerate the development process, and it taught me a lot about the changing nature of software engineering.

### From Coder to Orchestrator

Working with multiple agents simultaneously shifted my role from writing every line of code to orchestrating a team. Instead of getting bogged down in boilerplate, I found myself defining boundaries, reviewing architecture, and ensuring that the agents didn't step on each other's toes.

The speed of iteration was incredible. I could experiment with different strategies for combining text and image signals—like Weighted Linear Combination versus Reciprocal Rank Fusion (RRF)—much faster than if I were coding solo. If an approach didn't work, I could pivot immediately.

### Learnings from the Swarm

Here are a few key takeaways from this experience:

*   **Context is King:** Agents are powerful, but they need clear context. Defining strict interfaces and modular components allowed the agents to work autonomously on different parts of the system without breaking the whole.
*   **Review over Authoring:** My time was spent less on typing and more on code review. Catching subtle logic errors or hallucinations became the primary task. The "human in the loop" is essential for quality control.
*   **Rapid Prototyping:** The ability to spin up a Gradio UI to visualize modality weights or test different alpha values for MMR happened in minutes, not hours. This immediate feedback loop is a game-changer for research engineering.

### Conclusion

`qareen` is open source, and I invite you to check it out. It's a testament not just to the power of multimodal retrieval, but to a new way of building software—where human creativity is amplified by a swarm of tireless digital assistants.

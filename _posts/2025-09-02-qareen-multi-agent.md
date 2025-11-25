---
layout: post
title: "Notes from building qareen with multiple coding agents"
desc: "A quick reflection on coordinating specialized agents inside qareen."
keywords: "qareen, coding agents, orchestration, reflections"
date:   2025-09-02 12:00:00 +0000
lastmod: 2025-09-02 12:00:00 +0000
comments: true
permalink: qareen-multi-agents
---

I have been wiring up qareen—the open-source agentic platform I use to explore how many coding agents can safely co-create a codebase. The project is on GitHub: [qareen](https://github.com/zaffnet/qareen).

Here are the biggest lessons so far:

* **Give each agent a sharp edge.** A "planner" agent that only handles decomposition and acceptance criteria makes reviews calmer. A "fixer" agent that only touches a slice of the tree keeps merge conflicts down.
* **Tee up deterministic checks.** Lightweight lint and contract tests running after every agent step prevent error cascades. It is easier to intervene early than to unwind a patch chain.
* **Keep memory boring.** Instead of long prompts, I cache structured traces (intent, files touched, assertions) and replay them. It reduces drift when three agents try to solve the same ticket.
* **Humans stay in the loop.** I still gate major decisions—renames, migrations, and speculative refactors—because human taste matters. Agents that ask permission at the right moments feel like teammates, not autopilots.

qareen is forcing me to design conversation patterns, not just model prompts. It is a good reminder that coordination is the real product when you work with multiple agents.

---
layout: post
title: "What building Qareen taught me about multi-agent coders"
desc: "Reflections on orchestrating multiple coding agents at Pinterest."
keywords: "Qareen, multi-agent, coding agents, Pinterest, LLM"
date:   2025-10-05 12:00:00 +0000
lastmod: 2025-10-05 12:00:00 +0000
comments: true
permalink: qareen-multi-agent-coding
---

I've been building **Qareen** at Pinterest—a multi-agent coding teammate that borrows the best parts of pair programming and production reliability. Keeping it short, here are the lessons that keep resurfacing:

1. **Agents need roles, not vibes.** Planning, execution, verification, and narration deserve separate agents with clear contracts. When I merged roles, regressions spiked; when I split them, defects were easier to trace.
2. **Context hygiene beats fancy prompting.** Qareen's best gains came from ruthless pruning of scratchpads and tool output. Shallow memory with crisp pointers trumped long, sentimental transcripts.
3. **Verifiers are teammates, not hall monitors.** A lightweight critic agent that reran commands and diffed expectations rescued more PRs than any clever chain-of-thought trick.
4. **Evaluations have to mirror messy reality.** Synthetic tasks were fine for smoke tests, but dogfooding on hairy service repos revealed the real race conditions and API rough edges.
5. **Ship the rails, not just the brain.** Interfaces, fallbacks, and observability mattered more to users than which LLM we used that week. Reliability is the product.

Working with multiple coding agents feels less like managing bots and more like coaching a small, opinionated team. Qareen made that team dependable.

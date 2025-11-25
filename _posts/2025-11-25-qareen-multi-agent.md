---
layout: post
title: "Notes from Building Qareen"
desc: "What I'm learning while wrangling a team of coding agents."
keywords: "multi-agent, coding agents, Qareen, developer tools"
date:   2025-11-25 00:00:00 +0000
lastmod: 2025-11-25 00:00:00 +0000
comments: true
permalink: qareen-multi-agent
---

Qareen started as a weekend experiment to see how far a team of specialized coding agents could push a codebase without me babysitting every decision.
Here are the lessons that keep coming up:

* **Delegation beats depth-first prompts.** Splitting work between a planner, fixer, and reviewer cut dead ends dramatically. The reviewer catches silent regressions that a single agent misses.
* **Eval harnesses are the real product.** The agents only got useful when I wired them to tests, static analysis, and linting. Shipping new tools now means writing the checks first, then letting agents iterate until the harness goes green.
* **Guardrails need receipts.** Every agent action is logged with diffs and rationale before commits. That paper trail makes it easy to revert bad ideas and coach the agents with better context.
* **Human in the loop stays lightweight.** My role is to curate prompts and provide taste—setting constraints, picking libraries, and vetoing scope creep. The agents handle the rest at machine speed.

Qareen is still scrappy, but it already feels like pair programming with a reliable crew that never gets tired of refactoring.

---
layout: post
title: "Qareen: Notes from building with multiple coding agents"
desc: "A short reflection on what I'm learning while orchestrating agentic coding workflows."
keywords: "qareen, multi-agent systems, coding agents, LLM"
date:   2026-05-18 12:00:00 +0000
lastmod: 2026-05-18 12:00:00 +0000
comments: true
permalink: qareen-multi-agent-notes
---

I've been shipping a lean multi-agent coding orchestrator called [qareen](https://github.com/zaffnet/qareen). The goal is simple: make it easier to coordinate a handful of specialised agents without drowning in YAML or brittle prompt soup.

Here are the notes that keep sticking:

- **Roles beat monoliths.** Keeping agents small, opinionated, and short-lived makes failures obvious and recovery cheap. A reviewer agent only cares about deltas and acceptance criteria; a planner agent only sketches three steps ahead.
- **Explicit contracts calm the swarm.** Shared message schemas and tight tool signatures cut down on "oops, wrong format" loops. I default to structured JSON everywhere, even when the natural language is tempting.
- **Human-in-the-loop is a feature, not a fallback.** Qareen is happiest when I gate risky actions (git pushes, infra changes) behind a quick human confirmation. It preserves trust without slowing the flow.
- **Telemetry matters early.** Lightweight tracing on agent turns surfaces flaky steps quickly—especially when multiple LLM providers behave differently.

Building qareen feels like pairing with a tiny team of reliable colleagues: they don't replace judgment, but they keep me honest and focused.

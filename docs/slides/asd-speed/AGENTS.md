# ASD Speed — talk scope

## Purpose

Internal team deck for the Friday 2026-07-17 meeting. Explains, to the whole team,
**how the Active Speaker Detection (ASD) pipeline works** and **why it is slow** —
so we can align on where to spend engineering effort to hit the customer target.

## Audience

Small startup team. Mixed: engineers who know the code, plus product/leadership who
need the story without reading `main.py`. Assume video/ML literacy, not ASD-internals
literacy.

## Core message (do not dilute)

1. ASD is the bottleneck component of the long-to-short product.
2. On real 102-min HD content it runs at **1.04× realtime**. Customer wants **12×**.
3. The `detect` stage is **68%** of the time and it is **CPU-decode-bound — the GPU is idle**
   (`sm=0`, `dec=0` during detect). It is NOT the model, NOT a too-small GPU.
4. Two intuitions to correct on Friday: "TalkNet fusion is the bottleneck" (false — model is ~2.6%)
   and "a bigger GPU fixes it" (false — GPU is 0% busy). The "the read is the bottleneck" instinct
   was correct.
5. Existing YOLO batching optimized the one stage that was never the bottleneck.
6. Path to 12× exists but needs the full fix stack (strided detection + downscale-on-decode +
   true NVDEC/GPU tensors). Safe near-term customer promise = throughput via sharding.

## Guardrails

- All numbers come from the 102-min A10 run and the 7-min fixture — see
  `docs/research/2026-07-15-asd-speed-bottleneck.md` and the four `memory/` notes. Do not invent
  new figures.
- Honest tone. This deck admits the 7-min fixture flattered us (1.96× vs real 1.04×).
- No promising single-video 12× as a committed date; frame it as "possible but needs the stack".
- Keep brand: white canvas, Brandbox gradient for heroes/pills only, purple ≤10%.

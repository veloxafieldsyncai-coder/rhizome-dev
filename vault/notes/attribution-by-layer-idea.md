---
title: Hunch, attribution by layer changes model behavior
category: concepts
layer: brainstorm
tags: [raft, training, grounding, hypothesis]
summary: Provisional hunch that training the model to attribute notes and only quote research verbatim will measurably change how it grounds answers.
sources: [my-notes]
lifecycle: draft
created: 2026-06-14
updated: 2026-06-14
---

# Hunch, attribution by layer changes model behavior

My provisional thinking, to be tested.

My hunch is that if the RAFT training data quotes research verbatim but only ever attributes
my notes as provisional, the trained model will learn to say "the literature states" for facts
and "your notes propose" for my ideas, without me hard coding any rule at inference time.

The thing I am unsure about: whether a small number of cross-layer examples is enough for the
model to generalize the attribution behavior, or whether it needs a large balanced set.

I also suspect the asymmetric P split matters: never dropping the note from context should stop
the model from memorizing my speculation as fact. I have not verified this yet.

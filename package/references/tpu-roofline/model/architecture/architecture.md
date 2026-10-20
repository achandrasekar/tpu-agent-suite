---
name: model-architecture-format
description: Guidelines and standard template for documenting model architectures for roofline analysis.
---

# Model Architecture Documentation Guidelines

When documenting a model's architecture for roofline analysis (e.g., in `skills/model/architecture/<family>/<model>.md`), please follow these brief guidelines:

## 1. Required References
You **MUST** use the standard tensor dimension naming conventions defined in `../../NOTATION.md` (e.g., `D`, `F`, `H_q`, `D_h`, `L`, `P`). Do not invent your own ad-hoc variables.

## 2. Global Model Parameters
You **MUST** include all possible information necessary for Roofline calculations. At a minimum, provide a clear section explaining the model parameters. The recommended way is to provide a Markdown table listing the fundamental hyperparameters.

For example:
*   Total Parameters
*   Layers ($L$)
*   Hidden Dimension ($D$)
*   FFN Intermediate Dim ($F$)
*   Query Heads ($H_q$)
*   Head Dimension ($D_h$)
*   KV Heads ($H_{kv}$)

## 3. Layer Structure & Tensor Flow
You are **free** to describe the model's structure and tensor flow in whatever way makes the most sense. The only requirement is to write it so that **another AI agent can easily read and understand it** to construct an accurate Python script for Roofline estimation. You can use text, math formulas, bullet points, or code snippets as you see fit.

## 4. Updating Notation
If you need to introduce a new mathematical symbol or tensor dimension that is not currently covered, you **MUST** update `../../NOTATION.md` to define and document the new symbol appropriately.

## 5. Reference Existing Structures
When creating new skills, reference existing subcomponents under the current directory and try to follow a similar structure.

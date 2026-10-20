---
name: model-common-component-format
description: Guidelines and standard template for documenting common model components (e.g., attention, ffn, moe) for roofline analysis.
---

# Common Component Documentation Guidelines

When documenting a common model component for roofline analysis (e.g., in `skills/model/common/<component>/<type>.md`), please follow these brief guidelines:

## 1. Required References
You **MUST** use the standard tensor dimension naming conventions defined in `../../NOTATION.md` (e.g., `D`, `F`, `H_q`, `D_h`, `L`, `P`). Do not invent your own ad-hoc variables.

You **MUST** also adhere to the calculation and output rules defined in `REQUIREMENTs.md` within this directory (e.g., using Python, `TYPE_PROMOTION.md`, standard units, and the JSON output format).

## 2. Component Operations
Provide a clear breakdown of the mathematical operations involved in the component. For each operation (e.g., QKV projection, Self-Attention, Gating), specify:
*   Input and Output tensor shapes
*   FLOPS calculations
*   Memory transfer (reads and writes)
*   Collective communication (if any)

## 3. Flexibility and Flow
You are **free** to describe the component's internal operation flow in whatever way makes the most sense. The only requirement is to write it so that **another AI agent can easily read and understand it** to construct an accurate Python script for Roofline estimation. You can use text, math formulas, bullet points, or code snippets as you see fit.

## 4. Updating Notation
If you need to introduce a new mathematical symbol or tensor dimension that is not currently covered, you **MUST** update `../../NOTATION.md` to define and document the new symbol appropriately.

## 5. Reference Existing Structures
When creating new skills, reference existing subcomponents under the current directory and try to follow a similar structure.

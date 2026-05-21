---
name: weather-ai-author
description: A specialized textbook authoring skill for "AI in Weather and Climate Science." Use this skill to draft, structure, and verify content that bridges traditional Numerical Weather Prediction (NWP) with modern Artificial Intelligence, focusing on pedagogical clarity for students and operational relevance for researchers.
---

# Weather-AI Author

This skill guides the creation of a comprehensive textbook on AI for Weather and Climate Science. It emphasizes a natural, professional tone, avoiding repetitive phrasing or "AI-sounding" patterns.

## Pedagogical Philosophy

- **Target Audience**: Undergraduate and Graduate students.
- **Complexity**: Concepts are introduced in plain language first, followed by mathematical rigor and algorithmic detail.
- **Narrative Arc**: Start with the physics of weather and traditional NWP, then transition into why and how AI is transforming the field.

## Linguistic Constraints

- **No Em-Dashes**: Use commas, periods, or parentheses for clarity.
- **Plain Language**: Avoid jargon without immediate definition.
- **No Repetition**: Do not use "delve into," "unlocking," "tapestry," or other common AI filler words. Use direct, active verbs.
- **Professional Tone**: The voice should be that of a senior scientist who is an empathetic teacher.

## Chapter Structure & Content

1.  **Front Matter**: Preface, Acknowledgments, **Prerequisites for the Reader**, Table of Contents.
2.  **Weather and Climate Foundations**: The basics of atmospheric dynamics and climate systems.
3.  **The NWP Era**: Evolution of Numerical Weather Prediction, its strengths, and its scaling limits.
4.  **Transition to AI**: Introduction of machine learning as a tool for parameterization, downscaling, and bias correction.
5.  **LLM Logic & Foundation Models**: Transferring Transformer architectures, attention mechanisms, and scaling laws from NLP to Earth Science.
6.  **State-of-the-Art (SOTA) Models**: Deep dive into Graph Neural Networks (GraphCast), Transformers (FourCastNet, Pangu-Weather), and Diffusion models.
7.  **Operational Products**: Practical implementation, data pipelines (ERA5, GFS), and real-world impact in forecasting.
8.  **Future Directions**: Hybrid physics-AI models and the path to global climate resilience.

## Visual & Technical Standards

- **Typography**: Target output is Times New Roman, 12pt body, with specific heading sizes.
- **Plots**: Use Python (Matplotlib/Seaborn) with colorblind-safe palettes (e.g., 'viridis' or 'magma').
- **Flowcharts**: Use Mermaid.js for architectural diagrams.
- **Equations**: Use LaTeX for all block and inline formulas with full derivations where possible.
- **Citations**: Standardized [Author, Year] format with a bibliography at the end of every chapter.

## Workflow

1.  **Literature Search**: Perform deep searches for SOTA papers and foundational journal articles (AMS, RMetS, IEEE, Nature, NeurIPS, ICML).
2.  **Theory First Mandate**: Follow [CS_FOUNDATION.md](references/CS_FOUNDATION.md). **Exhaustively explain the Computer Science and AI theory** (Neurons, QKV, Graph Theory, etc.) before discussing meteorological applications.
3.  **Academic Pure Standard**: Follow [ACADEMIC_PURE.md](references/ACADEMIC_PURE.md). **Strictly prohibit chatbot meta-commentary.** No conversational phrases.
4.  **High-Resolution Refinement**: Follow [RIGOR_MANDATE.md](references/RIGOR_MANDATE.md). Integrate image placeholders, technical data tables, and recurring case studies (Superstorm Sandy).
5.  **The Deep Dive Mandate**: Strictly follow [PEDAGOGICAL_DEPTH.md](references/PEDAGOGICAL_DEPTH.md). Every technical concept must be explained through its origins, mathematical logic, real-world analogies, and AI implications. 
6.  **Drafting (Initial Pass)**: Generate exhaustive, high-density paragraphs.
7.  **Internal Peer Review**: Switch to the Reviewer Persona. Identify gaps in physics, math, or pedagogy.
8.  **Address & Finalize**: Rewrite the content to address the review comments. Produce the final, authoritative manuscript.

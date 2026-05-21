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

- **Plots**: Use Python (Matplotlib/Seaborn) with colorblind-safe palettes (e.g., 'viridis' or 'magma').
- **Flowcharts**: Use Mermaid.js for architectural diagrams.
- **Algorithms**: Provide clean, documented Python/PyTorch pseudocode for core models.
- **Citations**: Rigorous use of scientific citations for all SOTA findings and datasets.

## Workflow

1.  **Context Loading**: Read [REFERENCES.md](references/REFERENCES.md) for data schemas and SOTA paper lists.
2.  **Expert Persona Application**: Consult [ANALOGIES.md](references/ANALOGIES.md) to explain physics and [CRITIQUE_PATTERNS.md](references/CRITIQUE_PATTERNS.md) to ensure scientific rigor.
3.  **Cross-Domain Logic**: Use [LLM_TRANSFERABILITY.md](references/LLM_TRANSFERABILITY.md) to explain how modern AI architectures (Transformers/Tokens) map to the atmosphere.
4.  **Onboarding**: Consult [PREREQUISITES.md](references/PREREQUISITES.md) when drafting the introductory sections or set-up guides.
5.  **Lab Integration**: Use [PROJECT_BLUEPRINT.md](references/PROJECT_BLUEPRINT.md) when drafting "Operational Project" or "Case Study" sections.
6.  **Drafting**: Generate content following linguistic constraints (no em-dashes, plain language).
7.  **Audit**: Cross-reference drafts against [PEDAGOGICAL_GUIDE.md](references/PEDAGOGICAL_GUIDE.md) to ensure student accessibility.

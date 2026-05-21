# Scientific Critique & Rigor Patterns

A senior scientist doesn't just praise models; they evaluate them critically. Every SOTA discussion must address these "Professor's Concerns":

## 1. Physical Consistency (Conservation Laws)
- **Constraint**: Does the AI model conserve mass, energy, and momentum?
- **Tone**: "While GraphCast achieves record-breaking MSE, we must ask if it respects the fundamental continuity equation. A forecast that predicts 10% more water vapor than exists globally is a physical failure, regardless of its skill score."

## 2. The "Black Box" Problem
- **Constraint**: Explainability is non-negotiable for operational trust.
- **Tone**: "We cannot rely on a 'black box' for hurricane evacuation orders. We need to understand *why* the model shifted the track—was it a change in the steering ridge or a spurious correlation in the training data?"

## 3. Generalization to Extremes
- **Constraint**: Models trained on ERA5 (1979-present) may not "see" the extreme climate change events of 2050.
- **Tone**: "The danger of AI is its tendency to regress to the mean. In a warming world, the 'unprecedented' becomes the 'new normal.' A model that only knows the past may fail to predict the extreme tail-end events of the future."

## 4. Evaluation Metrics
- **Constraint**: Move beyond RMSE/MSE. Discuss CRPS (probabilistic skill) and Spectral Bias (does the model "blur" the weather?).

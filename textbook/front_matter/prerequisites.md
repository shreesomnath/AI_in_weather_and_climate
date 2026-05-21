# Prerequisites for the Reader

To successfully navigate this textbook and complete the operational projects, readers should possess a foundational set of skills. We have designed the content to be accessible to undergraduates, but certain technical "languages" are necessary to speak with the data and the models.

## 1. Programming with Python
Python is the lingua franca of the AI and weather science communities. You should be comfortable with:
*   **Basic Logic**: Loops, conditional statements, and data structures (lists, dictionaries).
*   **The Scientific Stack**: Familiarity with `NumPy` for numerical arrays and `Matplotlib` or `Seaborn` for data visualization.
*   **Xarray and NetCDF**: Weather data is almost always multi-dimensional. Understanding how to use `xarray` to open and manipulate NetCDF files is the single most important technical skill for this book.

## 2. Mathematical Foundations
We avoid unnecessary complexity, but the following concepts are foundational:
*   **Calculus**: A basic understanding of gradients and partial derivatives, which are used both in atmospheric physics and in the "backpropagation" of neural networks.
*   **Linear Algebra**: Concept of vectors and matrices (tensors), as these are how computers store atmospheric states.
*   **Probability and Statistics**: Understanding means, variances, and the concept of a "probability distribution" for ensemble forecasting.

## 3. Atmospheric Basics
While we introduce concepts as we go, it is helpful to have a general intuition for:
*   **Fluid Motion**: The idea that the atmosphere is a fluid on a rotating sphere.
*   **Conservation Laws**: The principle that mass and energy cannot be created or destroyed, only moved or transformed.

## 4. Artificial Intelligence Concepts
If you are new to AI, do not worry. We explain the core concepts in Chapter 4. However, it is useful to know:
*   **The Training Process**: The idea of showing a model data, measuring its error, and adjusting it to improve.
*   **AI Agents**: A basic awareness of how AI can assist in coding and data management, as we will use these tools in our lab projects.

If you meet these prerequisites, you are ready to engage with the full depth of this textbook. If you find yourself struggling with a specific concept, we provide a glossary and further reading lists in the appendices.

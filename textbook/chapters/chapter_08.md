# Chapter 8: Fourier and 3D Transformers: FourCastNet and Pangu-Weather

The evolution of planetary-scale AI has been defined by a constant struggle between **Physical Fidelity** and **Computational Efficiency**. As we explored in *Chapter 6*, the Transformer architecture provides the "Global Vision" required to capture teleconnections, but its quadratic memory cost ($O(N^2)$) creates an insurmountable wall at high resolutions. This chapter explores the two breakthrough architectures that have successfully breached this wall: **FourCastNet** and **Pangu-Weather**. We will examine how FourCastNet leverages the mathematical elegance of the **Fourier Transform** to perform learning in the frequency domain, and how Pangu-Weather utilizes a hierarchical **3D Transformer** design to respect the volumetric nature of the Earth. By combining the "Spectral Wisdom" of the past with the "Neural Attention" of the future, these models have established a new ceiling for what is possible in high-resolution global meteorology.

## 8.1 Learning in the Frequency Domain: Adaptive Fourier Neural Operators (AFNO)

The conceptual origin of **FourCastNet** is a return to the "Musical Logic" we discussed in *Section 2.3*: the idea that the atmosphere can be better understood as a sum of oscillating waves rather than a collection of isolated grid points. While most deep learning models (like CNNs) operate in the **Spatial Domain**—sliding filters over pixels—FourCastNet performs its primary learning in the **Frequency Domain**. It utilizes the **Adaptive Fourier Neural Operator (AFNO)**, a mathematical framework developed by researchers at NVIDIA and Caltech. The AFNO architecture is built upon the realization that many complex fluid dynamics problems, which are notoriously difficult to solve in physical space, become elegantly simple when viewed through their spectral frequencies. In our digital book, this is the "Spectral Bridge" between AI and classical NWP: we are using the neural network to "Learn the Score" of the atmospheric symphony directly in the language of the Fourier transform.

The mathematical heart of the AFNO layer involves a three-step process: the **Fast Fourier Transform (FFT)**, the **Frequency-Domain Interaction**, and the **Inverse FFT**. For an input weather tensor $X$, the layer first projects the data into frequency space:

$$\hat{X} = \text{FFT}(X) \quad (\text{Eq. 8.1})$$

Once in the frequency domain, the model applies a learned weight matrix $W$ to the complex-valued spectral coefficients. This is the **Neural Operator** step, where the AI "decides" which frequencies are most important for the forecast. Unlike a standard convolution, this operation is **Global**: because every spectral coefficient contains information from the entire spatial grid, a single multiplication in frequency space can capture long-range teleconnections across the entire planet. Finally, the updated coefficients are transformed back into the spatial domain:

$$X_{out} = \text{IFFT}(\text{Softmax}(\hat{X} \cdot W)) \quad (\text{Eq. 8.2})$$

This process allows the network to learn a **Global Stencil** that adapts to the atmospheric state, effectively combining the multi-scale vision of a spectral model with the non-linear flexibility of a deep neural network.

***Pause and Reflect:*** Consider the physical analogy of **Timbre and Pitch** in music. When you listen to an orchestra, your ear does not record the pressure of the sound wave at every microsecond; instead, your brain performs an intuitive "Fourier Transform," allowing you to distinguish the high-pitched violins (small-scale features) from the deep, low-pitched cellos (global-scale features). In weather AI, how does the **Frequency-Domain Interaction** in Eq. 8.2 act as this "Planetary Ear"? If a model learns to amplify the low-frequency coefficients during an El Niño year, is it "Hearing" the slow pulse of the Pacific?

The AI connection here is the successful creation of **Global Weather Emulators** that can run at $0.25^\circ$ resolution without crashing the GPU's memory. By using the FFT, which has a computational complexity of only **$O(N \log N)$**, FourCastNet bypasses the $O(N^2)$ quadratic wall of standard Transformers and the $O(N^3)$ wall of traditional spectral models. This efficiency allows the model to capture the **Entire Global Atmospheric State** in a single pass, evolving variables like temperature, wind, and moisture with startling accuracy. FourCastNet was the first AI model to prove that a data-driven emulator could match the skill of the IFS model on short-to-medium ranges while running **45,000 times faster**. This represented a "Zero-to-One" moment for the field: it proved that the mathematical elegance of the frequency domain was not a relic of the past, but the essential engine for the next generation of high-speed planetary intelligence.

**Table 8.1: Comparison of Spectral and Spatial Learning**

| Feature | CNN (Spatial) | Transformer (Attention) | AFNO (Frequency) |
| :--- | :--- | :--- | :--- |
| **Field of View** | Local (Filter size) | Global (All-to-all) | Global (Spectral) |
| **Complexity** | $O(N)$ | $O(N^2)$ | $O(N \log N)$ |
| **Physics Bias** | Local Gradients | Teleconnections | Wave Oscillations |
| **Operational Limit** | Multi-layer stacking | Memory (VRAM) | High-frequency detail |

## 8.2 Pangu-Weather: The 3D Earth-Specific Transformer

The architectural alternative to the spectral vision of FourCastNet is found in **Pangu-Weather**, a model developed by Bi et al. (2023) at Huawei Research that has fundamentally redefined the limits of **Volumetric Atmospheric Emulation**. While previous models struggled with the "VRAM Barrier" of 3D convolutions, Pangu-Weather utilizes a highly optimized **3D Transformer** design to capture the depth and stratification of the atmosphere. The conceptual origin of this model is the realization that the atmosphere is not a collection of flat 2D maps, but a single, continuous 3D volume where vertical interactions (such as the rising of warm air in a thunderstorm) are as physically important as horizontal winds.

The mathematical heart of Pangu-Weather is the **3D-v2 Transformer Block**, which incorporates a critical innovation known as **Earth-Specific Positional Bias (ESPB)**. As we explored in *Section 6.2*, standard Transformers require positional encodings to understand where data is located. Pangu-Weather takes this a step further by realizing that the physics of the atmosphere are **Anisotropic**: the rules that govern motion in the horizontal $(x, y)$ dimensions are fundamentally different from the rules that govern motion in the vertical $(z)$ dimension due to the overpowering influence of gravity. The ESPB term ($B$) is added directly to the attention matrix (Eq 6.2) to force the model to learn these physical asymmetries:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V \quad (\text{Eq. 8.3})$$

In this expression, $B$ is a learned matrix that encodes the physical "Distance and Direction" between all pairs of 3D tokens. Eq. 8.3 allows the network to learn that a change in pressure at 200 hPa in the jet stream has a specific, weighted physical relationship with the temperature at the surface, effectively embedding the **Hydrostatic and Geostrophic Balances** directly into the neural architecture.

***Pause and Reflect:*** Consider the physical analogy of an **Atmospheric MRI**. In a medical MRI, the machine takes thousands of cross-sectional images of the body and uses math to assemble them into a 3D model. If the machine only looked at the body from the front (2D), it would miss a tumor hidden behind a bone. In meteorology, if an AI model only looks at surface maps, it misses the **Upper-Level Waves** that drive the storms. How does the **3D Attention** in Eq. 8.3 act as this "Neural Depth Perception"? Does it allow the AI to "See Through" the clouds to understand the deep physical causes of the weather?

## 8.3 Summary: The Spectral-Attention Unification

The journey through the architectures of FourCastNet and Pangu-Weather brings us to a singular, transformative conclusion: we have successfully unified the **Spectral Elegance** of the 20th century with the **Neural Attention** of the 21st. For decades, the meteorological community was divided between those who believed in the global waves of the spectral method and those who believed in the localized grids of finite-volume physics. These two breakthrough models have proven that both perspectives are essential for high-fidelity planetary emulation. 

As we move forward into the Generative and Physics-Informed architectures of the final part of this book, we must carry with us the fundamental lesson of Chapter 8: **Architecture is a Physical Prior**. The success of Pangu-Weather and FourCastNet was not just a result of more data; it was a result of designing neural layers that "Understand" the Earth. Whether it is the frequency-domain adaptive weights of AFNO or the Earth-specific positional bias of ESPB, the "Intelligence" of these models is rooted in their respect for the **Symmetries and Scale of the Planet**.

> ### **Case Study Anchor: Sandy’s Spectral Signature**
> *In the forecasting of Superstorm Sandy, traditional models struggled with the "Phasing" of the storm with a mid-latitude trough. This phasing is essentially a synchronization of two different spatial frequencies. A spectral model like FourCastNet, operating in the frequency domain, can identify the "Resonance" between these two waves hours before a grid-point model can resolve the interaction. This ability to capture multi-scale resonance is why spectral AI is now the gold standard for predicting the most complex, multi-system interactions in our atmosphere.*

## Bibliography

*   Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., and Tian, Q. (2023). Accurate medium-range global weather forecasting with 3D neural networks. *Nature*, 619(7970), 533-538.
*   Guibas, S., et al. (2021). Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers. *arXiv preprint arXiv:2111.13587*.
*   Kurth, T., et al. (2022). FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators. *arXiv preprint arXiv:2202.11214*.
*   Li, Z., et al. (2020). Fourier neural operator for arbitrary partial differential equations. *arXiv preprint arXiv:2010.08895*.

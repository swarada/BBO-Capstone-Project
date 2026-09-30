Model Card for BBO Optimization Approach
========================================

-   **Overview**:

    -   **Approach Name**: Hybrid Multi-Surrogate Black-Box Optimization (BBO) Framework.

    -   **Model Type**: Iterative Bayesian Optimization and machine learning surrogate pipeline utilizing Gaussian Processes, Neural Networks, and Logistic Regression.

    -   **Version**: Version 2.0 (Stage 2 Capstone Iteration 13+).

-   **Intended Use**:

    -   **Suitable Tasks**: Optimizing unknown, expensive, or gradient-free black-box functions ranging from 2D to 8D across synthetic benchmarks simulating real-world tasks like robot control, sensor tuning, and drug discovery.

    -   **Avoided Use Cases**: Direct analytical reverse-engineering of hidden mathematical equations, high-dimensional unconstrained regression without parameter bounds checking, or real-time safety-critical control loops without fallback protections.

-   **Details**:

    -   **Strategy Evolution Across Rounds**: The approach evolved iteratively across submission cycles. Early rounds utilized random search and basic Gaussian Processes (`SingleTaskGP`) with adaptive UCB heuristics (\beta scaling). Subsequent iterations experimented with feed-forward Neural Networks (`NeuralNetworkSurrogate`) paired with random candidate sampling, followed by a classification-based approach using Logistic Regression with median-based output binarization. Recent rounds adopted a hybrid setup combining BoTorch GP optimization for stable functions with alternative architectures where appropriate.

-   **Performance**:

    -   **Summary of Results**: Evaluated across eight distinct synthetic functions scaling from 2D to 8D. Weekly cumulative updates (reaching 12+ data points per function) consistently improved output response values, successfully locating strong regional maxima.

    -   **Metrics Used**: Scalar performance output values (Y), maximum observed outputs (Y.max()), output standard deviation (Y.std()), and normalized Min-Max scaling scores.

-   **Assumptions and Limitations**:

    -   **Assumptions**: Assumes that surrogate models can reliably approximate complex black-box topologies and that maximizing acquisition functions or surrogate predictions leads toward global optima.

    -   **Constraints & Failure Modes**: Subject to sparse sampling gaps, localized clustering biases, and increased computational overhead for exact GP inference or neural network training. Handled via duplicate-avoidance distance checks and randomized perturbation fallbacks (`perturbation_scale = 1e-4`).

-   **Ethical Considerations & Reflections**:

    -   **Transparency & Reproducibility**: Full documentation through version-controlled code repositories, fixed random seeds (`torch.manual_seed(42)`), and explicit historical data archives (`initial_inputs.npy`, `initial_outputs.npy`) ensures complete reproducibility and transparent decision-making.

    -   **Clarity of Structure**: Adding detailed iteration logs, explicit decision heuristics, and failure fallback descriptions directly improves clarity, ensuring researchers and peers can fully understand how decisions are made under incomplete information.
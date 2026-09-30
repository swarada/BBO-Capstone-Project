### Datasheet for the BBO Capstone Project Data Set

#### 1\. Motivation

-   **Why was this data set created?** The data set was created as part of the Black-Box Optimization (BBO) capstone project to simulate real-world optimization challenges (such as robot control, drug discovery, or sensor tuning) where objective functions are opaque, expensive, or lack analytical gradients.

-   **What task does it support?** It supports sequential decision-making, surrogate modeling (Gaussian Processes, Neural Networks, and Logistic Regression), and acquisition function optimization to discover the global maximum of eight synthetic black-box functions.

#### 2\. Composition

-   **What does it contain?** It contains historical input-output pairs (`initial_inputs.npy` and `initial_outputs.npy`) augmented by weekly evaluated query results (`weekly_data_to_add`) across eight distinct functions.

-   **What is the size and format?**

    -   Dimensions scale from 2D up to 8D depending on the target function.

    -   Stored in NumPy binary array format (`.npy`) and structured Python dictionaries containing floating-point parameter vectors and scalar performance signals.

-   **Are there any gaps?** Yes, the data set represents a sparse sampling of a continuous search space, limited by a restricted query budget (one query per function per week).

#### 3\. Collection Process

-   **How were the queries generated?** Queries were generated iteratively using a variety of machine learning surrogate strategies, including random search, Gaussian Processes with Upper Confidence Bound (UCB) / Expected Improvement (EI), feed-forward Neural Networks, and Logistic Regression classification models.

-   **What strategy was used?** A hybrid exploration-exploitation strategy was applied, utilizing adaptive hyperparameter heuristics (such as dynamically scaling UCB's \beta parameter) and duplicate-avoidance perturbation fallbacks.

-   **Over what time frame?** Collected sequentially across multiple weekly submission cycles (spanning up to 13+ iterative rounds).

#### 4\. Preprocessing and Uses

-   **Have you applied any transformations?**

    -   Input filtering to remove invalid parameter configurations exceeding the strict bound of 0.999999.

    -   Min-Max scaling for comparative visualization.

    -   Output binarization (splitting around the median) when framing optimization as a classification task for Logistic Regression.

-   **What are the intended uses?** Training surrogate models, computing acquisition values, and guiding automated weekly experimental queries.

-   **What are inappropriate uses?** Direct analytical derivation of the underlying hidden mathematical equations or treating the synthetic data as an unconstrained general-purpose regression benchmark without bounds checking.

#### 5\. Distribution and Maintenance

-   **Where is the data set available?** Stored locally and within Google Drive project directories (`/content/drive/My Drive/Colab Notebooks/`) alongside version-controlled GitHub code repositories.

-   **What are the terms of use?** Restricted to educational and research use within the context of the BBO capstone project curriculum.

-   **Who maintains it?** Maintained and updated iteratively by the project researcher/developer through weekly capstone submission cycles.
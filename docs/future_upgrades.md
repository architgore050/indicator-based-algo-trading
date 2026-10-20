# Future Upgrades & Considerations

This document outlines planned improvements and exploratory ideas for the future development of the indicator-based algo trading system. 

While repository maintenance is identified as a necessary next step, the other two items are currently **exploratory considerations**. They are hypotheses for future improvement that still need to be discussed and validated to determine if they are actually beneficial for our specific use case.

---

## 1. Proper Repository Maintenance (Required)

As the project has grown from a collection of scripts into a complex system with a backend, frontend, and heavy computational pipelines, our repository hygiene needs to catch up. This is a foundational requirement for long-term sustainability.

**Key areas of focus:**
*   **Folder Structure Cleanup:** Reorganize directories to clearly separate raw data, processed data, core library code, executable scripts, and configuration files. Ensure the root directory isn't cluttered.
*   **Documentation:** Expand and maintain comprehensive documentation. This includes updating `README.md`, ensuring docstrings are complete and accurate, and creating onboarding guides for the backend/frontend architecture.
*   **Logging:** Standardize logging across the entire codebase. Move away from scattered `print()` statements to a unified, configurable logging system (e.g., using Python's `logging` module with structured JSON outputs for the backend) to make debugging and monitoring easier.
*   **Testing:** Implement a robust testing suite. This includes unit tests for indicator calculations, integration tests for the backtest engine, and end-to-end tests for the API endpoints. 

---

## 2. Reconsidering the Optuna Sampler Choice (To Be Discussed)

Currently, the Optuna calibration pipeline relies on the default TPE (Tree-structured Parzen Estimator) sampler. While TPE is generally a strong, sample-efficient Bayesian optimization algorithm, it may not be the optimal choice for the specific characteristics of financial time-series search spaces.

**Things to consider and discuss:**
*   **Search Space Characteristics:** Financial strategies often have highly noisy, non-convex, and multi-modal objective functions. Does TPE handle this well, or do we need a sampler that explores more aggressively?
*   **Alternative Samplers:** Should we evaluate other samplers provided by Optuna or third-party extensions? 
    *   *CMA-ES (Covariance Matrix Adaptation Evolution Strategy):* Excellent for continuous, non-separable, and ill-conditioned search spaces.
    *   *Random Search:* Sometimes surprisingly competitive in very high-dimensional or highly noisy spaces where Bayesian methods overfit the surrogate model.
    *   *Multi-objective Samplers (e.g., NSGA-II):* If we eventually move beyond a single metric (like Sharpe Ratio) and want to optimize for multiple conflicting goals simultaneously (e.g., Return vs. Drawdown).
*   **Decision Pending:** We need to analyze the optimization landscapes of our current strategies before deciding if a sampler change will yield materially better robustness.

---

## 3. Exposing Hardcoded Indicator Sensitivity Parameters (To Be Discussed)

In the current implementation, several technical indicators rely on hardcoded "sensitivity" parameters (e.g., specific smoothing periods, secondary moving average lengths, or volatility multipliers) that dictate how the indicator reacts to price action. Currently, the Optuna sampler only calibrates the primary strategy thresholds, leaving these underlying indicator mechanics static.

**Things to consider and discuss:**
*   **The Proposal:** Expose these hardcoded sensitivity parameters to the Optuna search space, allowing the sampler to calibrate the indicator's internal mechanics alongside the strategy's entry/exit rules.
*   **The Risk (Curse of Dimensionality):** Adding more parameters exponentially increases the size of the search space. This could lead to significantly longer calibration times and increase the risk of the optimizer finding "overfitted" parameter combinations that work perfectly on historical data but fail in live markets.
*   **The Benefit:** It could unlock hidden performance by allowing the indicator to adapt its sensitivity to the specific volatility and regime of the asset being traded, rather than relying on generic defaults.
*   **Decision Pending:** We need to carefully weigh the potential performance gains against the risks of overfitting and increased compute time. If we decide to proceed, we may need to implement stricter regularization or use Walk-Forward Validation more aggressively to ensure the calibrated sensitivities are robust.

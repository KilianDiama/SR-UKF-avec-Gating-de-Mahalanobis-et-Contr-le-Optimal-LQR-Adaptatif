⚡ Engineered by Kiliandiama | The Diama Protocol [10/10] | All rights reserved.
AlphaEngine_V10_Perfected

Full Square-Root Unscented Kalman Filter + Mahalanobis Guard + Adaptive LQR Controller

Table of Contents

Description

Features

Installation

Usage

Parameters and Configuration

Algorithms

Example

Limitations and Possible Improvements

License

Description

AlphaEngine_V10_Perfected is an advanced algorithmic trading engine for asset allocation and prediction.

It combines:

SR-UKF (Square-Root Unscented Kalman Filter) for robust estimation of latent state variables: [Price, Drift, Acceleration].

Mahalanobis Gate to ignore extreme observations (flash crashes/outliers).

Adaptive LQR Controller to dynamically compute optimal asset weights based on current volatility.

Cholesky Rank-1 Updates for maximum numerical stability.

Designed to trade leveraged positions while mitigating extreme market risks.

Features

Dynamic state estimation [Price, Drift, Acceleration] using SR-UKF.

Adaptive volatility tracking using a simple GARCH-style model.

Mahalanobis gating for outlier filtering and flash-crash protection.

Adaptive LQR for optimal position sizing with volatility-aware leverage costs.

Enhanced output: trading weight, estimated drift, and confidence score.

Max leverage enforcement via max_leverage parameter.

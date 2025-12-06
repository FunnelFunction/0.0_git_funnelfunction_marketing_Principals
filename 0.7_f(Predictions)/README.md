# 0.7 f(Predictions)

**Falsifiable Forecasts: Specific Claims That Can Be Proven Wrong**

The testable predictions that validate or invalidate the Funnel Function.

---

## Abstract

A scientific framework must make **specific, falsifiable predictions**. This layer documents the concrete claims of the Funnel Function—predictions precise enough to be proven wrong. Each prediction includes the test methodology, success criteria, and what falsification would mean for the theory.

---

## Part 1: Core Predictions

### Prediction 1: The Writability Gate Effect

```
CLAIM: Leads with W(x) < 0.3 will convert at rates < 1%.
       Leads with W(x) > 0.7 will convert at rates > 20%.

FALSIFICATION: If W(x) < 0.3 segment converts at > 5%, the writability
              gate is not functioning as theorized.

TEST PROTOCOL:
1. Score all leads via W(x) model
2. Segment: W < 0.3, W 0.3-0.7, W > 0.7
3. Track 90-day conversion rates by segment
4. Compute confidence intervals

SUCCESS METRIC: Monotonic increase in conversion rate by W segment
               with statistically significant differences (p < 0.05).
```

### Prediction 2: Time Discount Effect

```
CLAIM: Lead value decays at rate γ per month, where γ ∈ [0.90, 0.99].
       A lead converting in month 3 is worth γ³ of a month-0 conversion.

FALSIFICATION: If conversion value is independent of time (γ = 1),
              the time discount component adds no value.

TEST PROTOCOL:
1. Measure deal value by time-in-funnel
2. Fit exponential decay: V(t) = V₀ × γᵗ
3. Compare r² of decay model vs constant model
4. Estimate γ with confidence interval

SUCCESS METRIC: γ significantly < 1 (p < 0.05) and r²(decay) > r²(constant).
```

### Prediction 3: Activation Threshold

```
CLAIM: Conversion occurs when ∫A dτ > θ for some threshold θ.
       Below-threshold leads do not convert regardless of other factors.

FALSIFICATION: If activation integral has no predictive power for
              conversion, the accumulation model is wrong.

TEST PROTOCOL:
1. Compute ∫A dτ for all leads at conversion/non-conversion
2. Compare distributions (converters vs non-converters)
3. Test if threshold exists via ROC analysis
4. Measure AUC for activation as predictor

SUCCESS METRIC: AUC > 0.65 and significant difference in activation
               distributions (KS test p < 0.01).
```

### Prediction 4: Drift Equilibrium

```
CLAIM: Well-calibrated systems maintain D = ∇Ψ/∇Φ ≈ 1.0 ± 0.2
       over time. Persistent deviation indicates model failure.

FALSIFICATION: If D is random (no central tendency around 1.0),
              the model has no predictive validity.

TEST PROTOCOL:
1. Compute D daily for 90 days
2. Test if mean(D) significantly different from 1.0
3. Test if D has lower variance than random baseline
4. Check for mean reversion after calibration

SUCCESS METRIC: mean(D) ∈ [0.8, 1.2] with 95% confidence and
               variance(D) < variance(random baseline).
```

---

## Part 2: Behavioral Predictions

### Prediction 5: Loss Aversion Ratio

```
CLAIM: Loss-framed messages outperform gain-framed messages by
       factor λ ≈ 2.0-2.5 (consistent with Prospect Theory).

FALSIFICATION: If loss/gain ratio is 1.0 or < 1.0, loss aversion
              does not apply as theorized.

TEST PROTOCOL:
1. A/B test: Gain frame vs Loss frame on identical offer
2. Measure conversion rate for each
3. Compute ratio: CR(loss) / CR(gain)
4. Test if ratio significantly > 1

SUCCESS METRIC: Loss/Gain ratio ∈ [1.5, 3.0] with p < 0.05.
```

### Prediction 6: Nurture Decay Function

```
CLAIM: Engagement decays exponentially: E(t) = E₀ × exp(-λt)
       with channel-specific λ values.

FALSIFICATION: If decay is linear, step-function, or random,
              the exponential model is wrong.

TEST PROTOCOL:
1. Track engagement score over time without intervention
2. Fit exponential decay model
3. Compare r² against linear, step, and random models
4. Estimate λ by channel with confidence intervals

SUCCESS METRIC: Exponential model r² > 0.7 and superior to alternatives.
```

### Prediction 7: Optimal Cadence Formula

```
CLAIM: Optimal nurture cadence Δt* = (1/λ) × ln(E_threshold/E₀)
       maximizes engagement while minimizing fatigue.

FALSIFICATION: If random or fixed cadence outperforms calculated
              optimal, the formula has no value.

TEST PROTOCOL:
1. Three-arm test: Random vs Fixed vs Calculated Δt*
2. Measure MQL conversion rate over 90 days
3. Compare rates with statistical significance

SUCCESS METRIC: Calculated Δt* outperforms both alternatives (p < 0.05).
```

---

## Part 3: Business Predictions

### Prediction 8: Shapley Attribution Accuracy

```
CLAIM: Shapley value attribution predicts incremental contribution
       more accurately than last-touch or equal attribution.

FALSIFICATION: If holdout experiments show Shapley no more accurate
              than simpler methods, the complexity is unwarranted.

TEST PROTOCOL:
1. Compute attribution three ways: Last-touch, Equal, Shapley
2. Run randomized holdout experiments by channel
3. Compare predicted vs actual lift for each method
4. Compute MAPE for each attribution method

SUCCESS METRIC: MAPE(Shapley) < MAPE(Last-touch) with p < 0.05.
```

### Prediction 9: CLV Prediction Accuracy

```
CLAIM: CLV = Σ γᵗ × Revenue(t) × P(retention|t) predicts customer
       lifetime value within 20% MAPE over 24 months.

FALSIFICATION: If MAPE > 40%, the model fails to capture real dynamics.

TEST PROTOCOL:
1. Compute predicted CLV at customer acquisition
2. Track actual 24-month revenue
3. Compute MAPE: mean(|predicted - actual| / actual)
4. Segment accuracy by customer type

SUCCESS METRIC: Overall MAPE < 25% with no segment MAPE > 35%.
```

### Prediction 10: 97% Efficiency Principle

```
CLAIM: Processing only W(x) > θ leads achieves ≥ 90% of total
       conversions with ≤ 10% of total processing effort.

FALSIFICATION: If filtering to high-W leads misses > 20% of
              conversions, the efficiency claim is overstated.

TEST PROTOCOL:
1. Score all leads via W(x)
2. Process top 10% by W score
3. Measure conversions from processed vs total potential
4. Compute efficiency ratio

SUCCESS METRIC: Capture rate > 90% with processing rate < 15%.
```

---

## Part 4: Tensor Predictions

### Prediction 11: Six-Fan Balance

```
CLAIM: Organizations with balanced ICHTB fans (all six operational)
       outperform those with imbalanced fans.

FALSIFICATION: If fan balance has no correlation with business
              performance, the ICHTB framework is descriptive only.

TEST PROTOCOL:
1. Assess fan health scores for multiple organizations
2. Compute balance metric: min(fan_i) / max(fan_i)
3. Correlate balance with business performance metrics
4. Test significance of correlation

SUCCESS METRIC: Correlation r > 0.3 with p < 0.05.
```

### Prediction 12: Collapse Probability Calibration

```
CLAIM: P_collapse = exp(-(ΔΨ)²/2σ²) is well-calibrated such that
       predicted X% probability leads convert at ~X% rate.

FALSIFICATION: If calibration curve slope significantly ≠ 1,
              the probability model is miscalibrated.

TEST PROTOCOL:
1. Compute P_collapse for all opportunities
2. Bin by predicted probability (0-10%, 10-20%, ..., 90-100%)
3. Compute actual conversion rate per bin
4. Plot calibration curve and fit slope

SUCCESS METRIC: Calibration slope ∈ [0.8, 1.2] and Brier score < 0.15.
```

---

## Part 5: Prediction Registry

### Summary Table

| # | Prediction | Test Type | Success Metric | Risk Level |
|---|------------|-----------|----------------|------------|
| 1 | Writability gate | Segmentation | Monotonic CR by W | High |
| 2 | Time discount | Regression | γ < 1 significant | Medium |
| 3 | Activation threshold | ROC analysis | AUC > 0.65 | High |
| 4 | Drift equilibrium | Time series | D ∈ [0.8, 1.2] | Medium |
| 5 | Loss aversion | A/B test | Ratio ∈ [1.5, 3.0] | Low |
| 6 | Decay function | Curve fitting | r² > 0.7 | Medium |
| 7 | Optimal cadence | Multi-arm test | Δt* beats alternatives | Medium |
| 8 | Shapley accuracy | Holdout | MAPE < last-touch | High |
| 9 | CLV prediction | Longitudinal | MAPE < 25% | High |
| 10 | 97% efficiency | Filtering | 90% capture at 15% effort | High |
| 11 | Six-fan balance | Correlation | r > 0.3 | Medium |
| 12 | Collapse calibration | Calibration | Slope ∈ [0.8, 1.2] | High |

### Risk Assessment

```
High risk: Core theory component—falsification requires major revision
Medium risk: Important but adjustable—falsification requires recalibration
Low risk: Derived claim—falsification requires minor adjustment
```

---

## Part 6: Implementation

### Prediction Testing Engine

```python
class PredictionTestEngine:
    """
    Engine for running falsifiable prediction tests.
    """

    def __init__(self, config: 'Config'):
        self.predictions = self.load_predictions()
        self.test_results = {}

    def run_all_tests(self, data: 'DataFrame') -> 'TestReport':
        """
        Run all prediction tests against provided data.
        """
        results = {}

        for pred_id, prediction in self.predictions.items():
            result = self.run_test(prediction, data)
            results[pred_id] = result

        return TestReport(
            results=results,
            all_passed=all(r.passed for r in results.values()),
            falsified=[k for k, v in results.items() if not v.passed and v.p_value < 0.01],
            needs_recalibration=[k for k, v in results.items() if not v.passed and v.p_value >= 0.01]
        )

    def run_test(self, prediction: 'Prediction', data: 'DataFrame') -> 'TestResult':
        """
        Run a single prediction test.
        """
        test_func = getattr(self, f'test_{prediction.id}')
        return test_func(prediction, data)

    def test_writability_gate(self, pred: 'Prediction', data: 'DataFrame') -> 'TestResult':
        """Test Prediction 1: Writability gate effect."""
        # Segment data
        low_w = data[data['writability'] < 0.3]
        mid_w = data[(data['writability'] >= 0.3) & (data['writability'] < 0.7)]
        high_w = data[data['writability'] >= 0.7]

        # Compute conversion rates
        cr_low = low_w['converted'].mean()
        cr_mid = mid_w['converted'].mean()
        cr_high = high_w['converted'].mean()

        # Test monotonicity
        monotonic = cr_low < cr_mid < cr_high

        # Test significance
        from scipy.stats import chi2_contingency
        contingency = [
            [low_w['converted'].sum(), len(low_w) - low_w['converted'].sum()],
            [high_w['converted'].sum(), len(high_w) - high_w['converted'].sum()]
        ]
        _, p_value, _, _ = chi2_contingency(contingency)

        passed = monotonic and p_value < 0.05 and cr_low < 0.05

        return TestResult(
            prediction_id='writability_gate',
            passed=passed,
            p_value=p_value,
            effect_size=cr_high - cr_low,
            details={
                'cr_low': cr_low,
                'cr_mid': cr_mid,
                'cr_high': cr_high,
                'monotonic': monotonic
            }
        )
```

---

## Part 7: Response Protocol

### What to Do If Falsified

```
If prediction falsified:

1. Document the failure
   ├── Which prediction failed
   ├── How it failed (direction, magnitude)
   ├── Data and methodology
   └── Reproducibility check

2. Assess impact
   ├── Core theory component → Major revision needed
   ├── Derived claim → Minor adjustment
   └── Implementation issue → Fix without theory change

3. Revise model
   ├── Identify the broken assumption
   ├── Formulate alternative hypothesis
   ├── Test alternative
   └── Update documentation

4. Publish results
   ├── Transparency requirement
   ├── Update prediction registry
   └── Version the theory
```

### Scientific Integrity

```
Commitment:
├── All tests pre-registered before data collection
├── All results published regardless of outcome
├── Negative results are scientific progress
└── Theory evolves based on evidence
```

---

## Contents

| Section | Description |
|---------|-------------|
| [0.7.a f(Falsifiable_Claims)](./0.7.a_f(Falsifiable_Claims)/) | Detailed test specifications |

---

## Summary

### The Prediction Principle

> **Every claim must be specific enough to be proven wrong. A theory that can't be falsified isn't science.**

### The 12 Core Predictions

1. Writability gates conversion
2. Time discounts value
3. Activation threshold exists
4. Drift equilibrium holds
5. Loss aversion applies
6. Decay is exponential
7. Optimal cadence formula works
8. Shapley beats last-touch
9. CLV prediction accurate
10. 97% efficiency achievable
11. Six-fan balance matters
12. Collapse probability calibrated

### The Action

**Test every prediction. Publish every result. Revise on falsification.**

---

## References

- Popper, K. (1959). The Logic of Scientific Discovery.
- Kahneman, D. & Tversky, A. (1979). Prospect Theory.
- Knight, A. & Khan, A. (2025). The Funnel Function.

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Created by Armstrong Knight & Abdullah Khan

# 0.5 f(Validation_Framework)

**Scientific Validation: Falsifiability and Empirical Testing**

The framework for proving or disproving the Funnel Function through rigorous experimentation.

---

## Abstract

A theory that cannot be falsified is not science. This layer establishes the **validation protocols** for empirically testing the Funnel Function against alternative models. We define specific, measurable predictions that can be proven wrong, compare against established frameworks (Marketing Mix Modeling, Multi-Touch Attribution), and design incrementality tests that isolate causal effects.

---

## The Falsifiability Requirement

### Popper's Criterion

```
For f(x) to be scientific:
├── It must make specific predictions
├── Those predictions must be testable
├── Tests must be capable of disproving the theory
└── Negative results must force model revision
```

### The Funnel Function Predictions

```
f(x) = W(Φ,Ψ,ε) · γᵗ · ∫₀ᵗ A(u,m,τ) dτ

Predicts:
1. W = 0 → f(x) = 0 (non-writables never convert)
2. γᵗ decay follows discount curve
3. ∫A dτ > θ required for conversion
4. D = ∇Ψ/∇Φ should ≈ 1 if model is correct
```

---

## Part 1: Core Falsifiable Predictions

### Prediction 1: Writability Gating

```
H₁: Leads with W(x) < 0.3 will convert at rate < 1%
H₀: Conversion rate independent of W(x)

Test:
├── Score all leads via W(x) model
├── Segment: W < 0.3, W 0.3-0.7, W > 0.7
├── Track conversion rates by segment
└── Compare rates with statistical significance

Falsification criterion:
If W < 0.3 segment converts at > 5%, reject W gating hypothesis
```

### Prediction 2: Time Discount

```
H₁: Conversion value follows γᵗ decay (γ ≈ 0.95-0.99 monthly)
H₀: Conversion value independent of time delay

Test:
├── Measure value of leads converting at different delays
├── Fit γᵗ decay curve
├── Compare r² of decay model vs constant model

Falsification criterion:
If r²(decay) < r²(constant), reject time discount hypothesis
```

### Prediction 3: Activation Threshold

```
H₁: Conversion occurs when ∫A dτ > θ
H₀: Conversion random with respect to activation

Test:
├── Compute cumulative activation for all leads
├── Compare activation at conversion vs non-conversion
├── Test threshold model predictive power

Falsification criterion:
If activation distribution identical for converters/non-converters, reject
```

### Prediction 4: Drift Equilibrium

```
H₁: Well-functioning systems maintain D ≈ 1.0 ± 0.2
H₀: D values random

Test:
├── Compute D across multiple campaigns
├── Measure D distribution
├── Test if mean(D) significantly different from 1

Falsification criterion:
If D persistently < 0.5 or > 2.0 despite calibration, model fundamentally wrong
```

---

## Part 2: Comparison with Marketing Mix Modeling

### MMM Framework

```
Sales = β₀ + Σᵢ βᵢ · Media_i + Σⱼ γⱼ · Control_j + ε

Where:
β₀ = Baseline sales
βᵢ = Media coefficient for channel i
γⱼ = Control variable coefficient
ε = Error term
```

### Funnel Function vs MMM

| Aspect | MMM | Funnel Function |
|--------|-----|-----------------|
| Level | Aggregate | Individual + Aggregate |
| Attribution | Channel-level | Touchpoint-level |
| Mechanism | Statistical correlation | Causal mechanism |
| Real-time | No | Yes |
| Writability | Not modeled | Core concept |
| Memory | Adstock decay | ∇×F curl operator |

### Comparative Test Protocol

```
Design:
├── Run both models on same data
├── Generate predictions for holdout period
├── Compare MAPE (Mean Absolute Percentage Error)
├── Compare directional accuracy

Metrics:
├── MAPE_MMM vs MAPE_ff
├── Correlation(predicted, actual) for each
└── Out-of-sample R² for each

Success criterion:
f(x) valid if MAPE_ff ≤ MAPE_MMM × 1.1 (within 10%)
```

### Structural Advantages Test

```
Where f(x) should outperform MMM:
├── Individual-level predictions
├── Real-time optimization
├── Non-writable filtering efficiency
└── New customer acquisition accuracy

Test:
For each domain, measure relative performance
```

---

## Part 3: Incrementality Testing

### The Gold Standard

```
Incrementality = Causal lift attributable to treatment

True_lift = Y(treatment) - Y(control)

Where Y = Outcome under each condition
```

### Test Designs

#### Randomized Controlled Experiment

```
Design:
├── Random split: 50% treatment, 50% control
├── Treatment receives marketing
├── Control receives no/reduced marketing
├── Measure outcome difference

Analysis:
Lift = (Conversion_treatment - Conversion_control) / Conversion_control

Statistical test:
├── Two-sample t-test or chi-square
├── Require p < 0.05
└── Compute 95% CI for lift
```

#### Geographic Lift Test

```
Design:
├── Match similar geographic regions
├── Randomly assign to treatment/control
├── Run campaign in treatment regions only
├── Compare outcomes

Analysis:
Lift = Σ(Y_treatment - Y_control_matched) / Σ Y_control_matched
```

#### Ghost Ads / Intent-to-Treat

```
Design:
├── Identify users who would see ad
├── Randomly withhold ad from subset
├── Compare conversion: saw ad vs would have seen

Analysis:
True incrementality = P(convert | saw) - P(convert | would have but didn't)
```

### Funnel Function Incrementality Prediction

```
f(x) predicts:

Incremental_value = Σₓ∈writables [f(x,treatment) - f(x,control)]

Test:
├── Compute predicted incrementality
├── Run holdout experiment
├── Compare predicted vs actual
└── Validate within confidence interval
```

---

## Part 4: A/B Testing Framework

### Funnel Function A/B Protocol

```
For any model change:

1. Define hypothesis
   H₁: Change improves D or conversion rate
   H₀: No improvement

2. Power analysis
   n = (z_α + z_β)² × 2σ² / δ²
   Where δ = minimum detectable effect

3. Randomization
   ├── Hash-based user assignment
   ├── Verify balance on key covariates
   └── Document assignment mechanism

4. Measurement
   ├── Primary metric: conversion rate or D
   ├── Secondary: engagement, revenue, etc.
   └── Guardrails: quality, satisfaction

5. Analysis
   ├── Intent-to-treat analysis
   ├── Per-protocol sensitivity
   └── Subgroup effects
```

### Sample Size Calculator

```python
def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Compute required sample size per group for A/B test.
    """
    import scipy.stats as stats

    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    p_pooled = (p1 + p2) / 2

    numerator = (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2

    return int(math.ceil(numerator / denominator))
```

---

## Part 5: Model Validation Metrics

### Predictive Accuracy

```
MAPE = (1/n) Σᵢ |Predicted_i - Actual_i| / Actual_i

RMSE = √[(1/n) Σᵢ (Predicted_i - Actual_i)²]

R² = 1 - Σ(Actual - Predicted)² / Σ(Actual - Mean)²
```

### Classification Metrics (for W(x))

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)

AUC-ROC = Area under ROC curve
```

### Calibration Metrics

```
For probability predictions:
├── Brier Score = (1/n) Σ(p_i - o_i)²
├── Calibration curve: plot predicted vs actual rates
└── Expected Calibration Error (ECE)
```

### Target Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| MAPE | < 30% | < 20% | < 10% |
| R² | > 0.3 | > 0.5 | > 0.7 |
| AUC-ROC | > 0.6 | > 0.7 | > 0.8 |
| Brier Score | < 0.25 | < 0.15 | < 0.10 |

---

## Part 6: Specific Test Cases

### Test Case 1: Writability Filtering Efficiency

```
Claim: Processing only W(x) > θ leads achieves same conversions with 97% less effort

Test:
├── Control: Process all leads
├── Treatment: Process only W(x) > θ
├── Measure: Conversions, Cost, Efficiency

Success criterion:
Treatment achieves ≥ 90% of control conversions at ≤ 10% of cost
```

### Test Case 2: Drip Calculus Optimization

```
Claim: Optimal cadence Δt* = (1/λ) × ln(E_threshold/E₀) maximizes engagement

Test:
├── Variant A: Random cadence
├── Variant B: Fixed 7-day cadence
├── Variant C: Calculated optimal Δt*
├── Measure: MQL conversion rate

Success criterion:
Variant C outperforms A and B by > 20%
```

### Test Case 3: Prospect Theory Framing

```
Claim: Loss framing outperforms gain framing by factor of λ ≈ 2.25

Test:
├── Control: Gain frame ("Save $100")
├── Treatment: Loss frame ("Stop losing $100")
├── Measure: Conversion rate ratio

Success criterion:
Treatment/Control ratio between 1.5 and 3.0 (consistent with λ = 2.25 ± variation)
```

### Test Case 4: Collapse Probability Accuracy

```
Claim: P_collapse = exp(-(ΔΨ)²/2σ²) predicts conversion rates

Test:
├── Compute P_collapse for all opportunities
├── Bin by predicted probability
├── Compare predicted vs actual rates per bin

Success criterion:
Calibration curve slope between 0.8 and 1.2
```

---

## Part 7: Implementation Architecture

### Validation Engine

```python
class ValidationEngine:
    """
    Scientific validation framework for the Funnel Function.
    """

    def __init__(self, config: ValidationConfig):
        self.model = FunnelFunctionModel(config)
        self.baseline_models = {
            'mmm': MarketingMixModel(config),
            'mta': MultiTouchAttribution(config),
            'last_touch': LastTouchModel()
        }
        self.test_registry = TestRegistry()

    def run_comparative_validation(
        self,
        data: DataFrame,
        holdout_ratio: float = 0.2
    ) -> ComparisonResult:
        """
        Compare f(x) against baseline models.
        """
        # Split data
        train, test = train_test_split(data, test_size=holdout_ratio)

        # Train all models
        self.model.fit(train)
        for name, baseline in self.baseline_models.items():
            baseline.fit(train)

        # Predict on holdout
        predictions = {
            'funnel_function': self.model.predict(test),
            **{name: model.predict(test)
               for name, model in self.baseline_models.items()}
        }

        # Compute metrics
        metrics = {}
        for name, pred in predictions.items():
            metrics[name] = {
                'mape': mean_absolute_percentage_error(test.outcome, pred),
                'rmse': root_mean_squared_error(test.outcome, pred),
                'r2': r2_score(test.outcome, pred),
                'correlation': pearsonr(test.outcome, pred)[0]
            }

        # Determine winner
        best_model = min(metrics, key=lambda x: metrics[x]['mape'])

        return ComparisonResult(
            metrics=metrics,
            best_model=best_model,
            ff_competitive=(metrics['funnel_function']['mape'] <=
                          min(m['mape'] for n, m in metrics.items()
                              if n != 'funnel_function') * 1.1)
        )

    def run_incrementality_test(
        self,
        treatment_data: DataFrame,
        control_data: DataFrame
    ) -> IncrementalityResult:
        """
        Measure true causal effect and compare to prediction.
        """
        # Actual incrementality
        actual_lift = (
            treatment_data.conversions.mean() -
            control_data.conversions.mean()
        ) / control_data.conversions.mean()

        # Predicted incrementality
        predicted_treatment = self.model.predict(treatment_data)
        predicted_control = self.model.predict(control_data)
        predicted_lift = (
            predicted_treatment.mean() - predicted_control.mean()
        ) / predicted_control.mean()

        # Confidence interval for actual
        se = math.sqrt(
            treatment_data.conversions.var() / len(treatment_data) +
            control_data.conversions.var() / len(control_data)
        )
        ci_lower = actual_lift - 1.96 * se
        ci_upper = actual_lift + 1.96 * se

        # Validate prediction
        prediction_valid = ci_lower <= predicted_lift <= ci_upper

        return IncrementalityResult(
            actual_lift=actual_lift,
            predicted_lift=predicted_lift,
            ci=(ci_lower, ci_upper),
            prediction_valid=prediction_valid,
            p_value=ttest_ind(treatment_data.conversions,
                             control_data.conversions).pvalue
        )

    def run_falsifiability_tests(self) -> FalsifiabilityReport:
        """
        Run all core falsifiability tests.
        """
        results = {}

        for test_name, test_spec in self.test_registry.tests.items():
            result = test_spec.execute()
            results[test_name] = {
                'passed': result.passed,
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'details': result.details
            }

        return FalsifiabilityReport(
            tests=results,
            all_passed=all(r['passed'] for r in results.values()),
            falsified=any(not r['passed'] and r['p_value'] < 0.01
                         for r in results.values())
        )
```

### Test Registry

```python
class TestRegistry:
    """
    Registry of all falsifiable predictions and their tests.
    """

    def __init__(self):
        self.tests = {
            'writability_gating': WritabilityGatingTest(),
            'time_discount': TimeDiscountTest(),
            'activation_threshold': ActivationThresholdTest(),
            'drift_equilibrium': DriftEquilibriumTest(),
            'loss_aversion_ratio': LossAversionTest(),
            'decay_function': DecayFunctionTest()
        }

    def add_test(self, name: str, test: FalsifiabilityTest):
        self.tests[name] = test

    def run_all(self, data: DataFrame) -> Dict[str, TestResult]:
        results = {}
        for name, test in self.tests.items():
            results[name] = test.execute(data)
        return results
```

---

## Part 8: Continuous Validation

### Live Monitoring

```
Continuous validation requirements:
├── Daily drift quotient computation
├── Weekly calibration checks
├── Monthly model comparison
├── Quarterly full validation suite
```

### Automated Alerts

```python
def continuous_validation_check(
    model: FunnelFunctionModel,
    new_data: DataFrame
) -> ValidationStatus:
    """
    Daily validation check for production model.
    """
    predictions = model.predict(new_data)
    actuals = new_data.outcomes

    drift = compute_drift_quotient(predictions, actuals)
    calibration = compute_calibration_error(predictions, actuals)
    mape = compute_mape(predictions, actuals)

    alerts = []

    if drift < 0.5 or drift > 2.0:
        alerts.append(Alert('critical', 'Major drift detected'))
    elif drift < 0.8 or drift > 1.2:
        alerts.append(Alert('warning', 'Drift outside tolerance'))

    if calibration > 0.15:
        alerts.append(Alert('warning', 'Model miscalibrated'))

    if mape > 0.30:
        alerts.append(Alert('warning', 'Prediction accuracy degraded'))

    return ValidationStatus(
        drift=drift,
        calibration=calibration,
        mape=mape,
        healthy=len(alerts) == 0,
        alerts=alerts
    )
```

---

## Part 9: Failure Modes and Response

### What Would Falsify f(x)?

```
Core falsifications:
├── W(x) = 0 entities convert at same rate as W(x) = 1
├── Time delay has no effect on value (γ = 1)
├── Activation integral uncorrelated with conversion
├── D persistently ≠ 1 despite calibration
└── Simpler models consistently outperform f(x)
```

### Response to Falsification

```
If falsified:
├── Document the failure mode
├── Identify the component that failed
├── Revise that component specifically
├── Retest the revised model
└── Publish results either way (scientific integrity)
```

---

## Contents

| Section | Description |
|---------|-------------|
| [0.5.a f(MMM_Comparison)](./0.5.a_f(MMM_Comparison)/) | Head-to-head with Marketing Mix Models |
| [0.5.b f(Incrementality_Testing)](./0.5.b_f(Incrementality_Testing)/) | Causal inference experiments |
| [0.5.c f(Falsifiable_Predictions)](./0.5.c_f(Falsifiable_Predictions)/) | Specific testable claims |

---

## Summary

### The Validation Principle

> **A theory is only as good as its ability to be proven wrong. The Funnel Function must survive rigorous empirical testing or be revised.**

### The Core Equations

1. **Falsifiability:** D = ∇Ψ/∇Φ ≈ 1 if model valid
2. **Incrementality:** True_lift = Y(treatment) - Y(control)
3. **Sample size:** n = (z_α + z_β)² × 2σ² / δ²
4. **Comparison:** MAPE_ff ≤ MAPE_baseline × 1.1

### The Action

**Test, measure, falsify or validate, publish results.**

---

## References

- Popper, K. (1959). The Logic of Scientific Discovery.
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
- Kohavi, R. et al. (2020). Trustworthy Online Controlled Experiments.
- Knight, A. & Khan, A. (2025). The Funnel Function.

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Created by Armstrong Knight & Abdullah Khan

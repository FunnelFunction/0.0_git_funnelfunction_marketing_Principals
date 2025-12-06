# 0.2.b.ii f(Qualification)

**The Writability Gate: Formalizing Collapse Permission**

The mathematics of determining which leads satisfy the conditions for sales execution.

---

## Abstract

Qualification is the **writability gate**. Not all captured leads are writable—many exist in the field but cannot be collapsed. This framework formalizes the qualification function `W(x) = δ(Φ-Ψ) > ε` as the gating condition that filters leads into writables and non-writables, ensuring sales resources are allocated only where collapse is permitted.

---

## Part 1: The Writability Gate Equation

### Core Definition

```
f(Qualification) = W(x) = δ(Φ(x) - Ψ(x)) > ε
```

Where:
- **Φ(x)** = Lead's intent state (what they want)
- **Ψ(x)** = Offer state (what you're selling)
- **δ** = Distance function (gap measure)
- **ε** = Tolerance threshold (acceptable misalignment)
- **W(x)** = Writability (1 if writable, 0 if not)

### The Gate Condition

```
W(x) = {
    1  if |Φ(x) - Ψ(x)| < ε
    0  otherwise
}
```

A lead is qualified (writable) when the intent-offer gap is within tolerance.

### Connection to Hat Eligibility

From ICHTB geometry, writability equals hat eligibility:

```
W(x) ≡ ĥₙ

Where:
ĥₙ = 1 if ∇Φ, ∇²Φ, Ωⁿ are phase-aligned
ĥₙ = 0 otherwise
```

---

## Part 2: The 97% Efficiency Principle

### The Writables Problem

From the Writables Doctrine:

```
Traditional Approach:
├── Process 100% of leads
├── 97% waste (non-writables processed)
└── 3% efficiency

Writability Approach:
├── Pre-filter to writables only
├── 97% efficiency gain
└── Same absolute conversions
```

### The Qualification Imperative

```
Efficiency = Conversions / Processing_effort
           = W(x) · Collapse_rate / Total_effort

Maximize efficiency by minimizing effort on W(x) = 0
```

### CPU Analogy

```
if (is_writable(lead)) {
    execute(lead);  // Expensive operation
} else {
    skip(lead);     // No CPU spent
}
```

Qualification is the `is_writable()` check.

---

## Part 3: BANT as Tensor Gates

### Traditional BANT

```
BANT = Budget ∧ Authority ∧ Need ∧ Timeline
```

### BANT as ICHTB Mapping

| BANT | ICHTB Fan | Operator | Gate Condition |
|------|-----------|----------|----------------|
| **Budget** | Δ₃ (+X) | +∇²Φ | Shell expansion permitted |
| **Authority** | Δ₆ (-Z) | Φ = i₀ | Scalar anchor exists |
| **Need** | Δ₁ (+Y) | ∇Φ | Gradient non-zero |
| **Timeline** | Δ₅ (+Z) | ∂Φ/∂t | Temporal emergence aligned |

### The Tensor BANT Gate

```
W_BANT(x) = H(+∇²Φ > 0) · H(Φ = i₀) · H(∇Φ ≠ 0) · H(∂Φ/∂t → 0)
```

Where H is the Heaviside step function.

---

## Part 4: Extended Qualification Frameworks

### MEDDIC as Six-Gate Protocol

```
MEDDIC = Metrics × Economic_Buyer × Decision_Criteria × Decision_Process × Identify_Pain × Champion
```

| MEDDIC | ICHTB Fan | Tensor Condition |
|--------|-----------|------------------|
| Metrics | Δ₄ (-X) | -∇²Φ (measurable compression lock) |
| Economic Buyer | Δ₆ (-Z) | Φ = i₀ (decision anchor) |
| Decision Criteria | Δ₃ (+X) | +∇²Φ (expansion parameters) |
| Decision Process | Δ₅ (+Z) | ∂Φ/∂t (temporal sequence) |
| Identify Pain | Δ₁ (+Y) | ∇Φ (gradient intensity) |
| Champion | Δ₂ (-Y) | ∇×F (internal memory loop) |

### The Complete Writability Gate

```
W_MEDDIC(x) = ∏ᵢ₌₁⁶ Gate(Δᵢ)
```

All six gates must pass for full qualification.

---

## Part 5: The Qualification Probability Function

### From Binary to Gaussian

Hard qualification (traditional):
```
W(x) = {
    1  if all criteria pass
    0  otherwise
}
```

Soft qualification (modern):
```
W(x) = exp(-(ΔΨ)² / 2σ²)
```

Where ΔΨ = composite gap across all criteria.

### The Qualification Score

```
Score(x) = Σᵢ wᵢ · Criterion_i(x)

W(x) = σ(Score(x) - θ) = 1 / (1 + e^{-(Score - θ)})
```

The sigmoid function maps score to writability probability.

### Score to Collapse Probability

```
P_collapse(x) ∝ W(x) · Opportunity_factors

Where:
P_collapse ≈ exp(-(ΔΨ)²/2σ²) · γᵗ
```

---

## Part 6: The MQL/SQL Threshold System

### Threshold Definitions

```
MQL = W(x) > θ_MQL (Marketing Qualified Lead)
SQL = W(x) > θ_SQL (Sales Qualified Lead)

Where:
θ_MQL < θ_SQL
```

### The Threshold Ladder

| Stage | Threshold | Writability | Action |
|-------|-----------|-------------|--------|
| Raw Lead | θ = 0 | 0.0-0.3 | Capture only |
| MQL | θ_MQL = 0.3 | 0.3-0.5 | Nurture |
| SQL | θ_SQL = 0.5 | 0.5-0.7 | Sales engage |
| Opportunity | θ_opp = 0.7 | 0.7-0.9 | Active deal |
| Commit | θ_commit = 0.9 | 0.9-1.0 | Close imminent |

### Threshold Crossing Detection

```
Event(MQL) = H(W(x,t) - θ_MQL) - H(W(x,t-1) - θ_MQL)

If Event = 1: Just crossed MQL threshold → trigger handoff
If Event = 0: No crossing
If Event = -1: Dropped below MQL → nurture recapture
```

---

## Part 7: Qualification Criteria Mathematics

### Fit Scoring

```
Fit_score = Σᵢ wᵢ · Fit_criterion_i

Criteria:
- Industry match: w₁ = 0.20
- Company size: w₂ = 0.15
- Job title: w₃ = 0.25
- Geography: w₄ = 0.10
- Technology fit: w₅ = 0.15
- Budget range: w₆ = 0.15
```

### Engagement Scoring

```
Engagement_score = Σⱼ vⱼ · Action_j · Recency_decay(t_j)

Actions:
- Email open: v = 2
- Email click: v = 5
- Content download: v = 10
- Pricing page: v = 20
- Demo request: v = 30
- Webinar: v = 15
```

### Composite Qualification

```
W(x) = α · Fit_score + (1-α) · Engagement_score

Where α typically 0.4-0.6 (balance fit vs. engagement)
```

---

## Part 8: The Intent-State Gap (ΔΨ)

### Formal Definition

```
ΔΨ = ‖Φ - Ψ‖

Where:
Φ = Lead intent vector
Ψ = Ideal customer profile vector
‖·‖ = Norm in intent space
```

### ΔΨ Components

| Component | Dimension | Measurement |
|-----------|-----------|-------------|
| Need gap | Δ₁ | Problem severity mismatch |
| Timing gap | Δ₅ | Purchase timeline mismatch |
| Budget gap | Δ₃ | Financial capacity mismatch |
| Authority gap | Δ₆ | Decision power mismatch |
| Solution gap | Δ₄ | Product-need alignment |
| Champion gap | Δ₂ | Internal advocate existence |

### ΔΨ as Qualification Inverse

```
Low ΔΨ → High W(x) → Qualified
High ΔΨ → Low W(x) → Unqualified

Threshold: ΔΨ < ε → W(x) = 1
```

---

## Part 9: Negative Qualification

### Disqualification Signals

Not all leads should progress. Negative qualification identifies:

```
W(x) = 0 if:
├── Budget: insufficient (∇²Φ < 0 with no expansion)
├── Authority: none accessible (Φ ≠ i₀ for any contact)
├── Need: non-existent (∇Φ = 0)
├── Timeline: never (∂Φ/∂t → ∞)
├── Fit: competitor, student, job seeker
└── Behavior: spam, fraud signals
```

### The Suppression List

```
Suppression = {x : W(x) = 0 AND Permanent(x)}

Permanent conditions:
- Industry exclusion
- Competitor domain
- Previously churned + hostile
- Legal/compliance block
```

### Active Disqualification

```
Disqualify(x) = W(x) → 0

Triggers:
- Explicit "not interested"
- Budget confirmed unavailable
- Timeline > 18 months
- Wrong decision maker with no path
```

---

## Part 10: Predictive Qualification

### Machine Learning Approach

```
W(x) = f_θ(features(x))

Where:
f_θ = Trained classifier (logistic, random forest, gradient boosting)
features = [fit_features, engagement_features, temporal_features]
```

### Feature Engineering

```python
def compute_features(lead: Lead) -> FeatureVector:
    """
    Extract qualification features for ML model.
    """
    return FeatureVector(
        # Fit features
        industry_match=encode_industry(lead.industry),
        size_bucket=encode_size(lead.company_size),
        title_seniority=encode_title(lead.title),
        geo_tier=encode_geo(lead.location),

        # Engagement features
        total_score=lead.engagement_score,
        velocity=lead.score_velocity,
        recency=days_since_last_action(lead),
        depth=len(lead.actions),
        pricing_views=count_pricing_views(lead),
        demo_requested=lead.demo_requested,

        # Temporal features
        time_in_funnel=days_since_capture(lead),
        seasonality=encode_seasonality(now()),
        velocity_trend=compute_trend(lead.score_history)
    )
```

### Model Training

```python
def train_qualification_model(
    historical_leads: List[Lead],
    outcomes: List[bool]
) -> QualificationModel:
    """
    Train W(x) predictor from historical conversions.
    """
    X = [compute_features(lead) for lead in historical_leads]
    y = outcomes  # 1 = converted, 0 = did not

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )

    model.fit(X, y)

    return QualificationModel(
        model=model,
        threshold=calibrate_threshold(model, X, y)
    )
```

---

## Part 11: Implementation Architecture

### The Writability Engine

```python
class WritabilityEngine:
    """
    Qualification gate: W(x) = δ(Φ-Ψ) > ε
    Filters leads into writable and non-writable.
    """

    def __init__(self, config: Config):
        self.fit_weights = config.fit_weights
        self.engagement_weights = config.engagement_weights
        self.alpha = config.fit_engagement_balance
        self.thresholds = config.thresholds
        self.ml_model = load_model(config.model_path)

    def compute_writability(self, lead: Lead) -> WritabilityResult:
        """
        W(x) = α·Fit + (1-α)·Engagement

        Returns probability and classification.
        """
        # Compute fit score
        fit_score = self.compute_fit(lead)

        # Compute engagement score
        engagement_score = self.compute_engagement(lead)

        # Composite score
        composite = self.alpha * fit_score + (1 - self.alpha) * engagement_score

        # ML prediction (if available)
        if self.ml_model:
            ml_prob = self.ml_model.predict_proba(lead)
            composite = 0.5 * composite + 0.5 * ml_prob

        # Classification
        stage = self.classify_stage(composite)

        return WritabilityResult(
            writability=composite,
            is_writable=composite > self.thresholds['mql'],
            stage=stage,
            fit_score=fit_score,
            engagement_score=engagement_score,
            gaps=self.compute_gaps(lead)
        )

    def compute_fit(self, lead: Lead) -> float:
        """
        Fit score from demographic/firmographic criteria.
        """
        scores = {
            'industry': self.score_industry(lead.industry),
            'size': self.score_size(lead.company_size),
            'title': self.score_title(lead.title),
            'geo': self.score_geo(lead.location),
            'tech': self.score_tech(lead.technologies),
            'budget': self.score_budget(lead.budget_indicators)
        }

        return sum(
            scores[k] * self.fit_weights[k]
            for k in scores
        )

    def compute_engagement(self, lead: Lead) -> float:
        """
        Engagement score from behavioral signals.
        """
        score = 0.0
        for action in lead.actions:
            weight = self.engagement_weights.get(action.type, 1)
            recency = self.recency_decay(action.timestamp)
            score += weight * recency

        return min(score / 100, 1.0)  # Normalize

    def compute_gaps(self, lead: Lead) -> Dict[str, float]:
        """
        ΔΨ components: where is the lead deficient?
        """
        return {
            'need_gap': 1.0 - self.score_need(lead),
            'timing_gap': 1.0 - self.score_timing(lead),
            'budget_gap': 1.0 - self.score_budget(lead),
            'authority_gap': 1.0 - self.score_authority(lead),
            'champion_gap': 1.0 - self.score_champion(lead)
        }

    def check_disqualification(self, lead: Lead) -> Optional[str]:
        """
        Negative qualification: reasons to set W(x) = 0.
        """
        if lead.is_competitor:
            return 'competitor'
        if lead.budget_confirmed_zero:
            return 'no_budget'
        if lead.timeline_never:
            return 'no_timeline'
        if lead.unsubscribed:
            return 'unsubscribed'
        if lead.on_suppression_list:
            return 'suppressed'
        return None
```

### Threshold Crossing Handler

```python
class ThresholdCrossingHandler:
    """
    Detects and handles MQL/SQL threshold crossings.
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def check_crossing(
        self,
        lead: Lead,
        old_score: float,
        new_score: float
    ) -> List[ThresholdEvent]:
        """
        Detect threshold crossings.
        """
        events = []

        for stage, threshold in self.thresholds.items():
            # Upward crossing
            if old_score < threshold <= new_score:
                events.append(ThresholdEvent(
                    lead=lead,
                    stage=stage,
                    direction='up',
                    action=self.get_up_action(stage)
                ))
            # Downward crossing
            elif old_score >= threshold > new_score:
                events.append(ThresholdEvent(
                    lead=lead,
                    stage=stage,
                    direction='down',
                    action=self.get_down_action(stage)
                ))

        return events

    def get_up_action(self, stage: str) -> Action:
        """
        Action on upward threshold crossing.
        """
        actions = {
            'mql': Action('notify_marketing', 'New MQL'),
            'sql': Action('notify_sales', 'Sales ready'),
            'opportunity': Action('create_opportunity', 'Active deal'),
            'commit': Action('alert_close', 'Ready to close')
        }
        return actions.get(stage, Action('log', stage))

    def get_down_action(self, stage: str) -> Action:
        """
        Action on downward threshold crossing.
        """
        return Action('recycle_nurture', f'Dropped below {stage}')
```

---

## Part 12: Validation Metrics

### Qualification Accuracy Metrics

| Metric | Equation | Target |
|--------|----------|--------|
| MQL → SQL rate | SQLs / MQLs | 30-50% |
| SQL → Opp rate | Opps / SQLs | 50-70% |
| False positive rate | Unqualified marked qualified | < 20% |
| False negative rate | Qualified marked unqualified | < 10% |
| Score-conversion correlation | r(Score, Conversion) | > 0.5 |

### Drift Detection

```
D_qualification = Actual_SQL_rate / Expected_SQL_rate

If D < 0.7: Qualification criteria too loose
If D > 1.3: Qualification criteria too tight
```

### Calibration Check

```
For each score bucket [0.3-0.4], [0.4-0.5], ..., [0.9-1.0]:
    Expected_rate ≈ Bucket_midpoint
    Actual_rate = Conversions / Leads_in_bucket

If |Actual - Expected| > 0.1: Model miscalibrated
```

---

## Part 13: Connection to Master Equation

### Writability in f(x)

```
f(x) = W(Φ,Ψ,ε) · γᵗ · ∫₀ᵗ A(u,m,τ) dτ
```

**W is the gating multiplier.** If W = 0, the entire equation zeroes out regardless of other factors.

### Qualification as Permission

```
Collapse_value = W(x) · Potential_value

If W(x) = 0: No value extracted (non-writable)
If W(x) = 1: Full potential realized
If W(x) ∈ (0,1): Partial value (Gaussian interpretation)
```

### The 97% Principle in f(x)

```
Total_value = Σₓ f(x) = Σₓ W(x) · γᵗ · ∫A dτ

For W(x) = 0 on 97% of x:
    Total_value concentrated in 3% of leads
    Resources should match this distribution
```

---

## Part 14: Summary

### The Qualification Principle

> **Qualification determines writability—the permission condition that gates which leads can collapse into customers.**

### The Core Equations

1. **Writability Gate:** W(x) = δ(Φ-Ψ) > ε
2. **Binary:** W(x) = {1 if pass, 0 if fail}
3. **Gaussian:** W(x) = exp(-(ΔΨ)²/2σ²)
4. **Composite:** W(x) = α·Fit + (1-α)·Engagement
5. **Threshold:** MQL = W(x) > θ_MQL

### The Action

**Gate first, process only writables.**

---

## References

- Knight, A. & Khan, A. (2025). The Funnel Function.
- Intent Tensor Theory. https://intent-tensor-theory.com/code-equations/
- Mayer, R.C. et al. (1995). An Integrative Model of Organizational Trust. AMR.
- Rackham, N. (1988). SPIN Selling. McGraw-Hill.

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Created by Armstrong Knight & Abdullah Khan

# 0.2.a.ii f(Lead_Generation)

**Capture Mechanics as Gradient Lock**

The mathematics of converting attention into addressable contacts.

---

## Abstract

Lead generation is the first **gradient lock** in the collapse stack. Awareness (Δ₁) has captured attention; now we must lock that gradient into an addressable entity. This transforms anonymous attention into a writable record—the first condition for recursive collapse.

---

## Part 1: The Capture Equation

### Core Definition

```
f(Lead) = Capture(∇Φ) → Addressable_Entity
```

A lead exists when:
1. Attention gradient ∇Φ ≠ 0 (interest exists)
2. Contact information captured (addressable)
3. Permission granted (writable)

### The Capture Gate

```
Lead_captured = H(∇Φ > θ_attention) · H(info_provided) · H(permission_granted)
```

Where H is the Heaviside step function. All three conditions must pass.

---

## Part 2: Tensor Mapping

### Lead Generation in the Collapse Stack

```
Φ (latent desire)
    ↓
∇Φ (awareness gradient) ← AWARENESS CAPTURED
    ↓
∇Φ_locked (lead captured) ← LEAD GENERATION
    ↓
∇×F (memory loop) ← NURTURING
    ↓
∇²Φ (collapse) ← CONVERSION
```

Lead generation **locks the gradient** into an addressable record.

### ICHTB Mapping

| Gate | Operator | Lead Gen Role |
|------|----------|---------------|
| Δ₁ | ∇Φ | Gradient exists (attention) |
| Δ₆ | Φ = i₀ | Intent anchor exists (need) |
| Δ₃ | +∇²Φ | Expansion permitted (form presented) |
| Δ₄ | -∇²Φ | Compression lock (info captured) |

---

## Part 3: Form Friction Mathematics

### The Friction Equation

```
P(form_complete) = exp(-β · n_fields)
```

Where:
- **n_fields** = Number of form fields
- **β** = Friction coefficient (~0.1-0.3 per field)

### Optimal Form Length

| Fields | P(complete) @ β=0.15 | Use Case |
|--------|---------------------|----------|
| 1 | 0.86 | Email-only newsletter |
| 3 | 0.64 | Basic lead magnet |
| 5 | 0.47 | Gated content |
| 8 | 0.30 | Demo request |
| 12 | 0.16 | Full qualification |

### The Trade-off

```
Lead_Value = Quality(n) · Quantity(n)

Where:
Quality(n) ∝ n  (more fields = more info)
Quantity(n) ∝ exp(-βn)  (more fields = fewer completions)
```

**Optimal n:** Take derivative, set to zero:

```
d(Lead_Value)/dn = 0
→ n* = 1/β
```

For β = 0.15: **n* ≈ 6-7 fields**

---

## Part 4: Lead Scoring as Collapse Probability

### The Scoring Function

```
Lead_Score(x) = P_collapse(x) = exp(-(ΔΨ)²/2σ²)
```

Map lead attributes to intent-state gap ΔΨ:

| Attribute | ΔΨ Contribution | Weight |
|-----------|-----------------|--------|
| Job title match | -0.3 (closer) | High |
| Company size fit | -0.2 (closer) | Medium |
| Engagement level | -0.1 to -0.3 | High |
| Timing signals | -0.2 (closer) | Medium |
| Budget indicators | -0.3 (closer) | High |

### Scoring Implementation

```python
def compute_lead_score(lead: Lead, ideal_customer: ICP) -> float:
    """
    Lead score = P_collapse = exp(-(ΔΨ)²/2σ²)
    """
    # Compute attribute gaps
    gaps = {
        'title': title_gap(lead.title, ideal_customer.titles),
        'size': size_gap(lead.company_size, ideal_customer.size_range),
        'engagement': engagement_gap(lead.actions, ideal_customer.engagement),
        'timing': timing_gap(lead.signals, ideal_customer.timing),
        'budget': budget_gap(lead.indicators, ideal_customer.budget)
    }

    # Weighted sum
    weights = {'title': 0.25, 'size': 0.15, 'engagement': 0.25,
               'timing': 0.15, 'budget': 0.20}

    delta_psi = sum(gaps[k] * weights[k] for k in gaps)

    # Collapse probability
    sigma = 0.5  # Tolerance
    score = math.exp(-(delta_psi ** 2) / (2 * sigma ** 2))

    return score
```

---

## Part 5: Lead Magnets as Gradient Amplifiers

### The Value Exchange

```
Lead_Magnet_Value = ∂(∇Φ)/∂(submission)
```

The magnet must increase the gradient enough to overcome form friction:

```
∇Φ_after - ∇Φ_before > Friction_cost
```

### Magnet Types by Gradient Increase

| Magnet Type | Typical ∇Φ Boost | Best For |
|-------------|------------------|----------|
| Checklist | +0.1-0.2 | Low-commitment entry |
| Ebook | +0.2-0.3 | Education-stage leads |
| Template | +0.3-0.4 | Action-oriented leads |
| Calculator | +0.4-0.5 | High-intent evaluation |
| Demo | +0.5-0.7 | Decision-stage leads |

### Magnet-Form Optimization

```
Optimal_magnet = argmax{∇Φ_boost - β·n_fields}
```

High-value magnets justify more fields.

---

## Part 6: Channel Mathematics

### Channel Efficiency

```
f(Channel_efficiency) = Leads_captured / (Cost · Time)
```

### Channel-Gradient Mapping

| Channel | ∇Φ at Entry | Cost/Lead | Speed |
|---------|-------------|-----------|-------|
| Organic search | High (intent) | Low | Slow |
| Paid search | High (intent) | Medium | Fast |
| Social organic | Low-Medium | Low | Medium |
| Social paid | Low-Medium | Medium | Fast |
| Content syndication | Medium | High | Fast |
| Events | High | High | Slow |
| Referral | Very High | Low | Variable |

### The Gradient-Cost Frontier

```
Efficiency = ∇Φ_captured / Cost_per_lead
```

Optimize along this frontier based on:
- **Speed needed:** Paid > Organic
- **Budget:** Organic > Paid
- **Quality:** Referral > All others

---

## Part 7: Permission and Compliance

### The Writability Constraint

```
W(lead) = δ(Φ_lead - Ψ_regulations) > ε_compliance
```

Lead must satisfy:
- Explicit consent (GDPR, CCPA)
- Purpose limitation
- Data minimization
- Right to be forgotten

### Compliance as Gate

```
Lead_valid = Lead_captured · Compliance_gate

Where:
Compliance_gate = {
    1  if all regulations satisfied
    0  otherwise
}
```

Non-compliant leads have W(x) = 0 regardless of score.

---

## Part 8: From Lead to MQL

### The MQL Threshold

```
MQL = Lead where Score(lead) > θ_MQL
```

Typical thresholds:
- **Conservative:** θ_MQL = 0.7 (higher quality, fewer leads)
- **Moderate:** θ_MQL = 0.5 (balanced)
- **Aggressive:** θ_MQL = 0.3 (more volume, lower quality)

### MQL as Writability Gate

```
MQL ≡ W(lead) = δ(Score - θ_MQL) > 0
```

The MQL threshold is the writability gate for sales handoff.

---

## Part 9: Validation Metrics

### Core Metrics

| Metric | Equation | Target |
|--------|----------|--------|
| Capture rate | Leads / Visitors | 2-5% |
| Cost per lead | Spend / Leads | Industry-specific |
| MQL rate | MQLs / Leads | 15-25% |
| Lead velocity | Leads / Time | Growing |

### Drift Detection

```
D_leadgen = Actual_capture_rate / Expected_capture_rate

If D < 0.8: Form friction increased or offer degraded
If D > 1.2: Unqualified traffic or spam
```

---

## Part 10: Connection to Master Equation

### Lead Generation in f(x)

```
f(x) = W(Φ,Ψ,ε) · γᵗ · ∫₀ᵗ A(u,m,τ) dτ
```

Lead generation establishes:
- **Φ** = Lead's intent (captured via form data)
- **W** = Writability (permission and compliance)
- **A** = Baseline activation (initial engagement)

Without leads, the integral has nothing to integrate over.

### The Capture Moment

```
t_capture = argmin{t : Lead_exists(t) = True}
```

Everything before t_capture is pre-lead awareness.
Everything after is the integration period toward collapse.

---

## Summary

### The Lead Generation Principle

> **Lead generation locks the attention gradient into an addressable, writable record.**

### The Core Equations

1. **Capture:** Lead = H(∇Φ > θ) · H(info) · H(permission)
2. **Friction:** P(complete) = exp(-β·n_fields)
3. **Scoring:** Score = exp(-(ΔΨ)²/2σ²)
4. **Threshold:** MQL = Score > θ_MQL

### The Action

**Optimize the capture gate, not just the traffic.**

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Created by Armstrong Knight & Abdullah Khan

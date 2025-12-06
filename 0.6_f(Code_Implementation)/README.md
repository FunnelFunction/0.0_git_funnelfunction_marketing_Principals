# 0.6 f(Code_Implementation)

**Production Architecture: From Equations to Execution**

The complete Python implementation of the Funnel Function for production deployment.

---

## Abstract

Theory must become code. This layer provides the **production-ready implementation** of the Funnel Function, including the core collapse engine, pipeline modules for each funnel stage, and integration patterns for marketing technology stacks. Every equation from the framework translates to executable Python.

---

## Part 1: Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                  FUNNEL FUNCTION ENGINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   CAPTURE   │ →  │   PROCESS   │ →  │   COLLAPSE  │      │
│  │   (∇Φ)      │    │   (∇×F, W)  │    │   (∇²Φ→ρ_q) │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                 │                   │              │
│         ▼                 ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              DATA LAYER (Events, State)              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           INTEGRATION LAYER (CRM, MA, Analytics)     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
funnel_function/
├── core/
│   ├── __init__.py
│   ├── collapse_engine.py      # Main f(x) computation
│   ├── writability.py          # W(x) gate
│   ├── activation.py           # A(u,m,τ) integral
│   └── drift.py                # D = ∇Ψ/∇Φ monitoring
├── stages/
│   ├── __init__.py
│   ├── awareness.py            # ∇Φ creation
│   ├── lead_gen.py             # ∇Φ lock
│   ├── nurturing.py            # ∇×F installation
│   ├── qualification.py        # W(x) evaluation
│   ├── conversion.py           # ∇²Φ lock
│   └── close.py                # ρ_q formation
├── measurement/
│   ├── __init__.py
│   ├── attention.py            # Adelaide AU
│   ├── attribution.py          # Shapley values
│   └── metrics.py              # KPI computation
├── integrations/
│   ├── __init__.py
│   ├── crm.py                  # Salesforce, HubSpot
│   ├── analytics.py            # GA4, Mixpanel
│   └── ma.py                   # Marketing automation
└── config/
    ├── __init__.py
    └── settings.py             # Configuration management
```

---

## Part 2: Core Engine

### The Collapse Engine

```python
"""
core/collapse_engine.py
Main implementation of f(x) = W(Φ,Ψ,ε) · γᵗ · ∫A dτ
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

@dataclass
class CollapseResult:
    """Result of f(x) computation."""
    value: float
    writability: float
    time_discount: float
    activation_integral: float
    collapse_ready: bool
    stage: str
    recommendations: List[str]


class CollapseEngine:
    """
    The Funnel Function engine: f(x) = W(Φ,Ψ,ε) · γᵗ · ∫A dτ

    Computes collapse probability and value for any lead/opportunity.
    """

    def __init__(self, config: 'Config'):
        self.gamma = config.time_discount  # ~0.95-0.99 monthly
        self.epsilon = config.writability_threshold
        self.close_threshold = config.close_threshold
        self.sigma = config.collapse_sigma

    def compute(
        self,
        lead: 'Lead',
        offer: 'Offer',
        interactions: List['Interaction']
    ) -> CollapseResult:
        """
        Compute f(x) for a lead-offer pair.

        f(x) = W(Φ,Ψ,ε) · γᵗ · ∫₀ᵗ A(u,m,τ) dτ
        """
        # Compute writability gate
        W = self.compute_writability(lead, offer)

        # Compute time discount
        t = self.compute_time_in_funnel(lead)
        gamma_t = self.gamma ** t

        # Compute activation integral
        A_integral = self.compute_activation_integral(interactions)

        # Compute f(x)
        f_x = W * gamma_t * A_integral

        # Determine stage and readiness
        stage = self.determine_stage(lead, W, A_integral)
        collapse_ready = self.check_collapse_ready(W, A_integral)

        # Generate recommendations
        recommendations = self.generate_recommendations(
            lead, W, gamma_t, A_integral, stage
        )

        return CollapseResult(
            value=f_x,
            writability=W,
            time_discount=gamma_t,
            activation_integral=A_integral,
            collapse_ready=collapse_ready,
            stage=stage,
            recommendations=recommendations
        )

    def compute_writability(self, lead: 'Lead', offer: 'Offer') -> float:
        """
        W(x) = δ(Φ-Ψ) > ε

        Gaussian interpretation:
        W(x) = exp(-(ΔΨ)²/2σ²)
        """
        # Compute intent vector
        phi = self.compute_intent_vector(lead)

        # Compute offer vector
        psi = self.compute_offer_vector(offer)

        # Compute gap
        delta_psi = np.linalg.norm(phi - psi)

        # Gaussian writability
        W = math.exp(-(delta_psi ** 2) / (2 * self.sigma ** 2))

        return W

    def compute_intent_vector(self, lead: 'Lead') -> np.ndarray:
        """
        Φ(lead) = [need, timing, budget, authority, fit, engagement]
        """
        return np.array([
            lead.need_score,
            lead.timing_score,
            lead.budget_score,
            lead.authority_score,
            lead.fit_score,
            lead.engagement_score
        ])

    def compute_offer_vector(self, offer: 'Offer') -> np.ndarray:
        """
        Ψ(offer) = [problem_solved, timeline, price_point, decision_level, icp, engagement_required]
        """
        return np.array([
            offer.problem_fit,
            offer.timeline_fit,
            offer.price_normalized,
            offer.decision_level,
            offer.icp_match,
            offer.engagement_threshold
        ])

    def compute_time_in_funnel(self, lead: 'Lead') -> float:
        """
        t = time since first touch, in discount periods (months).
        """
        days_in_funnel = (datetime.now() - lead.first_touch).days
        months = days_in_funnel / 30.0
        return months

    def compute_activation_integral(
        self,
        interactions: List['Interaction']
    ) -> float:
        """
        ∫₀ᵗ A(u,m,τ) dτ ≈ Σᵢ A(interaction_i) × duration_i

        A = (B × M × S) / Σ
        """
        integral = 0.0

        for interaction in interactions:
            # Body: sensory signal
            B = self.compute_body(interaction)

            # Mind: relevance
            M = self.compute_mind(interaction)

            # Soul: resonance
            S = self.compute_soul(interaction)

            # Suppression
            sigma = self.compute_suppression(interaction)

            # Activation for this interaction
            A = (B * M * S) / max(sigma, 0.01)

            # Add to integral (duration-weighted)
            integral += A * interaction.duration_minutes

        return integral

    def compute_body(self, interaction: 'Interaction') -> float:
        """B = Sensory signal strength."""
        weights = {
            'email_open': 0.3,
            'email_click': 0.5,
            'page_view': 0.4,
            'video_watch': 0.7,
            'demo': 0.9,
            'call': 0.95,
            'meeting': 1.0
        }
        return weights.get(interaction.type, 0.3)

    def compute_mind(self, interaction: 'Interaction') -> float:
        """M = Relevance to lead's needs."""
        return interaction.relevance_score

    def compute_soul(self, interaction: 'Interaction') -> float:
        """S = Brand/value resonance."""
        return interaction.resonance_score

    def compute_suppression(self, interaction: 'Interaction') -> float:
        """Σ = Environmental suppression."""
        base = 1.0
        if interaction.competitor_present:
            base += 0.3
        if interaction.distraction_high:
            base += 0.2
        if interaction.fatigue_signals:
            base += 0.4
        return base

    def determine_stage(
        self,
        lead: 'Lead',
        W: float,
        A_integral: float
    ) -> str:
        """Determine current funnel stage."""
        if W < 0.3:
            return 'awareness'
        elif W < 0.5:
            return 'interest'
        elif A_integral < self.close_threshold * 0.3:
            return 'consideration'
        elif A_integral < self.close_threshold * 0.7:
            return 'intent'
        elif A_integral < self.close_threshold:
            return 'evaluation'
        else:
            return 'purchase'

    def check_collapse_ready(self, W: float, A_integral: float) -> bool:
        """Check if collapse conditions met."""
        return W > 0.7 and A_integral >= self.close_threshold

    def generate_recommendations(
        self,
        lead: 'Lead',
        W: float,
        gamma_t: float,
        A_integral: float,
        stage: str
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if W < 0.5:
            recommendations.append('Improve qualification - writability below threshold')

        if gamma_t < 0.8:
            recommendations.append('Accelerate - significant time decay occurring')

        if A_integral < self.close_threshold * 0.5:
            recommendations.append('Increase engagement - activation integral low')

        if stage == 'evaluation' and W > 0.8:
            recommendations.append('Ready for close attempt')

        return recommendations


class CollapseProbability:
    """
    P_collapse = exp(-(ΔΨ)²/2σ²)

    Probability computation for collapse events.
    """

    def __init__(self, sigma: float = 0.3):
        self.sigma = sigma

    def compute(self, delta_psi: float) -> float:
        """Gaussian collapse probability."""
        return math.exp(-(delta_psi ** 2) / (2 * self.sigma ** 2))

    def compute_from_gap(
        self,
        intent: np.ndarray,
        offer: np.ndarray
    ) -> float:
        """Compute from intent-offer vectors."""
        delta_psi = np.linalg.norm(intent - offer)
        return self.compute(delta_psi)
```

---

## Part 3: Stage Modules

### Awareness Module

```python
"""
stages/awareness.py
∇Φ creation: Gradient formation in the intent field.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class AwarenessResult:
    """Result of awareness stage processing."""
    gradient_magnitude: float
    gradient_direction: np.ndarray
    attention_score: float
    reach: int
    frequency: float


class AwarenessEngine:
    """
    Awareness = ∇Φ creation

    Manages gradient formation through media exposure.
    """

    def __init__(self, config: 'Config'):
        self.attention_decay = config.attention_decay_rates
        self.reach_weights = config.reach_weights

    def compute_gradient(
        self,
        impressions: List['Impression'],
        target_audience: 'Audience'
    ) -> AwarenessResult:
        """
        Compute ∇Φ from media impressions.

        ∇Φ = Σᵢ wᵢ × Attention(i) × Direction(i)
        """
        gradient = np.zeros(6)  # 6-dimensional intent space
        total_attention = 0.0

        for impression in impressions:
            # Compute attention for this impression
            attention = self.compute_attention(impression)
            total_attention += attention

            # Compute direction toward purchase intent
            direction = self.compute_direction(impression, target_audience)

            # Add weighted contribution to gradient
            weight = self.reach_weights.get(impression.channel, 1.0)
            gradient += weight * attention * direction

        magnitude = np.linalg.norm(gradient)
        direction = gradient / max(magnitude, 0.001)

        return AwarenessResult(
            gradient_magnitude=magnitude,
            gradient_direction=direction,
            attention_score=total_attention / max(len(impressions), 1),
            reach=len(set(i.user_id for i in impressions)),
            frequency=len(impressions) / max(len(set(i.user_id for i in impressions)), 1)
        )

    def compute_attention(self, impression: 'Impression') -> float:
        """
        Attention(t) = A₀ × exp(-λt)

        Apply attention decay based on platform.
        """
        lambda_decay = self.attention_decay.get(impression.channel, 0.1)
        A_0 = impression.initial_attention
        t = impression.view_duration_seconds

        return A_0 * math.exp(-lambda_decay * t)

    def compute_direction(
        self,
        impression: 'Impression',
        target: 'Audience'
    ) -> np.ndarray:
        """Direction vector toward purchase intent."""
        # Direction based on message type and audience fit
        base_direction = np.array([
            impression.need_signal,
            impression.timing_signal,
            impression.budget_signal,
            impression.authority_signal,
            impression.fit_signal,
            impression.engagement_signal
        ])

        # Weight by audience match
        audience_match = self.compute_audience_match(impression, target)

        return base_direction * audience_match
```

### Nurturing Module

```python
"""
stages/nurturing.py
∇×F installation: Memory loop through drip sequences.
"""

import math
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta


@dataclass
class NurtureResult:
    """Result of nurture computation."""
    curl_strength: float
    engagement_level: float
    decay_risk: float
    next_touch_recommended: datetime
    content_recommendation: str


class NurtureEngine:
    """
    Nurturing = ∇×F installation

    Maintains memory loop through scheduled and triggered touches.
    """

    def __init__(self, config: 'Config'):
        self.decay_constants = config.decay_by_channel
        self.fatigue_threshold = config.fatigue_threshold

    def compute_curl(self, lead: 'Lead') -> NurtureResult:
        """
        ∇×F = Circulation in engagement field

        ∮ F⃗ · dℓ⃗ ≠ 0 indicates active memory loop.
        """
        touchpoints = lead.touchpoints_last_30_days

        if len(touchpoints) < 2:
            return NurtureResult(
                curl_strength=0.0,
                engagement_level=lead.engagement_score,
                decay_risk=1.0,
                next_touch_recommended=datetime.now(),
                content_recommendation='re-engagement'
            )

        # Compute circulation
        circulation = sum(tp.engagement_delta for tp in touchpoints)

        # Compute phase coherence
        phase_variance = self.compute_phase_variance(touchpoints)

        # Curl = circulation / (1 + variance)
        curl = circulation / (1 + phase_variance)

        # Compute current engagement with decay
        engagement = self.compute_engagement(lead)

        # Compute decay risk
        decay_risk = self.compute_decay_risk(lead, engagement)

        # Recommend next touch
        next_touch = self.recommend_next_touch(lead, decay_risk)

        # Recommend content
        content = self.recommend_content(lead)

        return NurtureResult(
            curl_strength=curl,
            engagement_level=engagement,
            decay_risk=decay_risk,
            next_touch_recommended=next_touch,
            content_recommendation=content
        )

    def compute_engagement(self, lead: 'Lead') -> float:
        """
        E(t) = E₀ × exp(-λt) + Σ ΔE_touches
        """
        base_engagement = lead.initial_engagement
        lambda_decay = self.decay_constants.get(lead.primary_channel, 0.05)
        t = (datetime.now() - lead.last_touch).days

        decayed = base_engagement * math.exp(-lambda_decay * t)

        # Add touchpoint boosts
        boosts = sum(
            tp.boost * math.exp(-lambda_decay * (datetime.now() - tp.timestamp).days)
            for tp in lead.touchpoints
        )

        return min(decayed + boosts, 1.0)

    def compute_decay_risk(self, lead: 'Lead', engagement: float) -> float:
        """Risk that engagement will decay below threshold."""
        if engagement > 0.7:
            return 0.1
        elif engagement > 0.5:
            return 0.3
        elif engagement > 0.3:
            return 0.6
        else:
            return 0.9

    def recommend_next_touch(
        self,
        lead: 'Lead',
        decay_risk: float
    ) -> datetime:
        """
        Δt* = (1/λ) × ln(E_threshold / E₀)
        """
        lambda_decay = self.decay_constants.get(lead.primary_channel, 0.05)

        if decay_risk > 0.7:
            days = 1
        elif decay_risk > 0.5:
            days = 3
        elif decay_risk > 0.3:
            days = 7
        else:
            days = 14

        return datetime.now() + timedelta(days=days)

    def recommend_content(self, lead: 'Lead') -> str:
        """Recommend content based on stage and history."""
        stage = lead.stage

        content_map = {
            'awareness': 'educational_blog',
            'interest': 'industry_report',
            'consideration': 'case_study',
            'intent': 'comparison_guide',
            'evaluation': 'demo_offer',
            'purchase': 'pricing_proposal'
        }

        return content_map.get(stage, 'general_nurture')
```

---

## Part 4: Drift Monitoring

```python
"""
core/drift.py
D = ∇Ψ/∇Φ monitoring for model health.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats


@dataclass
class DriftReport:
    """Report on model drift status."""
    drift_quotient: float
    layer_drifts: Dict[str, float]
    trend: str
    alerts: List[str]
    healthy: bool


class DriftMonitor:
    """
    D = ∇Ψ/∇Φ

    Monitors divergence between predicted and actual outcomes.
    """

    def __init__(self, config: 'Config'):
        self.thresholds = config.drift_thresholds
        self.history_window = config.drift_history_window

    def compute_drift(
        self,
        predictions: Dict[str, float],
        actuals: Dict[str, float]
    ) -> DriftReport:
        """
        Compute drift quotient across all layers.

        D = Actual / Predicted

        D = 1.0 → Model accurate
        D < 0.8 or D > 1.2 → Drift detected
        """
        layer_drifts = {}
        alerts = []

        for layer in predictions:
            pred = predictions[layer]
            actual = actuals.get(layer, 0)

            if pred == 0:
                d = float('inf') if actual > 0 else 1.0
            else:
                d = actual / pred

            layer_drifts[layer] = d

            # Check thresholds
            if d < self.thresholds[layer]['min']:
                alerts.append(f'{layer}: Under-performing (D={d:.2f})')
            elif d > self.thresholds[layer]['max']:
                alerts.append(f'{layer}: Over-performing (D={d:.2f})')

        # Overall drift
        overall_d = np.mean(list(layer_drifts.values()))

        # Trend analysis
        trend = self.analyze_trend(layer_drifts)

        return DriftReport(
            drift_quotient=overall_d,
            layer_drifts=layer_drifts,
            trend=trend,
            alerts=alerts,
            healthy=len(alerts) == 0 and 0.8 < overall_d < 1.2
        )

    def analyze_trend(self, current_drifts: Dict[str, float]) -> str:
        """Determine if drift is improving, stable, or worsening."""
        if not hasattr(self, 'drift_history'):
            self.drift_history = []

        self.drift_history.append(current_drifts)

        if len(self.drift_history) < 3:
            return 'insufficient_data'

        # Look at recent trend
        recent = self.drift_history[-self.history_window:]
        means = [np.mean(list(d.values())) for d in recent]

        if len(means) < 2:
            return 'stable'

        slope, _, _, p_value, _ = stats.linregress(range(len(means)), means)

        if p_value > 0.05:
            return 'stable'
        elif slope > 0.01:
            return 'improving' if means[-1] < 1 else 'worsening'
        elif slope < -0.01:
            return 'worsening' if means[-1] < 1 else 'improving'
        else:
            return 'stable'

    def should_recalibrate(self, drift_report: DriftReport) -> bool:
        """Determine if model needs recalibration."""
        return (
            drift_report.drift_quotient < 0.5 or
            drift_report.drift_quotient > 2.0 or
            len(drift_report.alerts) > 2 or
            drift_report.trend == 'worsening'
        )
```

---

## Part 5: Integration Layer

### CRM Integration

```python
"""
integrations/crm.py
Integration with Salesforce, HubSpot, and other CRMs.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Lead:
    """Unified lead representation."""
    id: str
    email: str
    company: str
    title: str
    stage: str
    score: float
    first_touch: datetime
    last_touch: datetime
    touchpoints: List['Touchpoint']
    properties: Dict[str, any]


class CRMAdapter(ABC):
    """Abstract CRM adapter."""

    @abstractmethod
    def get_leads(self, filters: Dict) -> List[Lead]:
        pass

    @abstractmethod
    def update_lead(self, lead_id: str, updates: Dict) -> bool:
        pass

    @abstractmethod
    def get_interactions(self, lead_id: str) -> List['Interaction']:
        pass


class SalesforceAdapter(CRMAdapter):
    """Salesforce integration."""

    def __init__(self, config: 'SalesforceConfig'):
        self.client = Salesforce(
            username=config.username,
            password=config.password,
            security_token=config.security_token
        )

    def get_leads(self, filters: Dict) -> List[Lead]:
        """Fetch leads from Salesforce."""
        query = self.build_query(filters)
        results = self.client.query(query)

        return [self.map_lead(r) for r in results['records']]

    def update_lead(self, lead_id: str, updates: Dict) -> bool:
        """Update lead in Salesforce."""
        try:
            self.client.Lead.update(lead_id, updates)
            return True
        except Exception as e:
            logger.error(f'Failed to update lead {lead_id}: {e}')
            return False

    def map_lead(self, sf_record: Dict) -> Lead:
        """Map Salesforce record to unified Lead."""
        return Lead(
            id=sf_record['Id'],
            email=sf_record['Email'],
            company=sf_record['Company'],
            title=sf_record.get('Title', ''),
            stage=sf_record['Status'],
            score=sf_record.get('Lead_Score__c', 0),
            first_touch=parse_datetime(sf_record['CreatedDate']),
            last_touch=parse_datetime(sf_record['LastActivityDate']),
            touchpoints=[],
            properties=sf_record
        )


class HubSpotAdapter(CRMAdapter):
    """HubSpot integration."""

    def __init__(self, config: 'HubSpotConfig'):
        self.client = HubSpot(api_key=config.api_key)

    def get_leads(self, filters: Dict) -> List[Lead]:
        """Fetch contacts from HubSpot."""
        contacts = self.client.crm.contacts.get_all(
            properties=['email', 'company', 'jobtitle', 'lifecyclestage']
        )

        return [self.map_contact(c) for c in contacts]

    def map_contact(self, hs_contact: Dict) -> Lead:
        """Map HubSpot contact to unified Lead."""
        props = hs_contact['properties']
        return Lead(
            id=hs_contact['id'],
            email=props.get('email', ''),
            company=props.get('company', ''),
            title=props.get('jobtitle', ''),
            stage=props.get('lifecyclestage', 'subscriber'),
            score=float(props.get('hubspotscore', 0)),
            first_touch=parse_datetime(props.get('createdate')),
            last_touch=parse_datetime(props.get('lastmodifieddate')),
            touchpoints=[],
            properties=props
        )
```

---

## Part 6: Configuration

```python
"""
config/settings.py
Configuration management for Funnel Function.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import yaml


@dataclass
class Config:
    """Main configuration object."""

    # Core engine settings
    time_discount: float = 0.97  # γ monthly
    writability_threshold: float = 0.3
    close_threshold: float = 100.0
    collapse_sigma: float = 0.3

    # Decay rates by channel
    decay_by_channel: Dict[str, float] = field(default_factory=lambda: {
        'email': 0.05,
        'social': 0.10,
        'content': 0.03,
        'demo': 0.02,
        'call': 0.01
    })

    # Attention decay rates
    attention_decay_rates: Dict[str, float] = field(default_factory=lambda: {
        'tv': 0.02,
        'youtube': 0.05,
        'facebook': 0.25,
        'display': 0.50,
        'search': 0.03
    })

    # Drift thresholds
    drift_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'attention': {'min': 0.7, 'max': 1.5},
        'brand': {'min': 0.8, 'max': 1.3},
        'behavior': {'min': 0.8, 'max': 1.2},
        'business': {'min': 0.7, 'max': 1.4}
    })

    # Fatigue settings
    fatigue_threshold: float = 0.3
    drift_history_window: int = 7

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
```

---

## Contents

| Module | Description |
|--------|-------------|
| [0.6.a f(Core_Engine)](./0.6.a_f(Core_Engine)/) | Main f(x) computation |
| [0.6.b f(Pipeline_Modules)](./0.6.b_f(Pipeline_Modules)/) | Stage-specific processors |

---

## Summary

### The Implementation Principle

> **Every equation has a Python implementation. Every implementation has test coverage. Every deployment has drift monitoring.**

### Core Classes

```
CollapseEngine      → f(x) = W·γᵗ·∫A dτ
CollapseProbability → P = exp(-(ΔΨ)²/2σ²)
AwarenessEngine     → ∇Φ creation
NurtureEngine       → ∇×F installation
DriftMonitor        → D = ∇Ψ/∇Φ monitoring
```

### The Action

**Deploy the engine, integrate with stack, monitor for drift.**

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Created by Armstrong Knight & Abdullah Khan

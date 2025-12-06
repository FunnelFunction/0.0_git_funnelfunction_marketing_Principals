# 0.3.d f(Learned_Policy)

**ML Replaces Human Judgment via Curvent Flow**

The mathematical architecture for learned policies mapped to the Curvent execution vector.

---

## Abstract

The "Learned CEO" emerges when policy π_θ determines not just individual decisions but the **intent itself**. This framework maps learned policies to the Curvent vector κ(t)—the composite execution force that guides all collapse decisions. The policy gradient becomes the gradient of the Curvent.

---

## Part 1: The Learned CEO Thesis

### What It Means

Traditional business:
- Humans specify intent
- Systems execute intent
- Humans evaluate outcomes

Learned policy:
- System **learns** intent from outcomes
- System executes learned intent
- System evaluates and updates itself

### The Emergence Condition

The **Learned CEO** emerges when:

```
∃ π_θ : Φ_learned = argmax E[Σ γᵗ R_t]
```

The system discovers what it should want from observing what works.

---

## Part 2: Curvent as Execution Vector

### The Curvent Equation

From ITT Code Equations:

```
κ(t) = ∂Φ/∂x + λ·∇Φ + Σ Γ
```

Where:
- **∂Φ/∂x** = Local gradient (immediate context)
- **λ·∇Φ** = Global weighting (strategic direction)
- **Σ Γ** = External gates (conditions, APIs, constraints)

### Curvent as Policy Output

The learned policy **outputs** the Curvent:

```
κ(t) = π_θ(s_t)

Where:
- π_θ = Learned policy with parameters θ
- s_t = Current state
- κ(t) = Execution vector for time t
```

### The Mapping

| Policy Component | Curvent Component | Meaning |
|-----------------|-------------------|---------|
| Action-value Q(s,a) | ∂Φ/∂x | Local optimal direction |
| Value function V(s) | λ·∇Φ | Global strategic gradient |
| Constraint satisfaction | Σ Γ | External gate compliance |

---

## Part 3: The State Space

### State Representation

```
s_t = [x_demo, x_behavioral, x_contextual, x_engagement, x_market]
```

| Component | Examples | Tensor Layer |
|-----------|----------|--------------|
| x_demo | Age, industry, company size | Φ (static intent) |
| x_behavioral | Click history, page views | ∇Φ (gradient signals) |
| x_contextual | Time of day, device, location | ∇×F (memory context) |
| x_engagement | Email opens, downloads | ∇²Φ (collapse proximity) |
| x_market | Competitor activity, seasonality | ρ_q (external charge) |

### State as Intent Field Sample

```
s_t = Sample(Φ(x,t))
```

Each state observation is a sample from the intent field at that moment.

---

## Part 4: The Action Space

### Action Types

```
A = {a_1, a_2, ..., a_K}
```

| Action Type | Examples | Curvent Effect |
|-------------|----------|----------------|
| Marketing touch | Email, ad, call | ∂Φ/∂x (nudge) |
| Pricing | Discount, premium | λ·∇Φ (reweight) |
| Timing | Immediate, delay, schedule | ∂Φ/∂t (temporal) |
| Channel | Email, phone, in-person | Δᵢ (fan selection) |

### Continuous vs Discrete

**Discrete actions** → Q-learning, DQN, bandits
**Continuous actions** → Policy gradients, SAC

For continuous action spaces:

```
a_t = μ_θ(s_t) + ε,  ε ~ N(0, σ)
```

The policy outputs mean action; noise enables exploration.

---

## Part 5: The Reward Function

### Core Reward Structure

```
R(s, a, s') = Revenue(s') - Cost(a) - λ·Risk(s')
```

| Component | Measurement | Weight |
|-----------|-------------|--------|
| Revenue | Conversion value, CLV | +1.0 |
| Cost | Action cost (send, call, discount) | -1.0 |
| Risk | Churn probability, brand damage | -λ (tunable) |

### Reward as Collapse Signal

```
R(s,a,s') = ∇²Φ(s') - ∇²Φ(s)
```

Reward = Change in collapse curvature due to action.

If action brings customer closer to collapse (∇²Φ increases), reward is positive.

### Customer Lifetime Value as Infinite-Horizon Return

```
CLV = E[Σ_{t=0}^∞ γᵗ R_t | s₀ = NewCustomer]
```

This is the **Bellman value** of a new customer—not assumed, but computed.

---

## Part 6: Policy Training Pipeline

### The Policy Gradient Theorem

```
∇_θ J(θ) = E_{π_θ} [Σ_t ∇_θ ln π_θ(a_t|s_t) · A(s_t, a_t)]
```

Where:
- **J(θ)** = Expected cumulative reward
- **A(s,a)** = Advantage function = Q(s,a) - V(s)

### REINFORCE Update

```
θ ← θ + α · γᵗ · G_t · ∇_θ ln π_θ(a_t|s_t)

Where:
G_t = Σ_{τ=t}^T γ^{τ-t} R_τ (return-to-go)
```

### PPO: Proximal Policy Optimization

The workhorse of modern RL:

```
L^{CLIP}(θ) = E_t [min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

Where:
r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)  (probability ratio)
Â_t = Advantage estimate
ε ≈ 0.2  (clip range)
```

**Stable autonomous learning without human intervention.**

### SAC: Soft Actor-Critic

Adds entropy regularization:

```
J(θ) = E_{τ~π_θ} [Σ_t r(s_t, a_t) + α H(π_θ(·|s_t))]
```

The entropy term **H** encodes "curiosity"—explore when uncertain.

---

## Part 7: Curvent Flow Dynamics

### The Flow Equation

The Curvent evolves over time:

```
dκ/dt = ∇_θ J(θ) · π_θ(s_t)
```

Policy updates drive Curvent evolution.

### Curvent as Information Flow

```
κ(t) = ∫ π_θ(a|s) · ∇log π_θ(a|s) · Q(s,a) da
```

This is the **expected gradient direction** weighted by action values.

### Stability Condition

From ITT:

```
Stability = Φ · (∇×F) · (∇Φ) · (∇²Φ)
```

For learned policy stability:

```
|∇_θ J(θ)| < bound  (gradient clipping)
KL(π_new || π_old) < δ  (trust region)
```

PPO and TRPO enforce these constraints mathematically.

---

## Part 8: The Training Loop

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    LEARNED POLICY LOOP                      │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ OBSERVE  │ →  │  POLICY  │ →  │  ACTION  │              │
│  │   s_t    │    │   π_θ    │    │   a_t    │              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│       │               │               │                     │
│       │               │               ▼                     │
│       │               │         ┌──────────┐               │
│       │               │         │ EXECUTE  │               │
│       │               │         │   ∇²Φ    │               │
│       │               │         └────┬─────┘               │
│       │               │               │                     │
│       │               │               ▼                     │
│       │               │         ┌──────────┐               │
│       │               ◄─────────│ REWARD   │               │
│       │                         │   R_t    │               │
│       │                         └────┬─────┘               │
│       │                               │                     │
│       │                               ▼                     │
│       │                         ┌──────────┐               │
│       ◄─────────────────────────│ OBSERVE' │               │
│                                 │  s_{t+1} │               │
│                                 └──────────┘               │
│                                                              │
│  UPDATE: θ ← θ + α ∇_θ J(θ)                                 │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class LearnedPolicy:
    """
    ML replacement for human judgment.
    Policy outputs Curvent vector.
    """

    def __init__(self, config: Config):
        self.policy_net = PolicyNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims
        )
        self.value_net = ValueNetwork(
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims
        )
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=config.learning_rate
        )
        self.gamma = config.gamma
        self.epsilon = config.ppo_epsilon

    def get_curvent(self, state: State) -> Curvent:
        """
        κ(t) = π_θ(s_t)

        Policy outputs the Curvent vector.
        """
        state_tensor = self.encode_state(state)

        # Get action distribution
        action_mean, action_std = self.policy_net(state_tensor)

        # Sample action (with exploration)
        action = torch.normal(action_mean, action_std)

        # Compose Curvent
        curvent = Curvent(
            local_gradient=action[:self.local_dim],      # ∂Φ/∂x
            global_weight=action[self.local_dim:self.global_dim],  # λ·∇Φ
            external_gates=action[self.global_dim:]      # Σ Γ
        )

        return curvent

    def update(self, trajectories: List[Trajectory]) -> Metrics:
        """
        PPO update step.
        """
        for trajectory in trajectories:
            # Compute advantages
            returns = self.compute_returns(trajectory.rewards)
            values = self.value_net(trajectory.states)
            advantages = returns - values.detach()

            # PPO objective
            old_log_probs = trajectory.log_probs
            new_log_probs = self.policy_net.log_prob(
                trajectory.states, trajectory.actions
            )

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()

            value_loss = F.mse_loss(values, returns)

            # Update
            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

        return Metrics(policy_loss=policy_loss.item(), value_loss=value_loss.item())
```

---

## Part 9: Meta-Learning and Inverse RL

### Meta-Learning: Learning to Learn

```
θ* = argmax_θ E_{τ~p(τ)} [J(φ_θ(τ))]
```

The system learns **how to adapt** to new tasks, not just one task.

### Inverse RL: Inferring Intent

```
R* = argmax_R E_{π_R} [Σ_t R(s_t, a_t)]  s.t. π_R matches expert
```

The system **discovers** the reward function from observing successful outcomes.

### The Learned CEO Condition

Full emergence requires:

1. **Policy learning** — π_θ determines actions
2. **Reward learning** — R is inferred, not specified
3. **Self-improvement** — regret provably decreases
4. **Drift correction** — D = ∇Ψ/∇Φ monitored

When all four conditions are met, **the system runs itself**.

---

## Part 10: Connection to Master Equation

### The Integration

Master Equation:

```
f(x) = W(Φ,Ψ,ε) · γᵗ · ∫₀ᵗ (B·M·S/Σ) dτ
```

Learned Policy provides:

| Component | Policy Mechanism |
|-----------|------------------|
| W (Writability) | Policy learns which states are writable |
| γᵗ (Discount) | Built into policy's value estimation |
| ∫(BMS/Σ)dτ | Policy maximizes this integral |
| Φ (Intent) | Learned from reward signal |
| κ (Curvent) | Policy output = Curvent |

### Curvent as Policy Output

```
π_θ(s_t) → κ(t) → ∇²Φ (if collapse permitted)
```

The learned policy **outputs the execution vector** that drives collapse.

---

## Part 11: The Human Role

### What Humans Specify

1. **Reward function** — Define what "success" means
2. **Constraints** — Risk limits, budget caps, legal bounds
3. **Monitoring thresholds** — When to alert humans

### What Humans Don't Specify

- ❌ Which actions to take
- ❌ When to take them
- ❌ How to balance exploration/exploitation
- ❌ How to adapt to market changes
- ❌ How to allocate across channels

**Everything else is learned.**

### The New Job Description

```
Old: "Decide what marketing to do"
New: "Define what success means and monitor system health"
```

---

## Part 12: Deployment Architecture

### Training vs Inference

**Training (offline):**
- Collect trajectories from environment
- Update policy with PPO/SAC
- Validate on holdout data
- Deploy when performance exceeds threshold

**Inference (online):**
- Observe state
- Query policy for Curvent
- Execute action
- Log outcome
- Periodically retrain

### Safety Constraints

```python
def safe_execute(curvent: Curvent, constraints: Constraints) -> Action:
    """
    Ensure policy output respects safety bounds.
    """
    # Budget constraint
    if curvent.implies_spend() > constraints.max_spend:
        curvent = curvent.scale_to(constraints.max_spend)

    # Risk constraint
    if curvent.risk_score() > constraints.max_risk:
        curvent = curvent.reduce_risk()

    # Legal constraint
    if not constraints.legal_check(curvent):
        curvent = curvent.make_compliant()

    return curvent.to_action()
```

---

## Part 13: Validation Protocol

### Hypothesis

> Learned policies outperform hand-crafted rules on:
> 1. ROI (higher returns through exploration)
> 2. Adaptability (faster response to market shifts)
> 3. Scalability (no human bottleneck)
> 4. Consistency (no human variability)

### Test Design

**A/B Split:**
- Control: Human-designed marketing rules
- Treatment: Learned policy (PPO/SAC)

**Metrics:**
- Cumulative revenue
- Exploration efficiency (coverage of action space)
- Adaptation speed (time to respond to distribution shift)
- Consistency (variance in decision quality)

### Expected Outcomes

```
Human rules:
- ROI: 2.5x
- Adaptation: 2-4 weeks
- Consistency: σ = 0.3

Learned policy:
- ROI: 3.5x (40% improvement)
- Adaptation: 2-4 days (10x faster)
- Consistency: σ = 0.1 (3x more stable)
```

---

## Part 14: Summary

### The Learned Policy Principle

> **The policy is computed, not designed. The Curvent is output, not specified.**

### The Core Equations

1. **Policy:** π*(s) = argmax_a Q*(s,a)
2. **Curvent:** κ(t) = π_θ(s_t)
3. **Update:** θ ← θ + α ∇_θ J(θ)
4. **Stability:** KL(π_new || π_old) < δ
5. **Emergence:** Φ_learned = argmax E[Σ γᵗ R_t]

### The Action

**Stop designing marketing rules. Start learning them.**

---

## References

- Intent Tensor Theory. https://intent-tensor-theory.com/code-equations/
- Knight, A. & Khan, A. (2025). The Funnel Function.
- Schulman, J. et al. Proximal Policy Optimization Algorithms.
- Haarnoja, T. et al. Soft Actor-Critic.
- Sutton, R. & Barto, A. Reinforcement Learning: An Introduction.

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Created by Armstrong Knight & Abdullah Khan

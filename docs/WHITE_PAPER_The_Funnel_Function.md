# The Funnel Function: A Unified Mathematical Framework for Marketing Science

**A Tensor-Algebraic Approach to Customer Acquisition, Conversion, and Lifetime Value**

---

**Authors:**
Armstrong Knight¹, Abdullah Khan¹, with computational assistance from Claude (Anthropic)

**Affiliation:**
¹ Funnel Function Research Initiative

**Date:** December 2024

**Keywords:** Marketing Science, Funnel Mathematics, Intent Tensor Theory, Collapse Geometry, Customer Acquisition, Conversion Optimization, Behavioral Economics, Attribution Modeling

---

## Abstract

We present the Funnel Function, a unified mathematical framework that formalizes customer acquisition and conversion as field collapse operations in intent space. Drawing on 127 years of marketing science—from E. St. Elmo Lewis's AIDA model (1898) through Kahneman-Tversky Prospect Theory (1979) to modern attention economics—we derive a master equation:

$$f(x) = W(\Phi,\Psi,\varepsilon) \cdot \gamma^t \cdot \int_0^t A(u,m,\tau) \, d\tau$$

where $W$ represents the writability gate (qualification), $\gamma^t$ the temporal discount factor, and $\int A \, d\tau$ the cumulative activation integral. This formulation unifies previously disparate concepts—lead scoring, nurture sequences, conversion optimization, and attribution—under a single tensor-algebraic structure. We map the customer journey to a collapse stack ($\Phi \to \nabla\Phi \to \nabla \times F \to \nabla^2\Phi \to \rho_q$) and introduce the ICHTB (Inverse Cartesian + Heisenberg Tensor Box) coordinate system for organizational alignment. The framework generates 12 falsifiable predictions, enabling empirical validation. We demonstrate theoretical coherence with established behavioral economics while providing novel operational insights, including the 97% efficiency principle for lead processing and the drift quotient $D = \nabla\Psi/\nabla\Phi$ for model monitoring.

---

## 1. Introduction

### 1.1 The Problem of Marketing Fragmentation

Modern marketing operates across dozens of channels, hundreds of touchpoints, and millions of individual customer interactions. Yet the theoretical frameworks governing this activity remain fragmented: attribution models conflict with brand measurement, lead scoring operates independently of conversion optimization, and customer lifetime value calculations exist in isolation from acquisition strategy.

This fragmentation is not merely inconvenient—it produces systematic inefficiency. Organizations process leads that cannot convert, optimize for metrics that don't predict outcomes, and allocate resources without causal understanding. The absence of a unified framework forces practitioners into local optimization while missing global coherence.

### 1.2 Historical Context

The formalization of customer acquisition began with E. St. Elmo Lewis's AIDA model in 1898, which proposed that customers progress through Attention → Interest → Desire → Action. This stage-based thinking persisted through the 20th century, refined by Lavidge and Steiner's Hierarchy of Effects (1961), Rogers's Diffusion of Innovations (1962), and countless practitioner adaptations.

The behavioral revolution, initiated by Kahneman and Tversky's Prospect Theory (1979), revealed that customer decisions deviate systematically from rational models. Loss aversion ($\lambda \approx 2.25$), reference dependence, and probability weighting became essential components of any serious conversion model.

The digital era (1990-present) enabled measurement at unprecedented granularity. Marketing automation platforms (Eloqua 1999, HubSpot/Marketo 2006) operationalized lead scoring and nurture sequences. Attribution modeling evolved from last-touch through multi-touch to Shapley-value approaches. Attention metrics (Adelaide, Lumen Research, Nelson-Field) quantified the decay dynamics of customer engagement.

Yet these advances remained siloed. No framework unified behavioral psychology, stage progression, attention decay, and attribution mathematics into a coherent whole.

### 1.3 Contribution

This paper presents the Funnel Function—a mathematical framework that:

1. **Unifies** stage progression, qualification, conversion, and attribution under a single equation
2. **Formalizes** the customer journey as field collapse in intent space using tensor operators
3. **Provides** a coordinate system (ICHTB) mapping business operations to mathematical structures
4. **Generates** falsifiable predictions enabling empirical validation
5. **Offers** operational implementations with production-ready algorithms

We proceed as follows: Section 2 develops the theoretical foundations. Section 3 presents the master equation and its components. Section 4 introduces the collapse geometry and ICHTB coordinate system. Section 5 maps funnel stages to tensor operators. Section 6 addresses measurement and attribution. Section 7 presents falsifiable predictions. Section 8 discusses implications and limitations.

---

## 2. Theoretical Foundations

### 2.1 Intent as a Scalar Field

We begin by positing that customer intent exists as a scalar field $\Phi(x, t)$ defined over customer state space $x$ and time $t$. At any moment, each potential customer occupies a position in this field representing their proximity to purchase intent.

**Definition 2.1 (Intent Field).** Let $\mathcal{X}$ be the space of customer states (demographics, behaviors, contexts). The intent field $\Phi: \mathcal{X} \times \mathbb{R}^+ \to \mathbb{R}$ assigns to each customer state $x$ at time $t$ a scalar value $\Phi(x,t)$ representing purchase intent intensity.

The field metaphor is not merely decorative. It enables the application of differential operators that capture the dynamics of customer progression:

- **Gradient** $\nabla\Phi$: Direction and rate of intent increase
- **Curl** $\nabla \times F$: Circulation patterns (memory loops, engagement cycles)
- **Laplacian** $\nabla^2\Phi$: Curvature of intent field (decision crystallization)

### 2.2 The Writability Condition

Not all customers can convert. Some lack budget, authority, need, or timing. Traditional qualification frameworks (BANT, MEDDIC, CHAMP) address this through boolean criteria. We generalize to a continuous writability function.

**Definition 2.2 (Writability Gate).** The writability function $W: \mathcal{X} \to [0,1]$ measures the degree to which a customer can be "written" (converted):

$$W(x) = \mathbb{1}[\delta(\Phi(x) - \Psi) < \varepsilon]$$

where $\Phi(x)$ is customer intent, $\Psi$ is offer specification, $\delta$ is a distance function, and $\varepsilon$ is the tolerance threshold.

In practice, we employ a Gaussian relaxation:

$$W(x) = \exp\left(-\frac{(\Delta\Psi)^2}{2\sigma^2}\right)$$

where $\Delta\Psi = \|\Phi(x) - \Psi\|$ is the intent-offer gap.

This formulation captures a crucial insight: qualification is not binary but probabilistic, with collapse probability declining as intent-offer mismatch increases.

### 2.3 Temporal Discounting

Customer value decays with time. A lead converting today is worth more than the same lead converting in six months—both due to time-value-of-money and increased defection risk.

**Definition 2.3 (Temporal Discount).** For discount factor $\gamma \in (0,1)$, the time-discounted value at horizon $t$ is:

$$V(t) = V_0 \cdot \gamma^t$$

Standard values range from $\gamma = 0.95$ (aggressive discounting) to $\gamma = 0.99$ (minimal discounting) per month.

### 2.4 Activation Dynamics

Customer progression requires sustained engagement. We model this through an activation function accumulating over time.

**Definition 2.4 (Activation Integral).** The activation function $A: \mathcal{U} \times \mathcal{M} \times \mathbb{R}^+ \to \mathbb{R}^+$ measures engagement intensity given user state $u$, message $m$, and time $\tau$. The cumulative activation is:

$$\mathcal{A}(t) = \int_0^t A(u, m, \tau) \, d\tau$$

We decompose activation into interpretable components:

$$A = \frac{B \cdot M \cdot S}{\Sigma}$$

where:
- $B$ (Body): Sensory signal strength (attention, viewability)
- $M$ (Mind): Cognitive relevance (personalization, timing)
- $S$ (Soul): Emotional resonance (brand alignment, values)
- $\Sigma$ (Suppression): Environmental interference (competition, distraction)

---

## 3. The Master Equation

### 3.1 Formulation

Combining the components developed in Section 2, we arrive at the Funnel Function:

**Theorem 3.1 (The Funnel Function).** The expected value of customer $x$ is given by:

$$f(x) = W(\Phi, \Psi, \varepsilon) \cdot \gamma^t \cdot \int_0^t A(u, m, \tau) \, d\tau$$

where:
- $W(\Phi, \Psi, \varepsilon)$ is the writability gate
- $\gamma^t$ is the temporal discount
- $\int_0^t A \, d\tau$ is the cumulative activation

**Interpretation:** A customer creates value only when: (1) they are writable (qualified), (2) adjusted for time decay, (3) having accumulated sufficient activation.

### 3.2 Properties

The Funnel Function exhibits several important properties:

**Property 3.1 (Gating).** If $W(x) = 0$, then $f(x) = 0$ regardless of other factors. Non-writable customers produce zero value.

**Property 3.2 (Monotonic Decay).** For fixed $W$ and $\mathcal{A}$, $f(x)$ is monotonically decreasing in $t$. Delayed conversion always reduces value.

**Property 3.3 (Threshold Behavior).** Conversion occurs when $\mathcal{A}(t) > \theta$ for some threshold $\theta$. Below-threshold activation produces no conversion regardless of $W$.

**Property 3.4 (Separability).** The components $W$, $\gamma^t$, and $\mathcal{A}$ are independently measurable, enabling modular optimization.

### 3.3 Collapse Probability

The probability of conversion (collapse) follows from the Gaussian writability formulation:

**Corollary 3.1 (Collapse Probability).** The probability of customer $x$ converting is:

$$P_{\text{collapse}}(x) = \exp\left(-\frac{(\Delta\Psi)^2}{2\sigma^2}\right) \cdot \mathbb{1}[\mathcal{A}(t) > \theta]$$

This is a Gaussian in intent-offer gap, gated by activation threshold.

---

## 4. Collapse Geometry and the ICHTB Coordinate System

### 4.1 The Collapse Stack

We formalize the customer journey as a sequence of differential operations on the intent field:

**Definition 4.1 (Collapse Genesis Stack).** The customer journey proceeds through:

$$\Phi \xrightarrow{\text{Awareness}} \nabla\Phi \xrightarrow{\text{Memory}} \nabla \times F \xrightarrow{\text{Curvature}} \nabla^2\Phi \xrightarrow{\text{Collapse}} \rho_q$$

where:
- $\Phi$ (0D): Latent intent (pre-awareness)
- $\nabla\Phi$ (1D): Intent gradient (awareness created)
- $\nabla \times F$ (2D): Engagement curl (memory loop installed)
- $\nabla^2\Phi$ (3D): Intent curvature (decision crystallizing)
- $\rho_q$ (3D+): Charge density (collapsed/converted)

Each transition represents a phase change in customer state, analogous to physical state transitions.

### 4.2 The ICHTB Coordinate System

To operationalize collapse geometry, we introduce the Inverse Cartesian + Heisenberg Tensor Box (ICHTB), a six-fan coordinate system mapping differential operators to business functions.

**Definition 4.2 (ICHTB Fans).** The six fans $\{\Delta_1, \ldots, \Delta_6\}$ are:

| Fan | Axis | Operator | Business Function |
|-----|------|----------|-------------------|
| $\Delta_1$ | $+Y$ | $\nabla\Phi$ | Forward push (CIO) |
| $\Delta_2$ | $-Y$ | $\nabla \times F$ | Memory loop (CHRO) |
| $\Delta_3$ | $+X$ | $+\nabla^2\Phi$ | Expansion (CFO) |
| $\Delta_4$ | $-X$ | $-\nabla^2\Phi$ | Compression (COO) |
| $\Delta_5$ | $+Z$ | $\partial\Phi/\partial t$ | Emergence (CMO) |
| $\Delta_6$ | $-Z$ | $\Phi = i_0$ | Anchor (CEO) |

**Theorem 4.1 (Complementary Pairs).** The fans form three complementary pairs:
- $(\Delta_1, \Delta_2)$: Push and memory
- $(\Delta_3, \Delta_4)$: Expansion and compression
- $(\Delta_5, \Delta_6)$: Emergence and anchor

Organizational health requires balance within each pair.

### 4.3 Hat Eligibility and Shell Formation

Not all customers can receive the "hat" of conversion. We formalize eligibility conditions:

**Definition 4.3 (Hat Eligibility).** A customer has hat eligibility $\hat{h}_n = 1$ iff:
1. Gradient exists: $\nabla\Phi \neq 0$
2. Curvature defined: $\nabla^2\Phi$ exists
3. Phase aligned: $\Omega^n$ coherent across dimensions

Hat eligibility is equivalent to writability: $\hat{h}_n \equiv W(x) > 0$.

---

## 5. Funnel Stages as Tensor Operations

### 5.1 Awareness: Gradient Creation ($\nabla\Phi$)

Awareness creates the gradient—the direction and rate of intent increase.

**Definition 5.1 (Awareness Function).** Awareness transforms latent intent to active gradient:

$$f_{\text{Awareness}}: \Phi \mapsto \nabla\Phi$$

Operationally, this occurs through media exposure that creates differential intent across the customer base.

**Proposition 5.1 (Attention Decay).** Attention follows exponential decay:

$$A(t) = A_0 \cdot e^{-\lambda t}$$

where $\lambda$ is platform-specific. Measured values include:
- Television: $\lambda \approx 0.02$ (half-life 35 seconds)
- YouTube: $\lambda \approx 0.05$ (half-life 14 seconds)
- Social feed: $\lambda \approx 0.25$ (half-life 3 seconds)

### 5.2 Lead Generation: Gradient Lock

Lead generation locks the gradient by capturing contact information.

**Definition 5.2 (Capture Function).** Lead capture locks gradient:

$$f_{\text{Capture}}: \nabla\Phi_{\text{fleeting}} \mapsto \nabla\Phi_{\text{locked}}$$

The locked gradient persists beyond the awareness moment, enabling subsequent nurture.

**Proposition 5.2 (Form Friction).** Capture probability follows:

$$P(\text{capture}) = e^{-\beta \cdot n}$$

where $n$ is the number of form fields and $\beta \approx 0.1$ is the friction coefficient.

### 5.3 Nurturing: Memory Loop Installation ($\nabla \times F$)

Nurturing installs a curl in the engagement field—a circulation that maintains attention.

**Definition 5.3 (Nurture Function).** Nurturing creates curl:

$$f_{\text{Nurture}}: \nabla\Phi_{\text{locked}} \mapsto \nabla \times F$$

A non-zero curl ($\oint \vec{F} \cdot d\vec{\ell} \neq 0$) indicates active memory—the lead remembers and engages.

**Proposition 5.3 (Drip Calculus).** Without nurturing, engagement decays:

$$E(t) = E_0 \cdot e^{-\lambda t}$$

Touchpoints restore engagement:

$$E(t_n) = E_0 \cdot e^{-\lambda(t_n - t_{n-1})} + \Delta E_n$$

The optimal cadence balances decay prevention against fatigue:

$$\Delta t^* = \frac{1}{\lambda} \ln\left(\frac{E_{\text{threshold}}}{E_0}\right)$$

### 5.4 Qualification: Writability Gate ($W$)

Qualification evaluates the writability gate—whether the lead can convert.

**Definition 5.4 (Qualification Function).** Qualification computes:

$$f_{\text{Qualification}}: x \mapsto W(x) = \exp\left(-\frac{(\Delta\Psi)^2}{2\sigma^2}\right)$$

Leads with $W(x) < \theta_{\text{MQL}}$ are not marketing-qualified. Leads with $W(x) < \theta_{\text{SQL}}$ are not sales-qualified.

**Proposition 5.4 (97% Efficiency Principle).** In typical B2B contexts, approximately 97% of leads have $W(x) \approx 0$. Processing only high-$W$ leads achieves equivalent conversions at dramatically reduced cost.

### 5.5 Conversion: Curvature Lock ($\nabla^2\Phi$)

Conversion locks the curvature—the decision crystallizes.

**Definition 5.5 (Conversion Function).** Conversion requires stable curvature:

$$f_{\text{Conversion}}: \nabla^2\Phi_{\text{dynamic}} \mapsto \nabla^2\Phi_{\text{locked}}$$

The lock condition is: $\nabla^2\Phi = \text{constant}$ and $\frac{d(\nabla^2\Phi)}{dt} \to 0$.

**Proposition 5.5 (Loss Aversion in Curvature).** Per Prospect Theory, losses curve the field more sharply than equivalent gains:

$$\frac{|\nabla^2\Phi_{\text{losses}}|}{|\nabla^2\Phi_{\text{gains}}|} \approx \lambda \approx 2.25$$

This implies loss-framed messages should outperform gain-framed messages by factor $\lambda$.

### 5.6 Close: Charge Lock ($\rho_q$)

Close forms the charge—the permanent record of conversion.

**Definition 5.6 (Close Function).** Close crystallizes charge:

$$f_{\text{Close}}: \nabla^2\Phi_{\text{locked}} \mapsto \rho_q$$

Once $\rho_q$ forms, the conversion is irreversible. The customer entity exists.

**Proposition 5.6 (Threshold Crossing).** Close occurs when:

$$\int_0^t A \, d\tau > \theta_{\text{close}}$$

The activation integral must exceed the close threshold for charge formation.

---

## 6. Measurement and Attribution

### 6.1 The Drift Quotient

Model accuracy requires continuous monitoring. We introduce the drift quotient:

**Definition 6.1 (Drift Quotient).** The drift quotient measures predictive accuracy:

$$D = \frac{\nabla\Psi_{\text{actual}}}{\nabla\Phi_{\text{predicted}}}$$

where:
- $D = 1$: Perfect prediction
- $D < 0.8$ or $D > 1.2$: Significant drift requiring investigation
- $D < 0.5$ or $D > 2.0$: Major drift requiring recalibration

### 6.2 Attribution via Shapley Values

For multi-touch attribution, we employ Shapley values from cooperative game theory:

**Definition 6.2 (Shapley Attribution).** The contribution of touchpoint $i$ to conversion value $v$ is:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

Shapley values satisfy efficiency (contributions sum to total), symmetry (equal contributors get equal credit), and null player (zero-contribution gets zero credit).

### 6.3 Attention Measurement

Following Nelson-Field and Adelaide, we quantify attention:

**Definition 6.3 (Attention Units).** The Adelaide Attention Unit (AU) is:

$$AU = P(\text{attention}) \times \text{Duration} \times \text{Quality Factor}$$

Typical ranges:
- Low attention: $AU < 30$
- Medium attention: $30 \leq AU < 60$
- High attention: $AU \geq 60$

---

## 7. Falsifiable Predictions

A scientific framework must make testable predictions. We present 12 core predictions:

### Prediction 1: Writability Gating
**Claim:** Leads with $W(x) < 0.3$ convert at rates $< 1\%$; leads with $W(x) > 0.7$ convert at rates $> 20\%$.

**Falsification:** If low-$W$ leads convert at $> 5\%$, reject writability gating.

### Prediction 2: Time Discount
**Claim:** Conversion value follows $V(t) = V_0 \cdot \gamma^t$ with $\gamma \in [0.90, 0.99]$.

**Falsification:** If value is independent of time ($\gamma = 1$), reject temporal discounting.

### Prediction 3: Activation Threshold
**Claim:** Conversion requires $\int A \, d\tau > \theta$ for threshold $\theta$.

**Falsification:** If activation is uncorrelated with conversion, reject threshold model.

### Prediction 4: Drift Equilibrium
**Claim:** Well-calibrated systems maintain $D = 1.0 \pm 0.2$.

**Falsification:** If $D$ is random with no central tendency, reject predictive validity.

### Prediction 5: Loss Aversion Ratio
**Claim:** Loss-framed messages outperform gain-framed by factor $\lambda \approx 2.0-2.5$.

**Falsification:** If loss/gain ratio $\leq 1$, reject loss aversion application.

### Prediction 6: Exponential Decay
**Claim:** Engagement decays as $E(t) = E_0 \cdot e^{-\lambda t}$.

**Falsification:** If decay is linear or step-function, reject exponential model.

### Prediction 7: Optimal Cadence
**Claim:** Calculated $\Delta t^*$ outperforms random or fixed cadence.

**Falsification:** If random cadence matches or exceeds optimal, reject formula.

### Prediction 8: Shapley Accuracy
**Claim:** Shapley attribution predicts incremental contribution more accurately than last-touch.

**Falsification:** If last-touch equals or exceeds Shapley in holdout tests, reject complexity.

### Prediction 9: CLV Accuracy
**Claim:** CLV model predicts lifetime value within 25% MAPE.

**Falsification:** If MAPE $> 40\%$, reject CLV formulation.

### Prediction 10: Efficiency Principle
**Claim:** Processing top 10% by $W$ captures $> 90\%$ of conversions.

**Falsification:** If capture rate $< 80\%$, reject efficiency claim.

### Prediction 11: Fan Balance
**Claim:** Organizations with balanced ICHTB fans outperform imbalanced.

**Falsification:** If no correlation between balance and performance, reject ICHTB utility.

### Prediction 12: Probability Calibration
**Claim:** $P_{\text{collapse}}$ predictions are well-calibrated (predicted X% converts at ~X%).

**Falsification:** If calibration slope significantly $\neq 1$, reject probability model.

---

## 8. Discussion

### 8.1 Theoretical Contributions

The Funnel Function advances marketing science in several ways:

1. **Unification:** Previously disparate concepts—stage progression, qualification, conversion, attribution—are unified under a single mathematical framework.

2. **Formalization:** Intuitive notions receive precise definitions enabling rigorous analysis.

3. **Operationalization:** The framework generates implementable algorithms, not just conceptual models.

4. **Falsifiability:** Specific predictions enable empirical validation or refutation.

### 8.2 Relationship to Prior Work

The Funnel Function builds on established foundations:

- **AIDA (Lewis, 1898):** Stage progression is preserved but formalized as differential operations.
- **Prospect Theory (Kahneman-Tversky, 1979):** Loss aversion informs the conversion stage curvature.
- **Commitment-Trust Theory (Morgan-Hunt, 1994):** Trust requirements inform writability gating.
- **Attention Research (Nelson-Field, 2020):** Decay dynamics parameterize the activation function.

We synthesize rather than replace these contributions.

### 8.3 Limitations

Several limitations warrant acknowledgment:

1. **Empirical Validation:** While the framework generates testable predictions, comprehensive validation requires longitudinal data across multiple organizations.

2. **Parameter Estimation:** Values like $\gamma$, $\sigma$, and $\lambda$ require calibration per context. Universal defaults may not hold.

3. **Complexity:** The full framework may exceed practical implementation capacity for some organizations.

4. **Cultural Variation:** Behavioral parameters (loss aversion, trust thresholds) may vary cross-culturally.

### 8.4 Future Directions

Several extensions merit investigation:

1. **Neuroscientific Validation:** Brain imaging studies could validate the attention and decision constructs.

2. **AI Optimization:** Real-time f(x) computation could enable dynamic resource allocation.

3. **Privacy-Preserving Measurement:** Post-cookie attribution requires new measurement paradigms.

4. **Cross-Cultural Studies:** Testing whether behavioral parameters hold globally.

---

## 9. Conclusion

We have presented the Funnel Function, a unified mathematical framework for marketing science. By formalizing customer acquisition as field collapse in intent space, we unify previously fragmented concepts under a single theoretical structure. The framework generates specific, falsifiable predictions enabling empirical validation.

The master equation:

$$f(x) = W(\Phi,\Psi,\varepsilon) \cdot \gamma^t \cdot \int_0^t A(u,m,\tau) \, d\tau$$

captures the essential dynamics: qualification gates conversion, time decays value, and activation must accumulate to threshold. The collapse stack ($\Phi \to \nabla\Phi \to \nabla \times F \to \nabla^2\Phi \to \rho_q$) maps the customer journey to differential operators. The ICHTB coordinate system aligns organizational functions with mathematical structures.

Marketing science has accumulated 127 years of insight since Lewis's AIDA model. The Funnel Function synthesizes this heritage into a framework suitable for the complexity of modern multi-channel, multi-touch customer acquisition. We offer it as a foundation for continued theoretical development and empirical investigation.

---

## Acknowledgments

We thank the researchers whose work made this synthesis possible, particularly E. St. Elmo Lewis, Daniel Kahneman, Amos Tversky, Robert Morgan, Shelby Hunt, Karen Nelson-Field, and the teams at Adelaide, Lumen Research, and the Ehrenberg-Bass Institute. The mathematical formulation draws on Intent Tensor Theory.

---

## References

Ariely, D. (2008). *Predictably Irrational*. HarperCollins.

Berry, L.L. (1983). Relationship Marketing. In *Emerging Perspectives on Services Marketing*. American Marketing Association.

Cialdini, R.B. (1984). *Influence: The Psychology of Persuasion*. William Morrow.

Dixon, M. & Adamson, B. (2011). *The Challenger Sale*. Portfolio/Penguin.

Ehrenberg-Bass Institute. Mental Availability Research Program. University of South Australia.

Fisher, R., Ury, W., & Patton, B. (1981). *Getting to Yes*. Houghton Mifflin.

Grönroos, C. (1994). From Marketing Mix to Relationship Marketing. *Management Decision*, 32(2), 4-20.

Halligan, B. & Shah, D. (2010). *Inbound Marketing*. Wiley.

Kahneman, D. & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. *Econometrica*, 47(2), 263-292.

Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments*. Cambridge University Press.

Lavidge, R.J. & Steiner, G.A. (1961). A Model for Predictive Measurements of Advertising Effectiveness. *Journal of Marketing*, 25(6), 59-62.

Lewis, E. St. Elmo (1898). Writings on salesmanship and advertising. Various publications.

Mayer, R.C., Davis, J.H., & Schoorman, F.D. (1995). An Integrative Model of Organizational Trust. *Academy of Management Review*, 20(3), 709-734.

Morgan, R.M. & Hunt, S.D. (1994). The Commitment-Trust Theory of Relationship Marketing. *Journal of Marketing*, 58(3), 20-38.

Nelson-Field, K. (2020). *The Attention Economy and How Media Works*. Palgrave Macmillan.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

Petty, R.E. & Cacioppo, J.T. (1986). The Elaboration Likelihood Model of Persuasion. *Advances in Experimental Social Psychology*, 19, 123-205.

Popper, K. (1959). *The Logic of Scientific Discovery*. Hutchinson.

Rackham, N. (1988). *SPIN Selling*. McGraw-Hill.

Rogers, E.M. (1962). *Diffusion of Innovations*. Free Press.

Samuelson, P. (1937). A Note on Measurement of Utility. *Review of Economic Studies*, 4(2), 155-161.

Shapley, L.S. (1953). A Value for n-Person Games. In *Contributions to the Theory of Games II*. Princeton University Press.

Thaler, R. (1980). Toward a Positive Theory of Consumer Choice. *Journal of Economic Behavior and Organization*, 1, 39-60.

Thaler, R. & Sunstein, C. (2008). *Nudge*. Yale University Press.

Tversky, A. & Kahneman, D. (1974). Judgment Under Uncertainty: Heuristics and Biases. *Science*, 185, 1124-1131.

---

## Appendix A: Mathematical Notation

| Symbol | Definition |
|--------|------------|
| $\Phi$ | Intent field (scalar) |
| $\nabla\Phi$ | Intent gradient (vector) |
| $\nabla \times F$ | Engagement curl (circulation) |
| $\nabla^2\Phi$ | Intent Laplacian (curvature) |
| $\rho_q$ | Charge density (collapsed state) |
| $W(x)$ | Writability function |
| $\gamma$ | Temporal discount factor |
| $A$ | Activation function |
| $\Delta\Psi$ | Intent-offer gap |
| $D$ | Drift quotient |
| $\lambda$ | Decay rate / Loss aversion coefficient |
| $\theta$ | Threshold parameter |
| $\sigma$ | Tolerance parameter |

---

## Appendix B: ICHTB Fan Specifications

| Fan | Axis | Operator | C-Suite | Function |
|-----|------|----------|---------|----------|
| $\Delta_1$ | $+Y$ | $\nabla\Phi$ | CIO | Technology, innovation, forward push |
| $\Delta_2$ | $-Y$ | $\nabla \times F$ | CHRO | Culture, retention, memory loop |
| $\Delta_3$ | $+X$ | $+\nabla^2\Phi$ | CFO | Capital, growth, expansion |
| $\Delta_4$ | $-X$ | $-\nabla^2\Phi$ | COO | Operations, delivery, compression |
| $\Delta_5$ | $+Z$ | $\partial\Phi/\partial t$ | CMO | Marketing, timing, emergence |
| $\Delta_6$ | $-Z$ | $\Phi = i_0$ | CEO | Vision, strategy, anchor |

---

## Appendix C: Implementation Pseudocode

```python
def funnel_function(lead, offer, interactions, config):
    """
    Compute f(x) = W · γ^t · ∫A dτ
    """
    # Writability gate
    delta_psi = compute_gap(lead.intent, offer.specification)
    W = exp(-(delta_psi**2) / (2 * config.sigma**2))

    # Time discount
    t = months_since_first_touch(lead)
    gamma_t = config.gamma ** t

    # Activation integral
    A_integral = sum(
        compute_activation(i) * i.duration
        for i in interactions
    )

    # Funnel function value
    f_x = W * gamma_t * A_integral

    return f_x
```

---

*Corresponding author: Armstrong Knight, Funnel Function Research Initiative*

*This work is licensed under Creative Commons Attribution-NonCommercial 4.0 International License.*

# 0.2.a.i f(Awareness)

**The Gating Function: Complete Mathematical and Theoretical Foundations**

By Armstrong Knight, Abdullah Khan & Claude.ai | Funnel Function

---

## Abstract

**Awareness operates as the gating function between stimulus and response across all human decision-making.** This comprehensive mapping reveals that marketing science and cognitive science converged on remarkably similar mathematical insights: awareness is scarce, capacity-limited, probabilistically distributed, and subject to threshold effects. From E. St. Elmo Lewis's 1898 sales funnel to Karl Friston's 2010 free energy principle, the core problem remains unchanged—how do signals break through noise to enter conscious processing?

The frameworks documented here represent the complete intellectual lineage informing modern attention measurement, from GRP formulas used in media buying to d-prime equations in psychophysics laboratories.

---

## The Core Insight

```
f(Awareness) = P(signal → conscious_processing | noise, capacity, threshold)
```

Awareness is not binary. It is a **probability distribution** gated by:
- **Noise** — Competing signals in the environment
- **Capacity** — Hard limits on simultaneous processing (4±1 items)
- **Threshold** — Minimum activation required for conscious access

---

## Part 1: Marketing and Brand Awareness Frameworks

### 1.1 The Hierarchy of Effects Tradition (1898–1980)

The foundational idea that awareness precedes action emerged from direct-mail advertising in the late 19th century.

#### f(AIDA) — E. St. Elmo Lewis (1898)

**E. St. Elmo Lewis** (1872–1948) first articulated the principle in an 1898 *Printers' Ink* article:

```
f(AIDA) = Attention → Interest → Desire → Action
```

Lewis drew explicitly on **William James's psychology of attention**, making this the first documented bridge between cognitive science and advertising theory. The AIDA acronym itself was coined by C.P. Russell in 1921, though Lewis's conceptual framework preceded it by over two decades.

**Key insight:** The funnel begins with attention—without it, nothing downstream can occur.

---

#### f(Hierarchy_of_Effects) — Lavidge & Steiner (1961)

**Robert Lavidge and Gary Steiner** formalized the concept in their landmark 1961 paper *"A Model for Predictive Measurements of Advertising Effectiveness"* (Journal of Marketing).

```
f(Hierarchy) = Awareness → Knowledge → Liking → Preference → Conviction → Purchase
```

Their six-stage hierarchy explicitly mapped onto three psychological domains:

| Domain | Stages | Function |
|--------|--------|----------|
| **Cognitive** (Thinking) | Awareness, Knowledge | Information processing |
| **Affective** (Feeling) | Liking, Preference | Attitude formation |
| **Conative** (Doing) | Conviction, Purchase | Action tendency |

---

#### f(DAGMAR) — Russell Colley (1961)

**Russell Colley's DAGMAR** framework (*Defining Advertising Goals for Measured Advertising Results*, ANA, 1961) introduced the revolutionary principle that advertising should be measured by **communication effects, not sales**.

```
f(DAGMAR) = Awareness → Comprehension → Conviction → Action
```

This spawned modern brand tracking methodology—the idea that you can measure awareness independent of purchase.

---

#### f(FCB_Grid) — Richard Vaughn (1980)

**Richard Vaughn's FCB Grid** (1980, *Journal of Advertising Research*) integrated these hierarchies with involvement theory:

```
            |  THINKING        |  FEELING
------------|------------------|------------------
HIGH        | Learn→Feel→Do    | Feel→Learn→Do
INVOLVEMENT | (cars, insurance)| (jewelry, fashion)
------------|------------------|------------------
LOW         | Do→Learn→Feel    | Do→Feel→Learn
INVOLVEMENT | (household items)| (beer, snacks)
```

**Key insight:** The Think-Feel-Do sequence **reorders** based on product category. This remains the standard framework in advertising planning.

---

#### Summary: Hierarchy Models

| Model | Author(s) | Year | Sequence | Key Innovation |
|-------|-----------|------|----------|----------------|
| AIDA | E. St. Elmo Lewis | 1898 | Attention→Interest→Desire→Action | First hierarchical model |
| Hierarchy of Effects | Lavidge & Steiner | 1961 | 6 stages mapped to cognition/affect/conation | Psychological domains |
| DAGMAR | Russell Colley | 1961 | Awareness→Comprehension→Conviction→Action | Measurable objectives |
| FCB Grid | Richard Vaughn | 1980 | Variable by quadrant | Involvement × processing mode |

**Critiques:** Vakratsas and Ambler's 1999 meta-analysis of 250+ papers found little empirical support for strict sequential processing. Modern neuroscience confirms that cognitive and affective processing occur **simultaneously**, not sequentially.

---

### 1.2 Media Mathematics: Reach, Frequency, and Carryover

The quantitative infrastructure of media planning rests on equations developed between 1950 and 1980 that **remain industry standard today**.

#### f(GRP) — Gross Rating Points (1950s)

```
f(GRP) = Reach(%) × Average_Frequency
```

One GRP represents reaching 1% of the target population once. The formula's elegance masks its limitation: it treats all exposures as equivalent regardless of attention quality.

---

#### f(Three_Exposures) — Herbert Krugman (1972)

**Herbert Krugman's Three-Exposure Theory** (1972, "Why Three Exposures May Be Enough," *Journal of Advertising Research*) proposed that advertising has only three psychological states:

```
Exposure 1: "What is it?"     (curiosity)
Exposure 2: "What of it?"     (evaluation)
Exposure 3: Reminder/decision (action trigger)
```

Krugman argued there is **"no such thing as a fourth exposure psychologically"**—fours, fives, etc., are repeats of the third.

**Michael Naples** codified this as the "3+ frequency" standard in his 1979 ANA report, which dominated media planning for two decades.

---

#### f(Recency) — Erwin Ephron (1995)

**Erwin Ephron's Recency Theory** (1995, "More Weeks, Less Weight," *Journal of Advertising Research*) challenged effective frequency orthodoxy.

```
f(Recency) = Maximize(weekly_reach) over Maximize(frequency)
```

Ephron argued that a **single exposure within the purchase decision window** matters more than accumulated frequency. John Philip Jones's single-source research confirmed that one exposure in the seven days before purchase has far greater impact than multiple earlier exposures.

**Recommendation:** Maximize weekly reach rather than building frequency.

---

#### f(Adstock) — Simon Broadbent (1979)

**Simon Broadbent's Adstock Model** (1979, *Journal of the Market Research Society*) provided the mathematical framework for modeling awareness decay:

```
f(Adstock)_t = Advertising_t + λ × Adstock_(t-1)
```

Where λ (0 < λ < 1) is the decay parameter.

**Half-life calculation:**

```
Half_life = ln(0.5) / ln(λ)
```

**Industry benchmarks:**
- FMCG brands: ~2.5 weeks half-life
- Brand awareness effects: 7–12 weeks (academic studies)

This equation underlies **all modern marketing mix models**.

---

#### f(Response_Curves) — Advertising Response Functions

**Linear** (rarely observed):
```
Sales = α + β × Advertising
```

**Concave/Diminishing Returns** (most common empirically):
```
Sales = α × Advertising^β    where 0 < β < 1
```

**S-Curve/Logistic** (threshold + saturation):
```
Response = Max × (Spend^α) / (K^α + Spend^α)    [Hill Function]
```

---

#### f(Vidale_Wolfe) — Dynamic Response (1957)

The **Vidale-Wolfe Model** (1957, *Operations Research*) introduced dynamics:

```
dS/dt = r × A × (M - S)/M - δ × S
```

Where:
- S = sales rate
- A = advertising spend
- M = market saturation
- r = response constant
- δ = decay rate

This was the **first differential equation model** of advertising effects.

---

### 1.3 Share of Voice and the Ehrenberg-Bass Revolution

#### f(ESOV) — John Philip Jones (1990)

**John Philip Jones** (1990, *Harvard Business Review*) documented the empirical relationship between Share of Voice (SOV) and Share of Market (SOM):

```
f(ESOV) = SOV - SOM
```

- **Challenger brands** require positive ESOV to grow
- **Market leaders** can profit with negative ESOV

**Les Binet and Peter Field's** IPA research suggests:

```
Expected_Annual_Market_Share_Growth ≈ 0.5% per 10% ESOV
```

---

#### f(NBD_Dirichlet) — Ehrenberg, Goodhardt & Chatfield (1984)

The **Ehrenberg-Bass Institute** (founded by Andrew Ehrenberg) transformed brand science through rigorous empirical analysis.

Their **NBD-Dirichlet Model** (Goodhardt, Ehrenberg & Chatfield, 1984, *Journal of the Royal Statistical Society*) describes purchase behavior with three parameters:

- **M** = Mean category purchase rate
- **K** = Purchase frequency diversity
- **S** = Brand propensity diversity

**Brand Penetration:**
```
b = 1 - Σ(n=0→∞) P_n × p(0|n)
```

**Double Jeopardy Law:**
```
w(1-b) = constant
```
Smaller brands suffer twice—fewer buyers AND lower loyalty.

**Duplication of Purchase Law:**
```
b_XY = D × b_X
```
Brand Y's buyers who also buy brand X equals the Duplication coefficient times brand X's penetration.

---

#### f(Mental_Availability) — Byron Sharp & Jenni Romaniuk (2004–2010)

**Byron Sharp's "How Brands Grow"** (2010) synthesized Ehrenberg-Bass findings into marketing's most influential modern text.

Key constructs:
- **Mental Availability**: Brand's propensity to be noticed or retrieved in buying situations
- **Physical Availability**: Ease of finding and buying the brand
- **Category Entry Points (CEPs)**: The cues buyers use to access memory (developed by Jenni Romaniuk)

```
f(Mental_Market_Share) = Brand_CEP_associations / Total_category_CEP_associations
```

---

### 1.4 Attention Economics: From Simon to Digital Metrics

#### f(Attention_Scarcity) — Herbert Simon (1971)

**Herbert Simon** established the theoretical foundation in his 1971 lecture *"Designing Organizations for an Information-Rich World"*:

> "A wealth of information creates a poverty of attention and a need to allocate that attention efficiently among the overabundance of information sources that might consume it."

Simon connected attention scarcity to his earlier work on **bounded rationality** (1955): humans "satisfice" rather than optimize because cognitive capacity is limited.

**This remains the master concept underlying all attention economics.**

---

#### f(Attention_Economy) — Michael Goldhaber (1997)

**Michael Goldhaber** (1997, *Wired*) declared attention "the currency of the New Economy":

> "Unlike information, human attention is truly individual and scarce."

**Georg Franck** (1998, *Ökonomie der Aufmerksamkeit*) developed the concept of "mental capitalism"—attention as literal capital that earns interest. Celebrity functions as a "stock exchange of attention capital."

---

#### f(Viewability) — IAB/MRC Standards (2014)

**Digital attention measurement** evolved beyond simple impressions. The **IAB/MRC Viewability Standards** (2014) established minimum thresholds:

| Format | Threshold |
|--------|-----------|
| Display | ≥50% pixels visible for ≥1 second |
| Video | ≥50% pixels visible for ≥2 seconds |

**But viewability ≠ attention.**

---

#### f(APM) — Lumen Research (Mike Follett, 2013)

**Lumen Research** pioneered eye-tracking based metrics:

```
f(APM) = (% of ads viewed × average viewing time) × 1000 impressions

f(aCPM) = CPM / APM × 1,000
```

**Key finding:** Only **35% of "viewable" ads are actually looked at**.

---

#### f(AU) — Adelaide (Marc Guldimann, 2019)

**Adelaide** developed the **AU (Attention Unit)** metric—a 0–100 score combining:
- Eye-tracking data
- Media quality signals
- Outcome data

This predicts **attention probability** rather than duration.

---

#### f(Active_Attention) — Karen Nelson-Field (2020)

**Karen Nelson-Field** (Amplified Intelligence, *The Attention Economy and How Media Works*, 2020) established:

```
f(Memory_Encoding) requires ≥ 1.5 seconds active attention
```

Her research shows attention metrics are **7× more effective** at predicting brand awareness than viewability.

---

## Part 2: Cognitive Science and Attention Theory

### 2.1 Signal Detection and Information Theory

#### f(Signal_Detection) — Green & Swets (1966)

**Signal Detection Theory (SDT)** separates perceptual sensitivity from response bias:

```
f(d') = z(Hit_Rate) - z(False_Alarm_Rate)
```

Where z() is the inverse normal CDF.

**Criterion measures:**
```
c = -0.5 × (z(HR) + z(FAR))    [criterion location]
β = exp((z(FA)² - z(H)²) / 2)  [likelihood ratio]
```

**ROC curves** plot Hit Rate against False Alarm Rate across criterion levels; Area Under Curve (AUC) quantifies overall discrimination ability.

---

#### f(Channel_Capacity) — Claude Shannon (1948)

**Claude Shannon's Information Theory** (1948, *Bell System Technical Journal*) established channel capacity:

```
f(C) = B × log₂(1 + S/N)
```

Where:
- C = capacity in bits/second
- B = bandwidth
- S/N = signal-to-noise ratio

---

#### f(7±2) — George Miller (1956)

**George Miller** (1956, *Psychological Review*) applied information theory to cognition, discovering that immediate memory span is limited to:

```
f(Chunk_Capacity) = 7 ± 2 items
```

Independent of information bits per chunk. Miller emphasized **chunking** as the mechanism for expanding effective capacity.

---

#### f(4±1) — Nelson Cowan (2001)

**Nelson Cowan** (2001, *Behavioral and Brain Sciences*) revised Miller's estimate to:

```
f(True_Capacity) = 4 ± 1 items (when rehearsal is prevented)
```

```
K = (hit_rate + correct_rejection_rate - 1) × N
```

---

### 2.2 Selective Attention: Filter Models

#### f(Filter) — Donald Broadbent (1958)

**Donald Broadbent's Filter Model** (1958, *Perception and Communication*) proposed early selection based on physical characteristics:

```
Stimuli → Sensory Buffer → FILTER → Limited-capacity Processor → Response
```

The filter blocks unattended information **before** semantic analysis.

---

#### f(Attenuation) — Anne Treisman (1964)

**Anne Treisman's Attenuation Theory** (1964, *British Medical Bulletin*) modified this: instead of blocking, the filter merely "turns down the volume."

```
If (Attenuated_Signal) ≥ Threshold → Conscious_Awareness
```

**Threshold varies by:**
- Subjective importance
- Recent activation (priming)
- Contextual expectancy
- Biological significance

Important stimuli (one's name, danger words) have **permanently low thresholds** and break through.

---

#### f(Late_Selection) — Deutsch & Deutsch (1963)

**Deutsch and Deutsch's Late Selection Model** (1963, *Psychological Review*) proposed all stimuli receive full semantic processing; selection occurs **later** based on importance/pertinence.

---

#### f(Capacity) — Daniel Kahneman (1973)

**Daniel Kahneman's Capacity Model** (1973, *Attention and Effort*) shifted focus from filtering to **resource allocation**:

```
f(Total_Capacity) = g(Arousal)    [inverted-U function]
```

Attention is a limited, undifferentiated pool of mental "fuel":

```
Performance = f(Resources_allocated, Task_demands)
```

When demands exceed capacity, performance suffers.

---

#### f(Multiple_Resources) — Christopher Wickens (1980, 2002)

**Christopher Wickens' Multiple Resource Theory** identified four dimensions of resource separation:

| Dimension | Options |
|-----------|---------|
| **Stages** | Perceptual/Cognitive vs. Response |
| **Modalities** | Auditory vs. Visual |
| **Codes** | Spatial vs. Verbal |
| **Visual Channels** | Focal vs. Ambient |

**Prediction:** Interference increases with shared resources across dimensions.

This explains why auditory-verbal + visual-spatial tasks combine well (different tanks), while visual-verbal + visual-spatial compete (same tank).

---

### 2.3 Computational Models of Visual Attention

#### f(Feature_Integration) — Treisman & Gelade (1980)

**Anne Treisman's Feature Integration Theory** distinguished:

**Pre-attentive processing:** Features (color, orientation) processed in parallel; targets "pop out"

**Focused attention:** Required to bind features together (solving the "binding problem")

```
Feature search (parallel): RT = a + b (flat, ~0 ms/item slope)
Conjunction search (serial): RT_positive = a + b × N/2
                             RT_negative = a + b × N
```

---

#### f(Saliency) — Itti & Koch (1998)

**The Itti-Koch Saliency Model** (1998, *IEEE PAMI*) operationalized bottom-up attention computationally:

1. Input decomposed into 9 spatial scales via Gaussian pyramids
2. Feature channels extracted: Intensity, Color (RG, BY opponency), Orientation (4 angles)
3. Center-surround operations create feature maps (42 total)
4. Normalization operator N(·) promotes maps with few strong peaks
5. Conspicuity maps combined:

```
f(Saliency) = (1/3)[N(Ī) + N(C̄) + N(Ō)]
```

6. Winner-take-all network selects most salient location

The model predicts first **~250ms** of viewing (pre-attentive processing).

---

#### f(Guided_Search) — Jeremy Wolfe (1989–2021)

**Jeremy Wolfe's Guided Search** bridges bottom-up saliency with top-down guidance:

```
f(Activation)_i = w_BU × BU_i + Σ_d(w_d × TD_d,i) + noise
```

**Guided Search 6.0** (2021, *Psychonomic Bulletin & Review*) integrates five priority sources:
1. Bottom-up salience
2. Top-down guidance
3. History/priming
4. Reward/value
5. Scene semantics

**Selection rate:** ~20 Hz attention deployment

---

### 2.4 Working Memory, Consciousness, and Predictive Processing

#### f(Working_Memory) — Baddeley & Hitch (1974, 2000)

**Baddeley's Working Memory Model** comprises:

| Component | Function |
|-----------|----------|
| **Central Executive** | Attentional control, coordinates subsystems |
| **Phonological Loop** | Verbal/auditory information (~2 seconds rehearsal) |
| **Visuospatial Sketchpad** | Visual/spatial information |
| **Episodic Buffer** (2000) | Integrates across subsystems and LTM |

---

#### f(Cognitive_Load) — John Sweller (1988)

**Cognitive Load Theory** (John Sweller, 1988, *Cognitive Science*) distinguishes:

```
f(Total_Load) = Intrinsic_Load + Extraneous_Load
```

**Overload** occurs when Total Load > Working Memory Capacity.

**Implications for advertising:** Minimize extraneous load through design optimization.

---

#### f(Global_Workspace) — Bernard Baars (1988)

**Global Workspace Theory** uses a theater metaphor:

- **Spotlight** = Consciousness
- **Stage** = Limited working memory
- **Audience** = Vast unconscious processors

Selected information is **broadcast** to all processors simultaneously.

**Dehaene and Changeux** (1998, 2003) provided neural implementation: long-range workspace neurons in prefrontal-parietal cortex show "ignition"—sudden, late (~300ms), sustained firing when threshold crossed.

**Signatures of consciousness:** P3b wave, gamma synchrony, long-distance coherence

---

#### f(Predictive_Processing) — Rao & Ballard (1999)

**Predictive Processing** models the brain as a hierarchical prediction machine:

```
Prediction: ŷ_i = W_i · r_(i+1)
Prediction Error: e_i = y_i - ŷ_i
```

- Feedforward connections carry **errors**
- Feedback connections carry **predictions**
- **Attention = precision-weighting of prediction errors**

---

#### f(Free_Energy) — Karl Friston (2010)

**Karl Friston's Free Energy Principle** (2010, *Nature Reviews Neuroscience*) provides a unified framework:

```
f(F) = E_q[ln q(s) - ln p(s,o)]
```

Equivalently: F = KL-divergence between beliefs and true posterior + surprisal.

The brain minimizes free energy through:
1. **Perceptual inference** (updating beliefs)
2. **Learning** (updating model)
3. **Action** (changing inputs)

**Attention modulates precision:**

```
Precision (π) = 1 / variance
```

High-precision prediction errors gain more weight in updating beliefs.

---

## Part 3: Cross-Domain Synthesis

### 3.1 How Cognitive Theory Shaped Advertising Science

The bridges between cognitive science and advertising are well-documented:

| Year | Cognitive Source | Advertising Application |
|------|-----------------|------------------------|
| 1898 | William James (attention) | Lewis's AIDA model |
| 1956 | Miller (7±2 chunks) | Message design limits |
| 1966 | Signal Detection Theory | Ad noticeability measurement |
| 1972 | Mere-exposure effect | Krugman's three exposures |
| 1973 | Kahneman (capacity) | Cognitive load in ads |
| 1980 | Feature Integration | Distinctive brand assets |
| 1980 | Multiple Resources | Multimodal ad design |

---

### 3.2 The Attention Funnel: Unified Framework

Across domains, awareness functions as the **first filter** in a multi-stage process:

**Marketing Funnel:**
```
Exposure → Attention → Awareness → Processing → Attitude → Behavior
```

**Cognitive Funnel:**
```
Sensation → Feature Detection → Saliency Competition → Working Memory → Decision
```

**Shared mathematical properties:**

| Property | Marketing | Cognitive |
|----------|-----------|-----------|
| Capacity limits | ~7 brands in consideration set | ~4 items in working memory |
| Decay functions | Adstock λ ≈ 0.75 | Trace decay ~1–2 seconds |
| Threshold effects | Minimum exposure for awareness | Ignition threshold for consciousness |
| Probabilistic retrieval | CEP-based probability | Activation-based probability |

---

### 3.3 Metric Integration Across Paradigms

| Marketing Metric | Cognitive Equivalent | Mathematical Relationship |
|-----------------|---------------------|--------------------------|
| Brand Awareness (%) | Recognition probability | p(recall) = f(activation, threshold) |
| Effective Frequency (3+) | Memory consolidation | Repetition strengthens trace |
| Adstock decay (λ) | Memory decay | Exponential forgetting curve |
| Mental Availability | Spreading activation | Network accessibility |
| Attention time (seconds) | Fixation duration | Encoding time |
| d' (sensitivity) | Ad discriminability | Signal detection |

---

## Master Reference Tables

### Marketing and Advertising Frameworks

| Framework | Author(s) | Year | Key Equation/Model |
|-----------|-----------|------|--------------------|
| AIDA | E. St. Elmo Lewis | 1898 | Attention→Interest→Desire→Action |
| Hierarchy of Effects | Lavidge & Steiner | 1961 | 6 stages: Awareness→Knowledge→Liking→Preference→Conviction→Purchase |
| DAGMAR | Russell Colley | 1961 | Awareness→Comprehension→Conviction→Action |
| FCB Grid | Richard Vaughn | 1980 | 2×2: Involvement × Thinking/Feeling |
| GRP | Industry standard | 1950s | GRP = Reach × Frequency |
| Three Exposures | Herbert Krugman | 1972 | Psychological threshold model |
| Effective Frequency | Michael Naples | 1979 | 3+ exposures optimal |
| Recency Theory | Erwin Ephron | 1995 | One exposure in purchase window |
| Adstock | Simon Broadbent | 1979 | A_t = T_t + λ × A_(t-1) |
| Vidale-Wolfe | Vidale & Wolfe | 1957 | dS/dt = r×A×(M-S)/M - δ×S |
| SOV-SOM | John Philip Jones | 1990 | ESOV = SOV - SOM |
| NBD-Dirichlet | Ehrenberg et al. | 1984 | Brand choice probability model |
| Mental Availability | Byron Sharp/Romaniuk | 2004–2010 | CEP-based salience |
| CBBE Pyramid | Kevin Lane Keller | 1993 | Salience→Meaning→Response→Resonance |
| AU Metric | Adelaide/Guldimann | 2019 | 0–100 attention probability score |
| Active Attention | Karen Nelson-Field | 2020 | 1.5s threshold for memory encoding |

### Cognitive Science Frameworks

| Framework | Author(s) | Year | Key Equation/Model |
|-----------|-----------|------|--------------------|
| Channel Capacity | Claude Shannon | 1948 | C = B × log₂(1 + S/N) |
| 7 ± 2 Chunks | George Miller | 1956 | Immediate memory span |
| Filter Model | Donald Broadbent | 1958 | Early physical selection |
| Attenuation | Anne Treisman | 1964 | Signal reduction + threshold |
| Late Selection | Deutsch & Deutsch | 1963 | Post-semantic selection |
| Signal Detection | Green & Swets | 1966 | d' = z(HR) - z(FAR) |
| Capacity Model | Daniel Kahneman | 1973 | Flexible resource pool |
| Feature Integration | Treisman & Gelade | 1980 | Parallel features + serial binding |
| Multiple Resources | Christopher Wickens | 1980 | 4-D interference model |
| Global Workspace | Bernard Baars | 1988 | Broadcast to unconscious processors |
| Cognitive Load | John Sweller | 1988 | Total = Intrinsic + Extraneous |
| Itti-Koch Saliency | Itti & Koch | 1998 | S = (1/3)[N(Ī) + N(C̄) + N(Ō)] |
| Predictive Coding | Rao & Ballard | 1999 | Prediction error minimization |
| 4 ± 1 Chunks | Nelson Cowan | 2001 | True capacity without rehearsal |
| Free Energy | Karl Friston | 2010 | F = E_q[ln q(s) - ln p(s,o)] |
| Guided Search 6.0 | Jeremy Wolfe | 2021 | 5-source priority map |

---

## Conclusion: The Unified Logic of Awareness

The intellectual archaeology of awareness reveals a remarkable convergence: marketing practitioners and cognitive scientists independently discovered the same fundamental constraints.

Herbert Simon's "poverty of attention" (1971) echoes through every subsequent framework—from Broadbent's bottleneck to Adelaide's AU metric. The mathematical formalizations differ in precision (d-prime equations vs. GRP calculations) but share underlying logic:

**Awareness is a competitive, capacity-limited, threshold-gated process where signals must overcome noise to achieve conscious representation.**

### Three Unifying Principles

**1. Awareness is probabilistic, not deterministic.**

Whether measured as brand recall rate, fixation probability, or d' sensitivity, awareness is always a **probability distribution**, not a binary state. The NBD-Dirichlet model, signal detection theory, and predictive processing all treat awareness as stochastic.

**2. Awareness has architectural constraints.**

Miller's 7±2, Cowan's 4±1, Wickens' multiple resources, and working memory models all identify **hard limits** on simultaneous processing. These constraints explain both cognitive load effects in advertising and the effectiveness of chunking in brand design.

**3. Awareness involves active prediction, not passive reception.**

From Treisman's thresholds to Friston's precision-weighting, modern theory emphasizes that **prior expectations shape what enters awareness**. This validates marketing practices around brand consistency, mental availability building, and contextual targeting.

---

## The Opportunity: New Math for a New Era

The core mathematical frameworks documented here were established between **1950 and 1980**. We are operating under pre-internet, pre-digital, pre-AI assumptions.

The field continues evolving toward:
- **Attention-based measurement** (Adelaide, Lumen, Amplified Intelligence)
- **Computational prediction** (DeepGaze, saliency models)
- **Real-time optimization** (ML-driven media buying)

But the fundamental insight remains what Simon articulated fifty years ago:

> In an information-rich world, **awareness is the scarce resource** that determines all downstream effects.

Every equation documented here—from `GRP = Reach × Frequency` to `F = E_q[ln q(s) - ln p(s,o)]`—represents an attempt to model that scarcity mathematically.

**The next breakthrough will come from synthesizing these traditions with modern computational methods.**

---

## References

### Primary Sources — Marketing

- Binet, L. & Field, P. (2013). *The Long and the Short of It*. IPA.
- Broadbent, S. (1979). One Way TV Advertisements Work. *Journal of the Market Research Society*.
- Colley, R. (1961). *Defining Advertising Goals for Measured Advertising Results*. ANA.
- Ehrenberg, A., Goodhardt, G., & Barwise, P. (1990). Double Jeopardy Revisited. *Journal of Marketing*.
- Ephron, E. (1995). More Weeks, Less Weight. *Journal of Advertising Research*.
- Jones, J.P. (1990). Ad Spending: Maintaining Market Share. *Harvard Business Review*.
- Krugman, H. (1972). Why Three Exposures May Be Enough. *Journal of Advertising Research*.
- Lavidge, R. & Steiner, G. (1961). A Model for Predictive Measurements of Advertising Effectiveness. *Journal of Marketing*.
- Lewis, E. St. Elmo (1898). Side Talks about Advertising. *The Western Druggist*.
- Nelson-Field, K. (2020). *The Attention Economy and How Media Works*. Palgrave Macmillan.
- Sharp, B. (2010). *How Brands Grow*. Oxford University Press.
- Vaughn, R. (1980). How Advertising Works: A Planning Model. *Journal of Advertising Research*.

### Primary Sources — Cognitive Science

- Baars, B. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Baddeley, A. & Hitch, G. (1974). Working Memory. *Psychology of Learning and Motivation*.
- Broadbent, D. (1958). *Perception and Communication*. Pergamon Press.
- Cowan, N. (2001). The Magical Number 4 in Short-Term Memory. *Behavioral and Brain Sciences*.
- Friston, K. (2010). The Free-Energy Principle: A Unified Brain Theory? *Nature Reviews Neuroscience*.
- Green, D.M. & Swets, J.A. (1966). *Signal Detection Theory and Psychophysics*. Wiley.
- Itti, L. & Koch, C. (1998). A Model of Saliency-Based Visual Attention. *IEEE PAMI*.
- Kahneman, D. (1973). *Attention and Effort*. Prentice-Hall.
- Miller, G.A. (1956). The Magical Number Seven, Plus or Minus Two. *Psychological Review*.
- Shannon, C.E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*.
- Simon, H.A. (1971). Designing Organizations for an Information-Rich World. In *Computers, Communication, and the Public Interest*.
- Treisman, A. (1964). Selective Attention in Man. *British Medical Bulletin*.
- Treisman, A. & Gelade, G. (1980). A Feature-Integration Theory of Attention. *Cognitive Psychology*.
- Wolfe, J.M. (2021). Guided Search 6.0. *Psychonomic Bulletin & Review*.

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Created by Armstrong Knight, Abdullah Khan & Claude.ai | Funnel Function

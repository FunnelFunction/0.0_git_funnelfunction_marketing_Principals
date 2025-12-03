# 0.2 The Sales Funnel

**The 3IR Model: A Reference Architecture**

This section documents the traditional sales funnel—not as best practice, but as **historical reference**. Understanding why it fails is prerequisite to building what replaces it.

---

## The Funnel Equation

Traditional funnel conversion follows exponential decay:

```
Conversion = L₀ · e^{-λn}
```

Where:
- **L₀** = Initial leads
- **λ** = Loss rate per stage (~0.8 typically)
- **n** = Number of stages

### Example

```
Awareness:  1000 leads
Interest:    200 leads (80% loss)
Decision:     40 leads (80% loss)
Action:        8 leads (80% loss)
```

**Conversion rate: 0.8%**

This is exponential decay **by design**.

---

## The Funnel Structure

```
0.2_The_Sales_Funnel/
├── 0.2.a_Top_of_the_Funnel_TOF/
│   ├── 0.2.a.i_Awareness/
│   └── 0.2.a.ii_Lead_Generation/
├── 0.2.b_Middle_of_the_Funnel_MOF/
│   ├── 0.2.b.i_Nurturing/
│   └── 0.2.b.ii_Qualification/
└── 0.2.c_Bottom_of_the_Funnel_BOF/
    ├── 0.2.c.i_Conversion/
    └── 0.2.c.ii_Close/
```

---

## Why Document a Dead Model?

1. **Historical Context** — Most businesses still operate this way
2. **Comparison Baseline** — Measure 4IR gains against 3IR baseline
3. **Migration Path** — Understand what you're replacing
4. **Client Education** — Help clients see why change is necessary

---

## The Core Problem

The funnel assumes:
- Loss is normal and expected
- Humans must manage each stage
- Volume compensates for inefficiency
- Linear progression is the only model

The funnel **does not ask** why leads are lost—it accepts loss as inherent.

---

## Funnel vs Field Model

| Funnel (3IR) | Field (4IR) |
|--------------|-------------|
| Linear stages | Recursive loops |
| Manage dropout | Manage writability |
| MQLs → SQLs → Customers | Single Store of truth |
| Humans qualify at each gate | AI pre-filters writables |
| Loss is expected | Loss means bad targeting |
| Volume strategy | Precision strategy |

---

## Contents

| Section | Status | Description |
|---------|--------|-------------|
| [0.2.a TOF](./0.2.a_Top_of_the_Funnel_TOF/) | 🔴 Planned | Awareness & Lead Gen |
| [0.2.b MOF](./0.2.b_Middle_of_the_Funnel_MOF/) | 🔴 Planned | Nurturing & Qualification |
| [0.2.c BOF](./0.2.c_Bottom_of_the_Funnel_BOF/) | 🔴 Planned | Conversion & Close |

---

## The Funnel is Dead

See: [0.3 Non-Funnel Models](../0.3_Non_Funnel_Models/) for what replaces it.

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Created by Armstrong Knight & Abdullah Khan

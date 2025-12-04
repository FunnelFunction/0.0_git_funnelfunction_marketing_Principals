# 0.2 f(The_Sales_Funnel)

**The 3IR Model: A Reference Architecture**

This section documents the traditional sales funnel—not as best practice, but as **historical reference**. Understanding why it fails is prerequisite to building what replaces it.

---

## f(Funnel_Decay)

Traditional funnel conversion follows exponential decay:

```
f(Conversion) = L₀ · e^{-λn}
```

Where:
- **L₀** = Initial leads
- **λ** = Loss rate per stage (~0.8 typically)
- **n** = Number of stages

### Example

```
f(Awareness):  1000 leads
f(Interest):    200 leads (80% loss)
f(Decision):     40 leads (80% loss)
f(Action):        8 leads (80% loss)
```

**Conversion rate: 0.8%**

This is exponential decay **by design**.

---

## The Funnel Structure

```
0.2_f(The_Sales_Funnel)/
├── 0.2.a_f(Top_of_Funnel)/
│   ├── 0.2.a.i_f(Awareness)/
│   └── 0.2.a.ii_f(Lead_Generation)/
├── 0.2.b_f(Middle_of_Funnel)/
│   ├── 0.2.b.i_f(Nurturing)/
│   └── 0.2.b.ii_f(Qualification)/
└── 0.2.c_f(Bottom_of_Funnel)/
    ├── 0.2.c.i_f(Conversion)/
    └── 0.2.c.ii_f(Close)/
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

## f(Funnel) vs f(Field)

| f(Funnel) — 3IR | f(Field) — 4IR |
|-----------------|----------------|
| Linear stages | Recursive loops |
| Manage dropout | Manage writability |
| MQLs → SQLs → Customers | Single Store of truth |
| Humans qualify at each gate | AI pre-filters writables |
| Loss is expected | Loss means bad targeting |
| Volume strategy | Precision strategy |

---

## The Funnel is Dead

See: [0.3 f(Non_Funnel_Models)](../0.3_f(Non_Funnel_Models)/) for what replaces it.

---

## License

Creative Commons Attribution-NonCommercial 4.0 International License

Created by Armstrong Knight & Abdullah Khan

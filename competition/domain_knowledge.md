# Domain Knowledge: Geosteering and TVT Prediction

This file is for Claude Code and Codex workers to read before proposing features.

## What Is Geosteering?

When drilling a horizontal oil/gas well, the drill bit travels 2–5 km laterally
through layered rock. The goal is to keep the bit inside the productive
reservoir layer (usually sandstone). A geologist continuously steers the
bit based on real-time sensor readings.

## What Is TVT (True Vertical Thickness)?

TVT represents the well's position within the geological layer stack.
If TVT = 50 ft, the well is 50 ft below the top of the target layer.

TVT is currently estimated manually by an expert geologist using:
1. The lateral well's Gamma Ray (GR) log
2. A Typewell (nearby vertical reference well) with known geology
3. Pattern matching between the two GR logs

The competition asks us to automate this estimation.

## The Gamma Ray (GR) Log

- Measures natural radioactivity of the rock
- High GR (>100 API) = shale (cap rock, not productive)
- Low GR (<50 API) = sandstone / clean reservoir (productive!)
- Medium GR = siltstone or mixed

The GR log is the PRIMARY signal for TVT estimation.

## The Typewell

- A vertical well drilled near the horizontal well
- Has a complete GR log from surface to total depth
- Has known TVT at each depth (the "ground truth template")
- The horizontal well's GR should match the Typewell GR pattern
  at the depth corresponding to its TVT

## How TVT Is Estimated (The Key Insight)

```
Typewell GR (depth axis = TVT):  [--high--][--low--][--high--][--low--]
                                      shale   sand    shale   sand

Lateral well GR (depth axis = MD): [--high--][--low--][--high--][--low--]
                                       ^         ^
                                       |         |
                                  The GR patterns match!
                                  The shift between them = TVT estimate
```

The mathematical operation is:
1. Take a window of lateral GR values
2. Slide it along the Typewell GR at different lag offsets
3. Find the lag offset that gives the best correlation
4. That lag = TVT at this depth

## Physical Constraints

1. **Smoothness**: TVT changes slowly (geology doesn't jump). A sudden 20 ft
   change in TVT would mean the well drilled through a fault — rare.

2. **Monotonic tendency**: In a horizontal well drilled updip or downdip,
   TVT may trend systematically (structural dip effect).

3. **Bounded range**: TVT should stay within the thickness of the target
   formation, typically 0–200 ft. Outliers suggest a correlation error.

4. **No future leakage**: You cannot use GR values from deeper in the well
   to predict TVT at a shallower point during drilling. In CV, enforce this
   with GroupKFold.

## Most Important Features (Priority Order)

1. **Typewell cross-correlation / DTW distance** at various lag offsets
   — this is the geologist's actual workflow, automated
2. **GR rolling statistics** (mean, std) at multiple windows
3. **GR gradient** — rate of change detects layer boundaries
4. **TVD and depth position** — structural context
5. **Dog-leg severity** — high DLS means passing through a boundary
6. **GR percentile rank within well** — normalizes for well-to-well variation

## Common Pitfalls

- **Data leakage**: Never use knowledge of TVT from adjacent rows in the
  same well during CV. GroupKFold by well_id is mandatory.
- **Cross-well contamination**: Do not engineer features that blend
  information from different wells.
- **Typewell misalignment**: The Typewell may have a depth offset from the
  lateral well. Always search for the best alignment before correlation.
- **GR normalization**: Different wells may have different GR ranges due to
  tool calibration. Normalize GR within each well before Typewell matching.

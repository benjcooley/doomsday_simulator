# SETTLING ENERGY CONSERVATION BUG

Status: **root cause proven analytically; fix implemented; live browser still erupts → OPEN.**

## Symptom

At particle counts above ~40–60k, a *tiny* impactor (1 km **or** 10 km — identical result) hitting
an at-rest Earth produces a **planet-wide thermal detonation**: a shockwave crosses the whole globe
and eruptions appear everywhere, not just at the impact site. The energy readout shows absurd
values (millions × Chicxulub). Tuning material **damping / cohesion does nothing**. Below ~24k the
same impact is a clean local crater.

## TL;DR of the mechanism (verified, see "Headless proof" below)

The impact does **not** supply the energy. It flips a global switch that lets the planet release
energy it was already holding:

- A scenario spawns and is held still by `settleDrag` (a velocity-damping pull toward the rigid
  reference) with heating suppressed (`heatGate = 0`). It is then **frozen** (sim paused).
- The spawn lattice is **not** at true gravitational equilibrium — it wants to contract. `settleDrag`
  kills the *velocity* of that contraction, but the planet is frozen **before the positions finish
  settling**, so a positional gap (latent potential energy) is pinned in place.
- On first contact, `_heatArmed` flips and **two global flags change at once**: `settleDrag → 0`
  (rigid hold released, planet-wide) and `heatGate → 1` (heating live, planet-wide).
- The pinned positional gap now collapses to equilibrium; the leftover motion is converted to heat
  by the `dqc` (spring-compression) term that just turned on — everywhere simultaneously. That is
  the detonation. Size-independent because it is the **planet's own settling energy**, not the
  impactor's.

This is why it is size-independent, damping-independent, and reads as a fixed huge number.

## Key code locations

- `js/sim.js` — the global flags:
  - `heatGate`: `0` while settling, `1` once `_heatArmed` (any contact). Gates the `dqc` heat term.
  - `settleDrag`: `1/60` while settling (damps every particle toward its rigid reference), `0` once armed.
  - Freeze gate: a scenario LOADS frozen; it relaxes "later", held rigid by settle-drag.
- `js/shaders_sim.js` `pairPhysicsWGSL` (shared by BOTH the N² and fast engines):
  - dt-stability clamps: `kMax = 0.20·mPair/dt²`, `cMax = 0.08·mPair/dt`, cohesion cap `kMax·0.5`.
  - `dqc` shock/frictional heat = `0.3·|f|·(-vn)·… · heatGate` — **gated by heatGate**.
  - The **dashpot** (`fdamp = cp·vn`) only *removes velocity*; it does NOT generate heat. "Frictional
    heat" is the separate, gated `dqc` term. So a cool settle dissipates motion **without** heating —
    intentional, so a settling Earth reaches 288 K instead of glowing.
  - shock-softening: now gated to `heatGate > 0.5` only (full damping while cool, so a settle
    converges instead of ringing).

## Why material damping/cohesion "do nothing" (also verified)

When `dt ≫ dtStable`, every contact saturates the dt-stability clamps, so the per-material `k`,
`cohF`, `dampZ` never reach the contact — the clamp value does. The real high-N parameters are the
clamp coefficients (`0.20`, `0.08`), and they are pinned by explicit-integrator stability (cMax can't
exceed ~0.25·mPair/dt). Doubling `dampZ` changed the measured spike by <2% (see table).

NOTE: the dt floor is **not** the trigger at ~60k. At 58k, `dtStable≈0.65`, `dt=5·dtStable≈3.26`
(not floored at 0.3). The clamp ratio `k/kMax=(dt/dtStable)²/20≈1.25` is the same at 24k and 63k, so
clamp-domination is NOT what separates "settles" from "churns". The separator is **settle time** and
**packComp** (below).

## Headless proof — `test/arm_spike.mjs`

Builds the REAL layered Earth via `buildBlob`, wires the shipping FAST engine with the exact material
constants `particles.js` derives, runs the arming cliff **with no impactor at all**, so any thermal
rise is purely the planet releasing its own held energy. Phases:
- Phase S (settle): `heatGate=0`, `settleDrag=1/60` — "come to rest without heating".
- Phase A (arm): `heatGate=1`, `settleDrag=0` — exactly what first contact flips.

Run: `node test/arm_spike.mjs` (env: `SETTLE=`, `ARM=`, `RELAX=`, `CASES=`). Uses the `webgpu`
(Dawn-Node) package already in `test/node_modules`.

### Measured results (ΔthermalJ in × Chicxulub = 4.2e23 J)

| case | rested rms | armed rms | armed Tp99 | ΔthermalJ |
|---|---|---|---|---|
| **24k baseline** | **0.002** | 0.003 | 1928 → **1928** | **0** |
| 63k baseline (packComp 0.10) | 0.78 (churns) | 1.19 | → 11972 | 2.92e7 |
| 63k + 2× damping | 0.80 | 1.23 | → 11944 | **2.97e7** (≈ no change) |
| 63k + "cool relax" (settleDrag=0 first) | 1.81 | 1.34 | → 21155 | 7.05e7 (**worse**) |
| 63k packComp=0 | 0.030 | 0.79 | → 6605 | 1.81e7 |
| 63k packComp=0.05 | 0.016 | 0.90 | → 6736 | 1.70e7 |
| 63k pc0 **settle 1000** | 0.004 | 1.00 | → 7510 | 2.02e7 |
| 63k pc0 settle 1400 | 0.015 | 0.95 | → 6929 | 1.97e7 |
| 63k pc0 **settle 1800** | **0.000** | **0.000** | 1935 → **1935** | **0** |
| 63k pc0 settle 2000 | 0.000 | 0.000 | → 1935 | **0** |

### What the table proves

1. **24k settles to a dead stop and arming injects literally zero heat.** Below threshold there is no bug.
2. **A fully-settled 63k planet (settle ≥1800) also injects ZERO heat on arming.** The whole melt is
   the planet being frozen *before settling finished*.
3. **rms is NOT a reliable "settled" signal**: at settle=1000 rms is already 0.004 (looks done) but
   still cooks — settle-drag zeroes *velocity* while *positions* are still contracting. Must exit on
   **structural** convergence (`√(Iw/mass)` stops shrinking), not rms.
4. **`packComp=0.10` destabilizes the small-particle lattice** (rms 0.77, never settles). 0–0.05 is fine.
5. **Damping is genuinely ignored** at high N (2× = +1.7%).
6. **Releasing settle-drag during relax (=0) makes it worse**, not better — the hold is what dissipates.

## Fix implemented (this commit)

- `js/blob.js`: `PHYS.packComp` 0.10 → **0** (the pre-load was destabilizing at small particle size).
- `js/sim.js` auto-relax at scenario load (`this.relaxing`): runs the planet **live + cool**
  (`heatGate=0`, `settleDrag=1/60`) and **freezes only after structural convergence** —
  `√(Iw/mass)` change < 8e-5 — with a sim-time floor (400) and hard cap (12000). So normal play
  freezes the *settled* state, not the raw spawn.
- `js/sim.js`: `settleDrag` stays `1/60` during relax/settle (only a real impact frees it). The
  earlier `settleDrag=0` "cool relax" was the churn.
- `js/shaders_sim.js`: shock-softening gated to armed-heat only (`select(1.0, softened, heatGate>0.5)`)
  so a cool settle has full damping and converges.
- `js/sim.js`: manual **Settle** diagnostic (SYS panel) + heat-off settle; `Apply`/`Settle` buttons.

## OPEN — live browser still erupts

After a hard reload, a 10 km impactor at ~63k **still** produced a planet-wide eruption (screenshot).
So the headless-proven fix (settle to structural convergence → 0 spike) is **not yet being achieved
by the live `sim.js` relax**. Prime suspects to investigate next:

1. The relax convergence-exit (`8e-5` on `√(Iw/mass)`) may be firing too early, or `Iw` may not be
   populated on `statsCache.bodies[0]` in the live path the way the test assumes — verify the live
   relax actually runs ~1800+ substeps and reaches rms≈0 before freezing (add a one-shot log of
   `simTime`, substep count, and `√(Iw/mass)` at the moment `relaxing` flips false).
2. The want-boost (`liveCap·dtSubMax`) may not advance enough sim-time per frame at 63k, so the relax
   hits the readback cadence / cap before converging.
3. Confirm the relax is even engaging (not short-circuited by the freeze gate / `nearWake` / an
   impactor already close at load).
4. Consider mirroring the convergence-exit logic into `arm_spike.mjs` (a `relax-to-convergence` then
   arm variant) to confirm the EXACT exit rule yields 0 before trusting it live.

## Discarded / dead-end approaches (do not retry)

- **Spatial heat gate** (heat only near impact): would mask, not fix; needs a risky params-buffer
  uniform addition. Unnecessary once the planet truly settles.
- **Ramp settleDrag to 0 globally on arm**: conflicts with the local impact (damps the crater too).
- **Volume fudge** (spawn radius): both directions are traps (+ enlarges the held collapse, − is
  pre-compression that stores explosive energy). Replaced by build-radius scaling for *size* only.
- **Tuning material damping/cohesion**: clamped/ignored above ~40k.
- **Bug #1** (armed-blast deposit not normalized by receiver count): real but ~2× at 1M and scales
  with impactor size; the identical 1 km/10 km result rules it out as the cause here. Worth fixing
  someday for correctness, not for this bug.

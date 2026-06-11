# HANDOFF — read this first

## The actual problem (user-reported, June 2026)
**Impactors do not fall toward or intercept Earth.** The collision — the entire
point of the app — is not visibly happening. Everything else (heat, ocean
painting, surface-temperature tuning, bloom, settle phases) was polished *on top
of a sim where the event never occurs*. That was the mistake. Fix the collision
first; ignore all the rest until a basic smash looks right.

## The north-star test (do this before anything else)
> Earth + one impactor. Smash them together. Check that the result is realistic.

Build the dumbest possible version of this and get it correct with your own eyes,
**stripped of all the cinematic machinery** (auto-slow, freeze/ride-along,
stale-readback mirror sync, frame rebase, dissolve handoff). Those adaptive
layers were added before the core collision was ever verified, and any one of
them can be why the impactor doesn't visibly intercept.

Suggested approach:
1. Add a trivial debug scenario: Earth blob at origin, one rock impactor a short
   distance away on a **dead-simple head-on velocity straight at [0,0,0]**, fixed
   small timestep, **no** auto-slow / no freeze / no warp. Just integrate and watch.
2. Confirm in the particle sim (not just the CPU "mirror") that the impactor
   particles actually close the distance and contact the Earth particles.
3. Only once that looks physically right, re-introduce the cinematic layers one at
   a time, checking the smash still happens after each.

## Where the bug likely lives
- `js/sim.js` — `_ctx().approach(dir, dist, speed, b)` builds the impactor's
  spawn pos/vel. Verify the velocity vector genuinely points at Earth and the
  speed (Mm/s) is right. Check `spawnImpactor()` and the armed-vs-blob branch.
- `js/sim.js` `tick()` — the CPU `mirror` integrates Earth/Moon/impactor under
  sun + mutual gravity. **Suspect:** sun gravity or the anchor/rebase frame is
  dominating, pulling bodies off the Earth-intercept course. The user separately
  saw "bodies fall into the sun."
- The `frozen` ride-along + `disturbed`/stale-readback sync in `tick()` /
  `applyStats()` can move particles independently of the visible mirror. If the
  mirror says "approaching" but particles are frozen/desynced, you see no impact.
- `js/particles.js` `step()` ping-pong + `addArmedImpactor` (sub-resolution
  impactors deposit energy as a payload rather than as a real colliding body —
  make sure the test uses a *real blob* impactor so you can SEE it hit).

## What genuinely works (verified in screenshots — don't re-fix)
Textured Earth globe (continents/clouds/atmosphere/city lights/ocean glint),
full Kepler solar system + sun + belt + stars, Moon orbiting at the correct
377 Mm / 1.0 km/s, the whole UI (sidebar, time dial 0.02×–1mo/s, HUD, labels),
bloom/HDR pipeline, boot + WebGPU init. The rendering is fine. The *dynamics of
the collision* are the problem.

## Debug helpers already in the code
- `?hud=0` — hide UI for clean captures
- `window.__traj()` — console.table of every body's pos/vel/dist-to-Earth
- `window.__state()` — full sim snapshot (warp, fps, contacts, per-material temps)
- `window.__film(shots, hoursStep, scale)` — filmstrip grid of the sim over time
- `window.__sim` — the live Sim instance

## Meta-lesson for the next session
Verify the headline behavior end-to-end **before** tuning anything underneath it.
Work in short bursts and hand control back; don't disappear into long autonomous
fix-chains. The user must be able to steer.

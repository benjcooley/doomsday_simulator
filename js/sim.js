// sim.js — simulation orchestrator: scenario lifecycle, CPU mirror, time warp, frame anchor,
// dissolve handoff, population/temperature/mass model, headlines.
import { dateToJD, gmst, planetPos, planetVel, moonGeo, moonGeoVel, orbitPolyline, earthPoleEcliptic, G_SIM, GM_SUN, AU, PLANET_NAMES } from './orbits.js';
import { CATALOG, IMPACTORS, impactorMass, POP_2026, RECIPES } from './bodies.js';
import { buildBlob } from './blob.js';
import { getScenario } from './scenarios.js';
import { FILLER } from './headlines.js';
import { vadd, vsub, vscale, vnorm, vcross, vlen, vdot, clamp, lerp, curve } from './mathx.js';

const J2000 = 2451545.0;
const EARTH_OMEGA = 7.2921159e-5;
const EMPTY_F32 = new Float32Array(0);
const CHICX_J = 4.2e23;

export class Sim {
  constructor(gpu, ps, renderer, texData, quality) {
    this.gpu = gpu;
    this.ps = ps;
    this.renderer = renderer;
    this.texData = texData;                 // {earthDay, moon, mars, jupiter} ImageData
    this.quality = quality;                  // total particle budget
    this.warpUser = 60;
    this.paused = false;
    this.autoSlow = true;
    // Tunable sim knobs (set from the SYSTEM panel; relativistic/approach dials read live each
    // tick, so they take effect without a rebuild — handy for wiggling the Lance into behaving).
    //   wakeR    — particle SIM wakes when surface gap < wakeR · ΣR
    //   fineR    — within fineR · ΣR a frozen impactor switches to small fixed steps
    //   fineFrac — max frozen advance per frame, as a fraction of ΣR (tunnel-proofing)
    //   slowWin  — relativistic slow-mo engages within this many seconds of contact
    //   slowDiv  — slow-mo strength: target warp ≈ time-to-contact / slowDiv (lower = slower)
    this.sys = { wakeR: 3, fineR: 14, fineFrac: 0.4, slowWin: 120, slowDiv: 5 };
    this.settleMode = false;   // SYS-panel "Settle" diagnostic: run the planet live, shells off
    this.relaxing = false;     // auto relax-to-equilibrium phase at scenario load
    this.view = {
      clouds: true, atmo: true, cityLights: true, orbits: true, trails: true,
      labels: true, belt: true, forceParticles: false, autoFrame: true, ejecta: true,
      bloom: 0.18, exposure: 1.0, coolMul: 2500, heatMul: 1.0, starBoost: 0.7,
    };
    this.maxSub = 10;          // adaptive substep budget (performance governor)
    this.warpSmooth = 60;
    this.statsCache = null;
    this.frame = 0;
    this.fps = 60;
    this.mirror = [];
    this.scenarioId = 'peaceful';
    this.headlineQueue = [];
    this.pop = POP_2026;
    this.killTarget = 0;
  }

  // ---------- scenario lifecycle ----------
  loadScenario(id, dateOverride) {
    const def = getScenario(id) || getScenario('peaceful');
    this.def = def;
    this.scenarioId = def.id;
    const dateStr = dateOverride || def.date;
    this.jd0 = dateStr ? dateToJD(new Date(dateStr + (dateStr.includes('Z') || dateStr.includes('+') ? '' : 'Z'))) : dateToJD(new Date());
    if (!isFinite(this.jd0)) this.jd0 = dateToJD(new Date());
    this.simTime = 0;
    this.warpUser = def.warp0 || 60;
    this.warpSmooth = this.warpUser;
    this.warpEff = this.warpUser;
    // impactor trail frame, selectable per scenario — set BEFORE def.build() spawns anything.
    // Default 'geo' (target/Earth-relative): it's Earth's gravity well they animate against.
    this.impactorTrailFrame = def.impactorTrails || 'geo';
    this.paused = false;
    this.ps.reset();
    this.mirror = [];
    this.contacts = new Set();
    this.consumedMask = 0;
    this._impactHold = null;
    this.events = (def.headlines || []).map((h) => ({ ...h, fired: false }));
    this._fillerSeen = new Set();
    this._fillerT = 0;
    this._approachFired = false;
    this.cumImpactJ = 0;
    this.thermalJ0 = null;
    this.pop = POP_2026;
    this.killTarget = 0;
    this.dissolve = 0;
    this.dissolveGo = false;
    this.statsCache = null;
    this.statsBase = null;
    this.frozen = false;
    this.mergedNames = [];
    this._wake = false;
    this.kDust = 0;
    this.atmE = 0;
    this.atmT = 288;
    this._lastCumJ = 0;
    this.vClampNeed = 1;
    this._heatArmed = false;
    this._firstContactT = null;   // else next scenario's 'after:' headlines fire instantly
    this._burstV = null;
    this._gateWhy = '';
    this.banner = null;
    this.headlineQueue = [];
    this.settleUntil = 1800;
    this.renderer.clearTrails();
    this.renderer.clearGhosts();
    this.ejecta?.reset();
    this.renderer.writeAim(new Float32Array(0), 0);
    this.trails = [];
    this.settleUntil = 3600;
    this.relaxing = true;          // cool relax-to-equilibrium before the first freeze (no latent energy)
    this._relaxR = 0;              // structural-radius tracker for relax convergence
    this.frozen = false;
    // per-scenario view overrides (e.g. smash test hides orbit/belt clutter for a clean look)
    if (def.view) Object.assign(this.view, def.view);

    // frame anchor = Earth's heliocentric state at jd0
    const ep = planetPos('earth', this.jd0);
    const ev = planetVel('earth', this.jd0);
    const avs = def.anchorVelScale !== undefined ? def.anchorVelScale : 1;
    this.anchor = { pos: ep.slice(), vel: vscale(ev, avs) };

    // --- Earth blob ---
    const earthCount = Math.round(this.quality * 0.60);
    const th = gmst(this.jd0);
    const eb = buildBlob('earth', CATALOG.earth.R, CATALOG.earth.M, earthCount, {
      textures: { earthDay: this.texData.earthDay },
      pole: earthPoleEcliptic(), theta: th, spinRadPerSec: EARTH_OMEGA,
    });
    this.earthBody = this.ps.addBlob(eb, [0, 0, 0], [0, 0, 0], 'Earth');
    this._addMirror('Earth', CATALOG.earth.M, CATALOG.earth.R, [0, 0, 0], [0, 0, 0], this.earthBody.slot, [0.4, 0.7, 1]);
    this.mirror[0].spin = vscale(earthPoleEcliptic(), EARTH_OMEGA);

    // --- Moon ---
    const moonMode = def.moon || 'normal';
    if (moonMode !== 'none') {
      const mp = moonGeo(this.jd0);
      let mv = moonGeoVel(this.jd0);
      let moonCanonical = true;        // real lunar orbit unless the scenario alters its velocity
      if (moonMode === 'stopped') { mv = [0, 0, 0]; moonCanonical = false; }
      if (typeof moonMode === 'object' && moonMode.velScale !== undefined) { mv = vscale(mv, moonMode.velScale); moonCanonical = false; }
      const mb = buildBlob('moon', CATALOG.moon.R, CATALOG.moon.M, Math.round(this.quality * 0.13), {
        textures: { moon: this.texData.moon }, pole: earthPoleEcliptic(), theta: 0,
      });
      this.ps.addBlob(mb, mp, mv, 'Moon');
      this._addMirror('Moon', CATALOG.moon.M, CATALOG.moon.R, mp, mv, 1, [0.8, 0.8, 0.85], moonCanonical);
      const mm = this.mirror[this.mirror.length - 1];
      mm.shellTex = '2k_moon.jpg'; mm.shellFade = 1;
    }

    if (def.build) def.build(this._ctx());

    // capture each body's original orbit (fixed reference paths — deviation is drawn against these)
    this._captureOriginalOrbits();

    // SEED each mirror body's motion trail with its original orbit: the orbit IS the trail's
    // starting history, and live movement appends to it — so the body's real path draws
    // continuously out of its predicted ellipse, and any deviation peels away visibly.
    for (const b of this.mirror) {
      let arc = null;
      if (b.slot === 0) arc = this.origArcs.find((a) => a.kind === 'earth');
      else if (b.canonical) arc = this.origArcs.find((a) => a.kind === 'moon' && a.slot === b.slot);
      if (!arc) continue;
      // trails live in HELIOCENTRIC space (same frame as the planet ghost arcs). Recording
      // them in the Earth-co-moving local frame made every trail a position-relative-to-
      // Earth's-frame-at-record-time: at high warp the seeded year-ellipse rode rigidly
      // around the Sun with Earth — the "solar system orbits Earth" illusion.
      const M = arc.M;
      const seed = [];
      const k0 = Math.max(1, Math.ceil(M * 0.15));   // seed the trailing ~85% of one period —
      for (let k = k0; k < M; k++) {                 // a visible gap; the trail never wraps closed
        const px = arc.pts[k * 3], py = arc.pts[k * 3 + 1], pz = arc.pts[k * 3 + 2];
        // helio bodies (Earth) seed heliocentric; geo bodies (Moon) seed in Earth-relative
        // coordinates — matching their per-body trailFrame
        seed.push([px, py, pz]);
      }
      let circ = 0;
      for (let k = 1; k < seed.length; k++) circ += Math.hypot(seed[k][0] - seed[k - 1][0], seed[k][1] - seed[k - 1][1], seed[k][2] - seed[k - 1][2]);
      b.trail = seed;
      b.trailStep = Math.max(b.R * 0.5, circ / Math.max(seed.length, 1) * 0.7);  // live appends continue arc density
      b.trailMax = Math.min(500, Math.ceil(seed.length * 1.1));  // length cap ≈ the seeded span:
      b.dirty = true;                                            // stays shorter than one period
    }

    this.focusId = 'frame';   // frame mode is the default camera — it follows the action
    this.scenarioFocus = def.focus || 'earth';   // scenario hint, available in the dropdown
    this.camHint = { dist: def.camDist || 40 };
    this.statusText = 'SIMULATION READY';
  }

  _addMirror(name, M, R, pos, vel, slot, color, canonical = false) {
    this.mirror.push({
      name, M, R, pos: pos.slice(), vel: vel.slice(), slot, color, trail: [], trailAcc: 0,
      freezePos: pos.slice(), freezeVel: vel.slice(),
      canonical,   // on a real orbit? ghost predicted-orbit is only drawn for canonical bodies
      // trail frame: Earth's trail is heliocentric (its fixed ellipse in space); EVERYTHING
      // else — Moon, impactors — records target-relative (Earth-relative): the Moon's loop
      // travels with Earth and an impactor's trail is its approach path toward the target.
      trailFrame: slot === 0 ? 'helio' : 'geo',
    });
    if (this.frozen) this._wake = true;   // a new body always wakes the mechanics
  }

  _ctx() {
    const sim = this;
    return {
      jd0: this.jd0,
      LD: 384.4,
      dirFromAngles(azDeg, elDeg) {
        const az = azDeg * Math.PI / 180, el = elDeg * Math.PI / 180;
        return [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
      },
      // Earth's orbital direction of travel (unit, local-frame axes) — place things
      // "on our forward path" with this instead of guessing absolute azimuths
      prograde() {
        const v = sim.anchor.vel;
        return vlen(v) > 1e-9 ? vnorm(v) : [1, 0, 0];
      },
      // predict a body's position t seconds ahead (sun + mutual gravity, coarse steps) —
      // Earth wobbles ~20 Mm per Pluto-flight from the Moon's pull; dead reckoning misses
      predict(name, t) {
        const bodies = sim.mirror.map((b) => ({ p: b.pos.slice(), v: b.vel.slice(), M: b.M }));
        const G = G_SIM;
        const n = Math.max(8, Math.ceil(t / 600));
        const dt = t / n;
        for (let s = 0; s < n; s++) {
          const sunL = vscale(sim.anchor.pos, -1);
          const rA = Math.max(vlen(sunL), 1);
          const aA = vscale(sunL, GM_SUN / (rA * rA * rA));
          for (const b of bodies) {
            const ds = vsub(sunL, b.p);
            const ir = 1 / Math.max(vlen(ds), 1);
            let acc = vsub(vscale(ds, GM_SUN * ir * ir * ir), aA);
            for (const o of bodies) {
              if (o === b) continue;
              const d = vsub(o.p, b.p);
              const rr = Math.max(vlen(d), 1);
              acc = vadd(acc, vscale(d, G * o.M / (rr * rr * rr)));
            }
            b.v = vadd(b.v, vscale(acc, dt));
          }
          for (const b of bodies) b.p = vadd(b.p, vscale(b.v, dt));
        }
        const idx = sim.mirror.findIndex((b) => b.name === name);
        return bodies[Math.max(0, idx)].p;
      },
      approach(dir, dist, speed, b, targetName) {
        const pos = vscale(dir, dist);
        let perp = vcross(dir, [0, 0, 1]);
        if (vlen(perp) < 1e-6) perp = vcross(dir, [1, 0, 0]);
        perp = vnorm(perp);
        const tgt = targetName || 'Earth';
        const tof = dist / Math.max(speed, 1e-6);
        const wantOff = vscale(perp, b || 0);
        // ballistic solver: the sun's tidal field bends long-range shots by whole Earth-radii,
        // so shoot a test round, measure the miss at closest approach, correct, iterate
        let aim = vadd(this.predict(tgt, tof), wantOff);
        let vel = vscale(vnorm(vsub(aim, pos)), speed);
        for (let it = 0; it < 3; it++) {
          const closest = sim._shootMiss(pos, vel, tgt, tof);
          if (!closest) break;
          const err = vsub(closest, wantOff);
          if (vlen(err) < 0.5) break;
          aim = vsub(aim, err);
          vel = vscale(vnorm(vsub(aim, pos)), speed);
        }
        return { pos, vel };
      },
      spawnImpactor(opts) { return sim.spawnImpactor(opts); },
      launchAtLatLon(opts) { return sim.launchAtLatLon(opts); },
    };
  }

  spawnImpactor({ recipe, d_km, pos, vel, name, countScale = 1 }) {
    const M = impactorMass(d_km, recipe);
    const R = d_km / 2000;
    const mpEarth = CATALOG.earth.M / this.earthBody.count;
    const nProp = (M / mpEarth) * 1.15 * countScale;
    const reserve = Math.max(0, this.qualityMax() - this.ps.activeN);
    let body;
    if (R >= 0.22 && nProp >= 18 && reserve > 30) {
      const count = clamp(Math.round(nProp), 24, reserve);
      const texKey = RECIPES[recipe] && RECIPES[recipe].layers.some((l) => l.paint) ? RECIPES[recipe].layers.find((l) => l.paint).paint : null;
      const blob = buildBlob(recipe, R, M, count, { textures: this.texData, pole: [0, 0, 1], theta: 0 });
      body = this.ps.addBlob(blob, pos, vel, name);
    } else {
      const E = 0.5 * M * vdot(vel, vel);   // sim units (1e36 J)
      body = this.ps.addArmedImpactor(M, E, pos, vel, Math.max(R, 0.045), name);
    }
    this._addMirror(name, M, Math.max(R, 0.05), pos, vel, body.slot, recipe === 'comet' ? [0.5, 0.9, 1] : [1, 0.6, 0.3]);
    // impactor trail frame is selectable PER SCENARIO (impactorTrails: 'geo'|'helio');
    // default 'geo' = target(Earth)-relative — the readable approach path
    this.mirror[this.mirror.length - 1].trailFrame = this.impactorTrailFrame || 'geo';
    this.vClampNeed = Math.max(this.vClampNeed || 1, vlen(vel) * 1.3);   // relativistic impactors
    const SHELLS = {
      mars: '2k_mars.jpg', jupiter: '2k_jupiter.jpg', earth2: '2k_earth_daymap.jpg',
      moon: '2k_moon.jpg', venus: '2k_venus_atmosphere.jpg', pluto: '2k_pluto.jpg',
    };
    const shellTex = arguments[0].shellTex || SHELLS[recipe];
    if (body.count > 1 && shellTex) {
      const mb = this.mirror[this.mirror.length - 1];
      mb.shellTex = shellTex; mb.shellFade = 1;
    }
    this.settleUntil = Math.max(this.settleUntil, this.simTime + 1200);
    return body;
  }

  launchAtLatLon({ recipe, d_km, lat, lon, speed, entryDeg, azDeg, range, name }) {
    const v = this._latLonLaunch(lat, lon, speed, entryDeg, azDeg, range);
    return this.spawnImpactor({ recipe, d_km, pos: v.pos, vel: v.vel, name });
  }

  _latLonLaunch(lat, lon, speed, entryDeg, azDeg, range) {
    const th = gmst(this.jd0 + this.simTime / 86400);
    const pole = earthPoleEcliptic();
    const la = lat * Math.PI / 180, lo = lon * Math.PI / 180;
    // body basis (same as blob painting)
    const ze = vnorm(pole);
    let xe = vnorm(vsub([1, 0, 0], vscale(ze, ze[0])));
    const ye0 = vcross(ze, xe);
    const c = Math.cos(th), s = Math.sin(th);
    const xr = vadd(vscale(xe, c), vscale(ye0, s));
    const yr = vcross(ze, xr);
    const n = vadd(vadd(vscale(xr, Math.cos(la) * Math.cos(lo)), vscale(yr, Math.cos(la) * Math.sin(lo))), vscale(ze, Math.sin(la)));
    const tE = vnorm(vcross(ze, n));
    const tN = vcross(n, tE);
    const e = entryDeg * Math.PI / 180, az = azDeg * Math.PI / 180;
    const h = vadd(vscale(tE, Math.sin(az)), vscale(tN, Math.cos(az)));
    const vdir = vnorm(vadd(vscale(n, -Math.cos(e)), vscale(h, Math.sin(e))));
    const earthPos = this.mirror[0].pos;
    const surf = vadd(earthPos, vscale(n, CATALOG.earth.R));
    const pos = vsub(surf, vscale(vdir, range));
    return { pos, vel: vadd(vscale(vdir, speed), this.mirror[0].vel) };
  }

  // fire a test round and return the offset vector (projectile − target) at closest approach
  _shootMiss(p0, v0, targetName, tof) {
    const bodies = this.mirror.map((m) => ({ p: m.pos.slice(), v: m.vel.slice(), M: m.M }));
    const ti = Math.max(0, this.mirror.findIndex((m) => m.name === targetName));
    let p = p0.slice(), v = v0.slice();
    const n = 240;
    const dt = (tof * 1.3) / n;
    let best = null, bestD = Infinity;
    for (let s = 0; s < n; s++) {
      const sunL = vscale(this.anchor.pos, -1);
      const rA = Math.max(vlen(sunL), 1);
      const aA = vscale(sunL, GM_SUN / (rA * rA * rA));
      for (const m of bodies) {
        const ds = vsub(sunL, m.p);
        const ir = 1 / Math.max(vlen(ds), 1);
        let acc = vsub(vscale(ds, GM_SUN * ir * ir * ir), aA);
        for (const o of bodies) {
          if (o === m) continue;
          const d = vsub(o.p, m.p);
          const rr = Math.max(vlen(d), 1);
          acc = vadd(acc, vscale(d, G_SIM * o.M / (rr * rr * rr)));
        }
        m.v = vadd(m.v, vscale(acc, dt));
      }
      for (const m of bodies) m.p = vadd(m.p, vscale(m.v, dt));
      const dsp = vsub(sunL, p);
      const irp = 1 / Math.max(vlen(dsp), 1);
      let accP = vsub(vscale(dsp, GM_SUN * irp * irp * irp), aA);
      for (const m of bodies) {
        const d = vsub(m.p, p);
        const rr = Math.max(vlen(d), 1);
        accP = vadd(accP, vscale(d, G_SIM * m.M / (rr * rr * rr)));
      }
      v = vadd(v, vscale(accP, dt));
      p = vadd(p, vscale(v, dt));
      const off = vsub(p, bodies[ti].p);
      const dd = vlen(off);
      if (dd < bestD) { bestD = dd; best = off; }
    }
    return best;
  }

  qualityMax() { return Math.min(this.ps.maxN || 65536, Math.round(this.quality * 1.55)); }

  moltenDelta(slot) {
    if (!this.statsCache || !this.statsBase) return 0;
    const b = this.statsCache.bodies[slot];
    if (!b) return 0;
    return Math.max(0, b.moltenFrac - (this.statsBase.molten[slot] || 0));
  }
  boundLoss(slot) {
    if (!this.statsCache || !this.statsBase) return 0;
    const b = this.statsCache.bodies[slot];
    if (!b) return 0;
    return clamp((this.statsBase.bound[slot] || 1) - b.boundFrac, 0, 1);
  }

  // ---------- per-frame tick ----------
  tick(dtWall) {
    this.frame++;
    const out = { substeps: 0, dtSub: 0, rebase: null, readback: false, shifts: [] };
    if (this.paused) { this.warpEff = 0; this._updateDissolve(dtWall); return out; }
    const dtWallSim = Math.min(dtWall, 0.05);   // hitches must not balloon sim steps

    // performance governor: trade warp honesty for smoothness when the GPU is saturated.
    // Higher ceiling + lower fps target: users prefer faster sim over 60fps in the aftermath.
    if (this.frame % 30 === 0) {
      if (this.fps < 24 && this.maxSub > 2) this.maxSub--;
      else if (this.fps > 38 && this.maxSub < 16) this.maxSub++;
    }

    // smooth warp changes in log space (cinematic ramps, plays well with the slow-mo dial)
    // user warp applies INSTANTLY. (This used to ease in log space — a relic from when warp
    // directly scaled the timestep. A 60×→1wk/s jump is ~9 decades, so the ramp delayed the
    // user's input by ~2 s for nothing: dtSubMax, dtCap, the substep budgets and the frozen
    // no-tunnel cap already bound every physics step regardless of how warp jumps.)
    this.warpSmooth = Math.max(0.02, this.warpUser);

    // TWO DIALS for an approaching impactor, both in units of the pair's combined radii (ΣR):
    //   FINE_STEP_R — within this gap the frozen cruise switches to SMALL steps: the impactor
    //                 glides in smoothly and can't tunnel, but the particle SIM is still asleep
    //                 (cheap — only the mirror is moving). FINE_STEP_FRAC = max advance/frame.
    //   SIM_WAKE_R  — within this gap the particle SIMULATION starts (collision physics).
    // Separating them lets the approach look smooth from far out while the expensive sim only
    // ignites right before contact. SIM_WAKE_R < FINE_STEP_R.
    const { fineR: FINE_STEP_R, fineFrac: FINE_STEP_FRAC, wakeR: SIM_WAKE_R } = this.sys;
    // proximity from mirror → dt limits + cinematic auto-slow
    let minGap = 1e12, closing = 0, gapSumR = 1, tWakeMin = Infinity, tCloseMin = Infinity, tFineMin = Infinity, clSoon = 0, wakeNow = false, nearWake = false;
    for (let i = 0; i < this.mirror.length; i++) {
      for (let j = i + 1; j < this.mirror.length; j++) {
        const a = this.mirror[i], b = this.mirror[j];
        if (a.consumed || b.consumed) continue;
        const d = vsub(b.pos, a.pos);
        const dist = vlen(d);
        const gap = dist - a.R - b.R;
        const sumR = a.R + b.R;
        const vr = vsub(b.vel, a.vel);
        const cl = -vdot(vr, d) / Math.max(dist, 1e-6);
        // DIAL 2 — particle sim wakes (surface gap < SIM_WAKE_R · ΣR). Evaluated over EVERY pair
        // (the closest pair is often the non-closing Moon, which must not mask a distant closer).
        const wg = SIM_WAKE_R * sumR;
        if (gap < wg) wakeNow = true;
        if (gap < wg * 2) nearWake = true;              // hysteresis band: don't re-freeze at the line
        // soonest time-to-wake over every closing pair — the frozen cruise must never step past it
        if (cl > 1e-9) tWakeMin = Math.min(tWakeMin, (gap - wg) / cl);
        // DIAL 1 — within FINE_STEP_R · ΣR, hold the frozen advance to FINE_STEP_FRAC · ΣR per
        // frame so the impactor approaches smoothly (the cap is the time to cover that distance)
        if (cl > 1e-9 && gap < FINE_STEP_R * sumR) tFineMin = Math.min(tFineMin, (FINE_STEP_FRAC * sumR) / cl);
        // soonest time-to-CONTACT — drives the cinematic slow-mo (time-based, so a 0.4c
        // closer two minutes out engages it even while the Moon is the closest pair)
        if (cl > 1e-9 && gap > 0 && gap / cl < tCloseMin) { tCloseMin = gap / cl; clSoon = cl; }
        if (gap < minGap) {
          minGap = gap;
          gapSumR = a.R + b.R;
          closing = cl;
        }
      }
    }

    let warp = this.warpSmooth;
    // RELATIVISTIC MODE / cinematic slow-mo: engage on the usual proximity band OR whenever
    // ANY closing pair is under two minutes from contact (time-based — the closest pair is
    // often the non-closing Moon, which masked a 0.4c closer until two frames before impact).
    // Floor of 2× (was 120×) so ultrafast closers play their approach in fine-stepped frozen
    // slow-mo — the glint actually grows — at zero particle cost until the 2-radii wake shell.
    if (this.autoSlow && ((closing > 1e-9 && minGap > 0 && minGap < 30 * gapSumR) || tCloseMin < this.sys.slowWin)) {
      // SPEED-AWARE floor: slow closers keep the classic 120x floor (Moonfall never grinds
      // to 2x, and the contact-rate hold picks up seamlessly from ~120x — no jump at the
      // hit). Fast closers scale down toward 2x for constant-ish on-screen approach speed:
      // a 0.4c lance at 2x still crosses the sky briskly.
      const floorW = clamp(1.2 / Math.max(clSoon, 1e-6), 2, 120);
      warp = Math.min(warp, Math.max(floorW, tCloseMin / this.sys.slowDiv));
    }
    // carnage in slow-mo — and HOLD it: once contact has happened, the cap stays for as long
    // as the live physics runs (no sudden post-impact fast-forward when the fine-dt throttle
    // releases). It lifts naturally when the aftermath settles and the sim freezes to cruise.
    if ((minGap < 0.5 * gapSumR || (this._firstContactT && !this.frozen))) warp = Math.min(warp, 1500);
    // While a freshly-spawned blob is still settling, hold warp down so it gets real wall-time
    // to relax under gravity with small timesteps — otherwise high warp takes giant steps and
    // the planet shock-heats itself white before anything even hits it.
    const settling = this.simTime < this.settleUntil;
    if (settling && !this.frozen) warp = Math.min(warp, 500);   // cap only while actually live
    if (this.warpUser < warp) warp = this.warpUser;            // user slow-mo always wins

    // disturbance assessment → freeze (rigid ride-along) for calm high-warp travel
    const es0 = this.statsCache ? this.statsCache.bodies[0] : null;
    const dSunEarth = vlen(vadd(this.mirror[0].pos, this.anchor.pos));
    const surfT0adj = es0 && this.statsBase ? 288 + (es0.surfT - (this.statsBase.surfT[0] || 288)) : 288;
    // SOLAR ROAST without dynamics: a planet plunging toward the Sun cooks toward the solar
    // equilibrium temperature. Fold that into Earth's atmosphere temp so the FROZEN thermal
    // kernel heats the surface (temps climb, the hot:… headlines fire, population dies) while
    // the particle SIM stays asleep. A sun-plunge is planetary motion, not an impact — it must
    // NOT wake the dynamics, or the suddenly-vapor-pressured surface blows the planet apart.
    // At 1 AU this is ~279 K (= normal), so non-plunge scenarios are unaffected.
    const solarTeq = 278.6 * Math.sqrt(149597.87 / Math.max(dSunEarth, 700));
    const earthAtmT = Math.max(this.atmT || 288, solarTeq);
    // temperature/molten/boundLoss only count as disturbance AFTER a real event (contact,
    // anything touched). Before that they're spawn-settle artifacts (at 1M particles the loose
    // ocean shell alone trips 1% boundLoss). NOTE: sun proximity was here too and woke the sim
    // on a sun-plunge — removed, so Icarus roasts via temperature alone and never explodes.
    const realEvent = this._heatArmed || this.contacts.size > 0 || this.mirror.some((b) => b.touched);
    const disturbed = this.contacts.size > 0 || this.dissolveGo ||
      (realEvent && (surfT0adj > 400 || this.moltenDelta(0) > 0.01 || this.boundLoss(0) > 0.01));
    // aftermath sleep: when every body is mechanically at rest (per fresh readback), the
    // expensive dynamics can sleep even though "disturbed" — only heat keeps evolving
    const statsFresh = this.statsCache && (this.simTime - (this.statsCache.simTimeTag || 0)) < Math.max(1800, this.warpEff * 1.2);
    const allQuiet = !!(statsFresh && this.statsCache.bodies.every((sb) => !sb || sb.count < 3 || sb.rmsV < 0.05)) &&
      minGap > 10 * gapSumR;

    // ── PARTICLE-SIM GATE ──────────────────────────────────────────────────────────────────
    // The particle sim does NOT run until two bodies are reasonably near. The central blob
    // settles live first (the settle window), then the system CRUISES in the frozen rigid
    // ride-along — bodies move on their orbits, particles ride along, only heat evolves —
    // at any warp, on any machine. It WAKES (starts the particle sim) when bodies close to
    // within WAKE radii, so the final approach + impact run fully live. This reuses the exact
    // freeze path already used for aftermath sleep; it just triggers on proximity, not warp.
    // Live-physics dt ceiling must TRACK particle size: dtStable ∝ N^(-1/3) (smaller particles,
    // stiffer relative springs). The old hard floor of 2 s sat ~5× over dtStable at 42k (stable
    // thanks to the in-shader k/damp clamps) but ~12× over at 500k — marginal pairs pump energy
    // every step and the planet eventually detonates at rest. Keep the same 5× margin at every N
    // instead: identical behavior at 42k, proportionally finer steps at half a million particles.
    const dtCap = clamp(this.ps.dtStable * 5, 0.3, 8);
    if (this._wake && this.frozen) { this._wake = false; this.frozen = false; }
    // SETTLE MODE (SYS panel diagnostic): force the planet live with the rigid hold released so it
    // relaxes to equilibrium on screen (shells off). Never freezes while engaged.
    if (this.settleMode && this.frozen) {
      this.frozen = false;
      for (const b of this.mirror) {
        out.shifts.push({ slot: b.slot, dp: vsub(b.pos, b.freezePos || b.pos), dv: vsub(b.vel, b.freezeVel || b.vel) });
      }
    }
    // AUTO-RELAX: a fresh scenario eases to its TRUE equilibrium COOL (heat off, rigid hold released)
    // so the spawn-vs-equilibrium energy bleeds off as damped motion — THEN it freezes that SETTLED
    // state below. Freezing the raw spawn instead pins latent collapse energy that the first impact
    // unleashes globally (the "any touch detonates" bug). Exits the instant the lattice is at rest
    // (cheap once packComp is dialed in), or a safety cap. Hidden behind the textured globe at load.
    if (this.relaxing) {
      this.frozen = false;
      const eb = this.statsCache && this.statsCache.bodies[0];
      // Exit on STRUCTURAL convergence, not rms: settle-drag zeroes velocity long before the
      // positions reach equilibrium (measured — rms 0.004 still cooks on arm, the radius is still
      // contracting). √(Iw/mass) is the planet's structural radius; when it stops shrinking, the
      // lattice is truly at rest and arming releases nothing. Floor + hard cap as backstops.
      if (eb && eb.mass > 0 && eb.count > 2) {
        const rmsR = Math.sqrt((eb.Iw || 0) / eb.mass);
        const settled = this._relaxR > 0 && Math.abs(rmsR - this._relaxR) / rmsR < 8e-5;
        this._relaxR = rmsR;
        if ((this.simTime > 400 && settled) || this.simTime > 12000) this.relaxing = false;
      } else if (this.simTime > 12000) this.relaxing = false;
    }
    if (this.frozen) {
      if (disturbed || wakeNow) {
        this.frozen = false;          // wake: hand the rigid offset back, resume live physics
        for (const b of this.mirror) {
          out.shifts.push({ slot: b.slot, dp: vsub(b.pos, b.freezePos), dv: vsub(b.vel, b.freezeVel) });
        }
      }
    } else if (!this.settleMode && !this.relaxing && !disturbed && (!nearWake || (!settling && allQuiet))) {
      // NOTE: the settle window no longer blocks freezing when every body is far apart —
      // a scenario LOADS frozen (no visible churn at start, like cruise). The lattice
      // relaxes later during the live approach window, held rigid by settle drag.
      this.frozen = true;             // sleep: capture the rigid reference, stop the sim
      for (const b of this.mirror) { b.freezePos = b.pos.slice(); b.freezeVel = b.vel.slice(); }
    }
    // freeze-gate telemetry
    this._minGapR = minGap / gapSumR;
    this._settling = settling;
    this._wakeR = 2; this._sleepR = 3.2;   // nominal (radii); speed run-up can widen the trigger
    // why is the expensive sim awake? (sim._gateWhy in the console — no more guessing)
    this._gateWhy = this.frozen ? 'frozen'
      : settling ? 'settling'
      : disturbed ? (this.contacts.size > 0 ? 'contacts' : this.dissolveGo ? 'dissolve' : dSunEarth < 42000 ? 'sun' : 'aftermath')
      : wakeNow ? 'proximity' : 'awake(hysteresis)';

    // timestep ceiling. Frozen cruise takes huge steps when far, but must not step PAST the
    // wake point (so a fast/relativistic impactor can't tunnel through it). Live physics is
    // capped near the stable dt, shrinking further on close approach (lance slow-mo).
    let dtSubMax;
    if (this.frozen) {
      dtSubMax = isFinite(tWakeMin) ? clamp(tWakeMin, 0.05, 1200) : 1200;
    } else {
      // Drive dt by the fastest body motion near Earth. Using a body's SPEED (not the closing
      // rate, which flips sign at Earth's centre) keeps a penetrator at a fine timestep through
      // its WHOLE traversal — it cannot tunnel out the back half without depositing its energy.
      // CRITICAL: gated PER BODY on that body's OWN distance — a 0.4c slug 80,000 Mm out must
      // not pin the whole sim to microsecond steps before it even arrives.
      let vMax = 0;
      for (let j = 1; j < this.mirror.length; j++) {
        const b = this.mirror[j];
        if (b.consumed) continue;
        const sumRj = this.mirror[0].R + b.R;
        const gapJ = vlen(vsub(b.pos, this.mirror[0].pos)) - sumRj;
        if (gapJ < 50 * sumRj) vMax = Math.max(vMax, vlen(vsub(b.vel, this.mirror[0].vel)));
      }
      dtSubMax = vMax > 0.05 ? clamp((0.4 * this.ps.minRp) / vMax, 1e-5, dtCap) : dtCap;
    }
    // Particle-sim workload is DECOUPLED from the slider: live collision physics runs a small
    // FIXED substep budget per frame (smooth at any warp), so the slider can't pile up O(N²)
    // passes and stutter. The frozen cruise still takes many big steps, so the slider speeds up
    // the planetary/approach motion — not the collision. The slider then reads back the MEASURED
    // rate (warpEff) so it never claims 60× while actually running 12×.
    // Substep budget is AUTO-DERIVED from the no-tunnel ceiling: substeps = ceil(want / dtSubMax),
    // so the faster the impactor, the smaller dtSubMax, the more substeps it automatically takes.
    // Normal collisions stay at 4 (smooth, warp-decoupled — no jitter). Only when the no-tunnel
    // ceiling has forced a tiny dt (a hypervelocity/relativistic impactor) is the larger budget it
    // physically needs unlocked — the impact is brief, so the short, heavier burst is worth it.
    const hyper = !this.frozen && dtSubMax < 0.5;
    // SUBSTEP THROUGHPUT GOVERNOR: the live budget of 4/frame was tuned so weak GPUs never
    // stutter — but it also caps a 4090 at 4 tiny steps per frame at 1M particles (slo-mo with
    // the GPU mostly idle). Climb the budget slowly while frames stay fast, drop it fast when
    // they sag: big GPUs buy back sim-rate at high N, small ones never leave the floor.
    if (!this.frozen) {
      const fps = this.fps || 60;
      if (fps > 47) this._subBoost = Math.min((this._subBoost || 4) + 0.25, 24);
      else if (fps < 30) this._subBoost = Math.max((this._subBoost || 4) - 1, 4);
    }
    const liveCap = this.frozen ? 12 : (hyper ? 32 : Math.round(this._subBoost || 4));
    let want = warp * dtWallSim;
    // Auto-relax advances at the full live budget (not the warp) so it converges in a frame or two,
    // hidden behind the globe — otherwise a slow-warp scenario would relax for many real seconds.
    if (this.relaxing && !this.settleMode) want = Math.max(want, liveCap * dtSubMax);
    // FROZEN NO-TUNNEL: cap the whole frame's advance at the time-to-reach-the-wake-shell, so
    // a hypervelocity closer lands ON the shell and wakes next tick instead of stepping through
    // it. The floor was 0.05 s — but at 0.4c that's a 6 Mm hop, big enough to skip the entire
    // shell and bury the Lance inside Earth before the sim ever ticked. A tiny floor (1e-4 s)
    // keeps progress without overshooting (≈12 km past the shell at 0.4c, negligible).
    if (this.frozen && isFinite(tWakeMin)) want = Math.min(want, Math.max(tWakeMin, 1e-4));
    // DIAL 1 — fine-step approach: once inside FINE_STEP_R, cap the frozen advance so the
    // impactor moves only FINE_STEP_FRAC·ΣR per frame (smooth, tunnel-proof) while still asleep.
    if (this.frozen && isFinite(tFineMin)) want = Math.min(want, Math.max(tFineMin, 1e-4));
    let substeps = clamp(Math.ceil(want / dtSubMax), 1, liveCap);
    let dtSub = Math.min(want / substeps, dtSubMax);
    const simDt = dtSub * substeps;
    this.warpEff = simDt / dtWallSim;
    this.simTime += simDt;
    this._lastSimDt = simDt;
    this.disturbedNow = disturbed;

    // settle boost decay
    const settleBoost = this.simTime < this.settleUntil ? lerp(4.5, 1, clamp(1 - (this.settleUntil - this.simTime) / 1800, 0, 1)) : 1;

    for (const b of this.mirror) b.prevPos = b.pos.slice();   // for swept contact tests
    // integrate anchor + mirror (symplectic Euler) in Earth's free-falling frame.
    // The anchor itself free-falls under the Sun (so the frame tracks Earth's true curved
    // orbit), and bodies feel DIFFERENTIAL (tidal) sun gravity = their pull minus the
    // anchor's pull. A body sitting at the anchor (Earth) stays pinned at the origin instead
    // of drifting sunward off its own orbit line.
    for (let s = 0; s < substeps; s++) {
      const sunL = vscale(this.anchor.pos, -1);          // Sun position in the local frame
      const rA = Math.max(vlen(sunL), 1);
      const aAnchor = vscale(sunL, GM_SUN / (rA * rA * rA));
      this.anchor.vel = vadd(this.anchor.vel, vscale(aAnchor, dtSub));
      this.anchor.pos = vadd(this.anchor.pos, vscale(this.anchor.vel, dtSub));
      for (const b of this.mirror) {
        if (b.consumed) continue;          // halted at the Sun's centre — no more motion
        const ds = vsub(sunL, b.pos);
        const r2s = vdot(ds, ds);
        const ir = 1 / Math.sqrt(Math.max(r2s, 1));
        let acc = vsub(vscale(ds, GM_SUN * ir * ir * ir), aAnchor);
        for (const o of this.mirror) {
          if (o === b || o.consumed) continue;
          const d = vsub(o.pos, b.pos);
          const r2 = vdot(d, d) + 0.05;
          const irr = 1 / Math.sqrt(r2);
          acc = vadd(acc, vscale(d, G_SIM * o.M * irr * irr * irr));
        }
        b.vel = vadd(b.vel, vscale(acc, dtSub));
      }
      for (const b of this.mirror) { if (!b.consumed) b.pos = vadd(b.pos, vscale(b.vel, dtSub)); }
    }

    // contact events (first touch per pair) — SWEPT test: a 0.1c body must not be able to
    // cross a planet between two ticks without the segment registering the hit
    for (let i = 0; i < this.mirror.length; i++) {
      for (let j = i + 1; j < this.mirror.length; j++) {
        const a = this.mirror[i], b = this.mirror[j];
        const key = i + '-' + j;
        if (this.contacts.has(key)) continue;
        let dist = vlen(vsub(b.pos, a.pos));
        if (a.prevPos && b.prevPos) {
          const r0 = vsub(b.prevPos, a.prevPos);
          const dr = vsub(vsub(b.pos, a.pos), r0);
          const len2 = vdot(dr, dr);
          if (len2 > 1e-9) {
            const t = clamp(-vdot(r0, dr) / len2, 0, 1);
            dist = Math.min(dist, vlen(vadd(r0, vscale(dr, t))));
          }
        }
        if (dist < (a.R + b.R) * 1.04) {
          this.contacts.add(key);
          a.touched = true; b.touched = true;
          const vrel = vlen(vsub(b.vel, a.vel));
          const mu = (a.M * b.M) / (a.M + b.M);
          const E = 0.5 * mu * vrel * vrel * 1e36;
          this.cumImpactJ += E;
          this._fireCond('contact');
          if (a.name === 'Earth' || b.name === 'Earth') {
            this.dissolveGo = true;
            this.banner = { text: `IMPACT — ${(b.name === 'Earth' ? a.name : b.name)} · ${(vrel * 1000).toFixed(1)} km/s · ${this._fmtEnergy(E)}`, kind: 'danger', until: performance.now() + 9000 };
            const earth = a.name === 'Earth' ? a : b, imp = a.name === 'Earth' ? b : a;
            this._spawnEjecta(earth, imp, vrel, E);
          }
        }
      }
    }
    // predictive dissolve & approach event
    if (!this.dissolveGo && this.mirror.length > 1) {
      const e = this.mirror[0];
      for (let j = 1; j < this.mirror.length; j++) {
        const b = this.mirror[j];
        const d = vsub(b.pos, e.pos);
        const dist = vlen(d);
        const gap = dist - e.R - b.R;
        const cl = -vdot(vsub(b.vel, e.vel), d) / Math.max(dist, 1e-6);
        if (gap < 8 * e.R && cl > 0) this._fireCond('approach');
        // globe stays until the bodies are ACTUALLY touching (user: no early particle flash)
        if (gap <= 0) this.dissolveGo = true;
      }
    }
    if (this.statsCache && this.statsBase) {
      const es = this.statsCache.bodies[0];
      const sT = es ? 288 + (es.surfT - (this.statsBase.surfT[0] || 288)) : 288;
      // temp/molten only dissolve the globe after a REAL event — wake-relaxation churn must not
      if (realEvent && es && (sT > 430 || this.moltenDelta(0) > 0.012)) this.dissolveGo = true;
      if (sT > 330) this._fireCond('hot:330');
      if (sT > 373) this._fireCond('hot:373');
      if (sT > 1500) this._fireCond('hot:1500');
    }
    this._updateDissolve(dtWall);
    this._updateSunConsumption();

    // first-contact clock — drives 'after:' headlines AND the held impact slow-mo
    if (this.contacts.size > 0 && !this._firstContactT) {
      this._firstContactT = this.simTime;
      // pull the DIAL itself down to THE RATE YOU ARE WATCHING AT — the measured slow-mo of
      // the approach (throttled physics), not an arbitrary cap. 1500× still read as warp
      // speed for debris dynamics once the throttle eased. The held rate is now the real
      // request — it stays this slow after the dust settles until the user pushes the dial.
      this.warpUser = Math.min(this.warpUser, clamp(this.warpEff || 300, 30, 600));
    }
    // timed headlines
    for (const ev of this.events) {
      if (!ev.fired && ev.t !== undefined && this.simTime >= ev.t) this._fire(ev);
      if (!ev.fired && ev.cond && ev.cond.startsWith('after:') && this.contacts.size > 0) {
        if (this.simTime - this._firstContactT > parseFloat(ev.cond.slice(6))) this._fire(ev);
      }
    }
    this._fillerHeadlines(dtWall);

    // atmospheric heat reservoir: impacts dump a slice of their energy into the air, which
    // bakes the whole surface (the K-Pg global-broil mechanism) and radiates away over days
    const newJ = this.cumImpactJ - (this._lastCumJ || 0);
    this._lastCumJ = this.cumImpactJ;
    this.atmE = ((this.atmE || 0) + newJ * 0.02) * Math.exp(-simDt / 2.0e5);
    this.atmT = Math.min(2600, 288 + this.atmE / 5.1e21);

    this._population(simDt);

    if (this.frozen) {
      // particles ride rigidly along mirror trajectories — no dynamics dispatch.
      // Heat still evolves via the O(N) thermal kernel (cooling/bake at any timestep),
      // and readbacks keep temps/population/livable-land flowing.
      for (const b of this.mirror) {
        const dp = vsub(b.pos, b.freezePos);
        // rigid SPIN while frozen: pristine bodies use their known spin (Earth's sidereal
        // rate), disturbed remnants use the ω estimated from angular momentum in the stats —
        // planets keep visibly rotating through cruise and aftermath sleep.
        const w = (b.omegaEst && vlen(b.omegaEst) > 1e-9) ? b.omegaEst : (b.spin || null);
        let rot = null;
        if (w) {
          const mag = vlen(w);
          if (mag * simDt > 1e-7) {
            rot = { axis: vscale(w, 1 / mag), angle: mag * simDt, com: b.freezePos, vcom: b.freezeVel };
          }
        }
        if (vdot(dp, dp) > 1e-10 || rot) {
          out.shifts.push({ slot: b.slot, dp, dv: [0, 0, 0], rot });
          b.freezePos = b.pos.slice();
        }
      }
      this.ps.writeBodyDyn(this.mirror.map((b) => ({
        slot: b.slot, vel: b.vel, pos: b.pos, spin: b.spin,
        atmT: b.slot === 0 ? earthAtmT : 0,
      })));
      this.ps.writeParams({
        dt: simDt, settleBoost: 1, sunPos: vscale(this.anchor.pos, -1), gmSun: GM_SUN,
        coolMul: this.view.coolMul, heatMul: this.view.heatMul, time: this.simTime,
        solarLum: 1, heatGate: 1, settleDrag: 0,
        vClamp: Math.max(1, this.vClampNeed || 1),
      });
      out.thermal = true;
      out.substeps = 0;
      out.readback = this.frame % 40 === 0;
    } else {
      this.ps.writeBodyDyn(this.mirror.map((b) => ({
        slot: b.slot, vel: b.vel, pos: b.pos, spin: b.spin,
        atmT: b.slot === 0 ? earthAtmT : 0,
      })));
      this.ps.writeParams({
        dt: dtSub,
        settleBoost,
        sunPos: vscale(this.anchor.pos, -1),
        gmSun: GM_SUN,
        coolMul: this.view.coolMul,
        heatMul: this.view.heatMul,
        time: this.simTime,
        solarLum: 1,
        vClamp: Math.max(1, this.vClampNeed || 1),
        // frictional heat fires ONLY from a REAL event — an actual contact, or a sun-plunge.
        // It must NOT arm on boundLoss/molten/surfT, because at low density a coarse blob
        // churns slightly as it settles, and arming on that churn creates a self-amplifying
        // leak that boils the planet before anything hits it. Latches on so aftermath cooks.
        // Settle mode stays COOL — a pure spatial relaxation. Heat is off so you watch the lattice
        // find its resting shape (hold = good, collapse/fly-apart = bad) without cooking it.
        heatGate: (this.simTime > this.settleUntil && (this._heatArmed = this._heatArmed || this.contacts.size > 0 || dSunEarth < 42000)) ? 1 : 0,
        // A body that has NOT been hit by a real event stays locked to rigid-body motion (bulk
        // drift + spin), so its surface can't drift off and a coarse low-density blob can't shed
        // its skin. The instant a real impact arms heat, the drag releases and physics splashes
        // it freely. This replaces the old fixed settle window — calm planets just hold their shape.
        // settle-drag (the damping that ends bulk movement) stays ON during relax/settle — releasing
        // it (=0) is what let the lattice churn instead of dissipating. Only a real impact frees it.
        settleDrag: (this._heatArmed || this.contacts.size > 0) ? 0 : 1 / 60,
        consumedMask: this.consumedMask || 0,
      });
      out.substeps = substeps;
      out.dtSub = dtSub;
      out.readback = this.relaxing || this.frame % 20 === 0;   // fresh rmsV so relax exits on cue
    }

    // rebase if Earth wandered from origin
    if (this.statsCache) {
      const es = this.statsCache.bodies[0];
      if (es && vlen(es.com) > 1500) {
        out.rebase = { dp: es.com.slice(), dv: es.cov.slice() };
        this._applyRebase(out.rebase);
      }
    }

    // trails — a body's trajectory story ENDS at impact: the approach arc lingers briefly,
    // then clears (post-impact CoM slosh would otherwise scribble spaghetti)
    for (const b of this.mirror) {
      if (b.touched) {
        if (!b.trailEndAt) b.trailEndAt = performance.now() + 3500;
        if (b.trail.length && performance.now() > b.trailEndAt) { b.trail = []; b.dirty = true; }
        continue;
      }
      const last = b.trail.length ? b.trail[b.trail.length - 1] : null;
      const step = b.trailStep || Math.max(1.5, b.R * 0.5);
      // per-body trail frame: 'geo' records Earth-relative (Moon), default heliocentric
      const cur = b.trailFrame === 'geo'
        ? vsub(b.pos, this.mirror[0].pos)
        : vadd(this.anchor.pos, b.pos);
      if (!last || vlen(vsub(cur, last)) > step) {
        b.trail.push(cur);
        while (b.trail.length > (b.trailMax || 500)) b.trail.shift();
        b.dirty = true;
      }
    }
    return out;
  }

  _applyRebase(rb) {
    this.anchor.pos = vadd(this.anchor.pos, rb.dp);
    this.anchor.vel = vadd(this.anchor.vel, rb.dv);
    for (const b of this.mirror) {
      b.pos = vsub(b.pos, rb.dp);
      b.vel = vsub(b.vel, rb.dv);
      // trails are heliocentric — a local-frame rebase doesn't touch them
      b.dirty = true;
    }
    if (this.statsCache) for (const sb of this.statsCache.bodies) { sb.com = vsub(sb.com, rb.dp); sb.cov = vsub(sb.cov, rb.dv); }
    this.ps.epoch++;   // invalidate in-flight readbacks (old frame)
  }

  _updateDissolve(dtWall) {
    if (this.view.forceParticles || this.settleMode) { this.dissolve = 1; return; }
    if (this.dissolveGo && this.dissolve < 1) this.dissolve = Math.min(1, this.dissolve + dtWall / 1.6);
  }

  // THE SUN IS A UNIVERSAL DESTROYER. A whole body is annihilated the instant its path crosses
  // within 5× the Sun's radius. A swept (point-to-segment) test catches fast in-fallers that
  // would tunnel a sphere test. Sets the slot bit → the GPU kills every particle of that body
  // (halted, mass 0, parked at the Sun's centre). The mask latches; consumption is forever.
  _updateSunConsumption() {
    const R_SUN = 695.7, killR2 = (5 * R_SUN) ** 2;
    if (this.consumedMask === undefined) this.consumedMask = 0;
    for (const b of this.mirror) {
      const helioNow = vadd(this.anchor.pos, b.pos);   // Sun sits at the heliocentric origin
      if (!b._helioPrev) b._helioPrev = helioNow.slice();
      if (!b.consumed) {
        const a = b._helioPrev, ab = vsub(helioNow, a), dd = vdot(ab, ab);
        const t = dd > 1e-9 ? clamp(-vdot(a, ab) / dd, 0, 1) : 0;
        const closest = vadd(a, vscale(ab, t));
        if (vdot(closest, closest) < killR2) {
          b.consumed = true;
          this.consumedMask |= (1 << b.slot);
          b.vel = [0, 0, 0];
          this.banner = { text: `☀ ${b.name.toUpperCase()} CONSUMED BY THE SUN`, kind: 'danger', until: performance.now() + 9000 };
          this.headlineQueue.push(`${b.name} has fallen into the Sun. Nothing remains.`);
          if (b.name === 'Earth') { this.dissolveGo = true; this._heatArmed = true; this.killTarget = 1; this.pop = 0; }
        }
      }
      b._helioPrev = helioNow.slice();
    }
  }

  applyStats(stats) {
    this.statsCache = stats;
    if (this.thermalJ0 === null) this.thermalJ0 = stats.thermalJ;
    if (!this.statsBase) {
      // baseline: Earth legitimately spawns with a molten interior — only deltas count as damage
      this.statsBase = {
        molten: stats.bodies.map((b) => b.moltenFrac),
        bound: stats.bodies.map((b) => b.boundFrac),
        surfT: stats.bodies.map((b) => b.surfT),
        crustLiv: Math.max(1, stats.crustLiv || 0),
      };
    }
    // Sync mirror blobs to particle truth ONLY after real disruption and only from fresh
    // snapshots. For pristine orbits the analytic mirror is the better integrator — syncing
    // it to a stale snapshot at high warp rewinds the orbit (bodies spiral into the sun).
    const staleness = this.simTime - (stats.simTimeTag || 0);
    const fresh = staleness < Math.max(900, this.warpEff * 0.5);
    if (!this.frozen && this.disturbedNow && fresh) {
      for (const b of this.mirror) {
        if (b.consumed) continue;          // its particles are parked at the Sun — ignore their CoM
        const sb = stats.bodies[b.slot];
        if (sb && sb.count > 2) { b.pos = sb.com.slice(); b.vel = sb.cov.slice(); }
      }
    }
    // measured spin (L/I) — drives rigid rotation during frozen cruise/aftermath sleep
    for (const b of this.mirror) {
      const sb = stats.bodies[b.slot];
      if (sb && sb.count > 8 && sb.omega) b.omegaEst = sb.omega.slice();
    }
    // merge swallowed bodies into Earth: a CoM deep inside the planet means accretion is
    // done — keeping it as a separate "body" poisons the calm detector and threat logic
    for (let j = this.mirror.length - 1; j >= 1; j--) {
      const b = this.mirror[j];
      const sb = stats.bodies[b.slot];
      const d = vlen(vsub(b.pos, this.mirror[0].pos));
      if (sb && sb.count > 2 && d < this.mirror[0].R * 0.85) {
        const e = this.mirror[0];
        const pTot = vadd(vscale(e.vel, e.M), vscale(b.vel, b.M));
        e.M += b.M;
        e.vel = vscale(pTot, 1 / e.M);
        this.mergedNames.push(b.name);
        this.mirror.splice(j, 1);
      }
    }
  }

  _population(simDt) {
    const s = this.statsCache;
    const es = s ? s.bodies[0] : null;
    // baseline-relative surface temperature (the shell average includes warm inner-crust by design)
    const surfT = es && this.statsBase ? 288 + (es.surfT - (this.statsBase.surfT[0] || 288)) : 288;
    const massLoss = this.boundLoss(0);
    const molten = this.moltenDelta(0);
    // POPULATION = livable land (user spec): fraction of Earth's CRUST particles still
    // attached to the planet below 322K, times a climate factor for sub-resolution impacts
    // (a real Chicxulub barely warms any particle — its kill is dust/winter, carried by kE).
    const livBase = this.statsBase ? this.statsBase.crustLiv : 0;
    const liv = (s && livBase > 0) ? clamp((s.crustLiv || 0) / livBase, 0, 1) : 1;
    // Every death needs a VISIBLE cause on a visible timescale:
    //  - hot atmosphere kills as the air cooks (you watch the bake char the land)
    //  - impact-winter (dust) kills SLOWLY over days, never instantly at 17°C
    const kAtm = curve([[305, 0], [318, 0.08], [335, 0.35], [360, 0.75], [400, 0.95], [500, 1]], this.atmT || 288);
    const kE = curve([[15, 0], [18, 0.0005], [20, 0.015], [21, 0.06], [22, 0.18], [23, 0.45], [23.63, 0.72], [24.3, 0.9], [25, 0.985], [26, 1]], Math.log10(Math.max(this.cumImpactJ, 1)));
    this.kDust = (this.kDust || 0) + (kE - (this.kDust || 0)) * (1 - Math.exp(-simDt / 216000));
    const kill = 1 - liv * (1 - kAtm) * (1 - this.kDust);
    this.livFrac = liv;
    // No casualties until a real event has armed the sim — a coarse blob settling on a weak
    // laptop must never read as a mass-extinction before anything has actually happened.
    if (this._heatArmed || this.contacts.size > 0) this.killTarget = Math.max(this.killTarget, kill);
    const target = POP_2026 * (1 - this.killTarget);
    // deaths are effectively instant at apocalypse scale; only slow-burn kills linger
    const tau = this.killTarget > 0.5 ? 240 : 3600;
    this.pop += (target - this.pop) * (1 - Math.exp(-simDt / tau));
    if (this.pop < 1000) this.pop = 0;
    this.surfTDisplay = clamp(surfT, 2.7, 30000);
    this.massLossDisplay = massLoss;
    this.energyDisplay = Math.max(this.cumImpactJ, s ? Math.max(0, s.thermalJ - (this.thermalJ0 || 0)) : 0);
  }

  // rotating absurd-news ticker, fired on a wall-clock cadence and chosen by doom phase
  _fillerHeadlines(dtWall) {
    this._fillerT = (this._fillerT || 0) + dtWall;
    const period = this.killTarget > 0.05 ? 7 : 11;   // crank up the panic during carnage
    if (this._fillerT < period || this.headlineQueue.length > 1) return;
    this._fillerT = 0;
    let phase = 'calm';
    if (this.pop <= 0 || this.killTarget > 0.92) phase = 'aftermath';
    else if (this.killTarget > 0.04) phase = 'doom';
    else if (this.dissolveGo || this.contacts.size > 0 || this._approachFired) phase = 'panic';
    const pool = FILLER[phase];
    if (!this._fillerSeen) this._fillerSeen = new Set();
    let pick = pool[Math.floor(this._rng() * pool.length)];
    for (let i = 0; i < 4 && this._fillerSeen.has(pick); i++) pick = pool[Math.floor(this._rng() * pool.length)];
    if (this._fillerSeen.has(pick)) return;            // pool exhausted this phase
    this._fillerSeen.add(pick);
    this.headlineQueue.push(pick);
  }

  _rng() {
    // deterministic-ish PRNG (no Math.random — keeps the sim reproducible)
    this._rngS = ((this._rngS || (this.frame * 2654435761)) ^ (this.frame << 13)) >>> 0;
    this._rngS = (this._rngS * 1664525 + 1013904223) >>> 0;
    return this._rngS / 4294967296;
  }

  _fire(ev) {
    if (ev.cond === 'approach') this._approachFired = true;
    ev.fired = true;
    this.headlineQueue.push(ev.text);
  }
  _fireCond(c) {
    for (const ev of this.events) if (!ev.fired && ev.cond === c) this._fire(ev);
  }
  _fmtEnergy(J) {
    const mt = J / 4.184e15;
    if (J > CHICX_J * 0.05) return (J / CHICX_J).toFixed(J > CHICX_J * 3 ? 0 : 2) + '× Chicxulub';
    if (mt >= 1) return Math.round(mt).toLocaleString() + ' Mt';
    return (mt * 1000).toFixed(1) + ' kt';
  }

  moonStatus() {
    if ((this.mergedNames || []).includes('Moon')) return 'MERGED WITH EARTH';
    const m = this.mirror.find((b) => b.name === 'Moon');
    if (!m) return '—';
    const e = this.mirror[0];
    const sb = this.statsCache?.bodies[m.slot];
    const d = vlen(vsub(m.pos, e.pos));
    if (sb && (this.boundLoss(m.slot) > 0.7 || d < e.R * 1.1)) return 'MERGED WITH EARTH';
    if (sb && (this.moltenDelta(m.slot) > 0.22 || this.boundLoss(m.slot) > 0.25)) return 'DISRUPTED ☠';
    const vrel = vlen(vsub(m.vel, e.vel));
    const eps = vrel * vrel / 2 - G_SIM * e.M / Math.max(d, 0.1);
    if (eps > 0 && d > 1200) return 'ESCAPING';
    if (this.scenarioId === 'moonfall' && this.contacts.size === 0) return `FALLING — ${Math.round(d - e.R - m.R).toLocaleString()} Mm`;
    return `ORBITING · ${Math.round(d).toLocaleString()} Mm`;
  }

  // ---------- camera / labels / scene ----------
  jdNow() { return this.jd0 + this.simTime / 86400; }

  focusTargets() {
    const t = [{ id: 'frame', name: '◆ Frame (auto)', R: CATALOG.earth.R },
      { id: 'earth', name: 'Earth', R: CATALOG.earth.R }];
    for (const b of this.mirror) if (b.name !== 'Earth') t.push({ id: 'body:' + b.name, name: b.name, R: Math.max(b.R, 0.3) });
    t.push({ id: 'sun', name: 'Sun', R: 695.7 });
    for (const n of PLANET_NAMES) if (n !== 'earth') t.push({ id: 'planet:' + n, name: CATALOG[n].name, R: CATALOG[n].R });
    return t;
  }

  // One full original orbit sampled in EQUAL TIME STEPS (leapfrog) from a state at sim start.
  // Returns { pts (M×3 relative to attractor), M, dt } — fixed for the run; the body deviates from it.
  _fullOrbit(r0, v0, GM, M) {
    let px = r0[0], py = r0[1], pz = r0[2], vx = v0[0], vy = v0[1], vz = v0[2];
    const r = Math.hypot(px, py, pz), v2 = vx * vx + vy * vy + vz * vz;
    const energy = v2 / 2 - GM / Math.max(r, 1e-6);
    let period;
    if (energy < -1e-12) { const a = -GM / (2 * energy); period = 2 * Math.PI * Math.sqrt(a * a * a / GM); }
    else period = 8 * r / Math.max(Math.sqrt(v2), 1e-6);   // unbound: pseudo-span
    period = Math.min(period, 6.5e9);                        // cap ~205 yr (Neptune)
    const dt = period / M;
    const pts = new Float64Array(M * 3);
    for (let i = 0; i < M; i++) {
      pts[i * 3] = px; pts[i * 3 + 1] = py; pts[i * 3 + 2] = pz;
      let rr = Math.hypot(px, py, pz), f = -GM / (rr * rr * rr);
      vx += px * f * dt * 0.5; vy += py * f * dt * 0.5; vz += pz * f * dt * 0.5;
      px += vx * dt; py += vy * dt; pz += vz * dt;
      rr = Math.hypot(px, py, pz); f = -GM / (rr * rr * rr);
      vx += px * f * dt * 0.5; vy += py * f * dt * 0.5; vz += pz * f * dt * 0.5;
    }
    return { pts, M, dt };
  }

  // Capture every standard-orbit body's ORIGINAL orbit at sim start: Earth + planets (heliocentric),
  // canonical Moon (geocentric). These stay fixed — the whole point is to watch bodies deviate from them.
  _captureOriginalOrbits() {
    this.origArcs = [];
    const e = this.mirror[0], M = 256;
    this.origArcs.push({ ...this._fullOrbit(vadd(this.anchor.pos, e.pos), vadd(this.anchor.vel, e.vel), GM_SUN, M), frame: 'helio', color: [0.35, 0.7, 1], kind: 'earth' });
    for (const nm of PLANET_NAMES) {
      if (nm === 'earth') continue;
      this.origArcs.push({ ...this._fullOrbit(planetPos(nm, this.jd0), planetVel(nm, this.jd0), GM_SUN, M), frame: 'helio', color: [0.5, 0.58, 0.72], kind: 'planet', name: nm });
    }
    for (const b of this.mirror) {
      if (b.slot === 0 || !b.canonical) continue;   // Moon on its real orbit → geocentric ring
      this.origArcs.push({ ...this._fullOrbit(vsub(b.pos, e.pos), vsub(b.vel, e.vel), G_SIM * e.M, M), frame: 'geo', color: b.color.slice(), kind: 'moon', slot: b.slot });
    }
  }

  // Draw each original orbit as a trailing arc: HEAD = the body's live centre (dead-centre, recomputed
  // each frame), then the fixed original orbit walked backward ~95% of its period, fading the tail.
  // When undisturbed the head sits exactly on the arc; under perturbation it visibly peels away.
  _buildOrbitArcs(focus, jd) {
    this.renderer.clearGhosts();
    if (!this.view.orbits || !this.origArcs) return false;
    const e = this.mirror[0], eHelio = vadd(this.anchor.pos, e.pos);
    let gi = 0;
    for (const arc of this.origArcs) {
      if (gi >= 15) break;
      if (arc.kind !== 'planet') continue;   // mirror bodies carry their orbit as their TRAIL now
      let curHelio;   // body's ACTUAL heliocentric centre now
      if (arc.kind === 'earth') curHelio = eHelio;
      else if (arc.kind === 'planet') curHelio = planetPos(arc.name, jd);
      else { const b = this.mirror.find((x) => x.slot === arc.slot); if (!b || b.consumed) continue; curHelio = vadd(this.anchor.pos, b.pos); }
      const M = arc.M, head = Math.floor(this.simTime / arc.dt), count = Math.floor(0.95 * M), nPts = count + 1;
      const buf = new ArrayBuffer(nPts * 16), f = new Float32Array(buf), u8 = new Uint8Array(buf);
      const hr = vsub(curHelio, focus);
      f[0] = hr[0]; f[1] = hr[1]; f[2] = hr[2];
      u8[12] = arc.color[0] * 255; u8[13] = arc.color[1] * 255; u8[14] = arc.color[2] * 255; u8[15] = 255;
      for (let k = 0; k < count; k++) {
        const idx = ((head - k) % M + M) % M;
        const wx = arc.pts[idx * 3], wy = arc.pts[idx * 3 + 1], wz = arc.pts[idx * 3 + 2];
        const j = k + 1;
        if (arc.frame === 'geo') { f[j * 4] = eHelio[0] + wx - focus[0]; f[j * 4 + 1] = eHelio[1] + wy - focus[1]; f[j * 4 + 2] = eHelio[2] + wz - focus[2]; }
        else { f[j * 4] = wx - focus[0]; f[j * 4 + 1] = wy - focus[1]; f[j * 4 + 2] = wz - focus[2]; }
        const frac = j / nPts;
        const al = frac < 0.7 ? 1 : Math.max(0, 1 - (frac - 0.7) / 0.3);   // fade the trailing 30%
        const o = j * 16 + 12;
        u8[o] = arc.color[0] * 255; u8[o + 1] = arc.color[1] * 255; u8[o + 2] = arc.color[2] * 255; u8[o + 3] = al * 255;
      }
      this.renderer.writeGhost(gi++, new Float32Array(buf), nPts);
    }
    return gi > 0;
  }

  // Spawn an impact ejecta curtain from the surface point under the impactor. Count scales with
  // impact energy (cube-root → excavated-mass-ish), so Chicxulub gets a modest spray and the Lance
  // a planet-shrouding fountain. Earth-centred frame; the ejecta pool integrates ballistically.
  _spawnEjecta(earth, imp, vrel, E) {
    if (!this.ejecta || !this.view.ejecta) return;
    // Crater regime only. Once the impact approaches Earth's gravitational binding energy
    // (~2.2e32 J) the whole planet disperses — the DEM particle burst IS the effect, and an
    // ejecta crater-curtain would be nonsense. Planet-killers (Lance, Theia, Jupiter…) eject nothing.
    if (E > 2e31) return;
    const d = vsub(imp.pos, earth.pos), dl = vlen(d) || 1;
    const normal = [d[0] / dl, d[1] / dl, d[2] / dl];
    const R = CATALOG.earth.R;
    const point = [normal[0] * R, normal[1] * R, normal[2] * R];
    const count = clamp(Math.round(250 * Math.cbrt(E / 4e22)), 250, 20000);
    this.ejecta.spawn({ point, normal, vImpact: vrel, count, color: [0.55, 0.32, 0.16], hot: 3200 });
  }

  autoFrameTarget() {
    if (!this.view.autoFrame || this.mirror.length < 2) return null;
    const e = this.mirror[0];
    let best = null, bestScore = Infinity;
    for (let j = 1; j < this.mirror.length; j++) {
      const b = this.mirror[j];
      if (b.consumed) continue;            // eaten by the Sun — nothing to frame
      const d = vsub(b.pos, e.pos);
      const dist = vlen(d);
      const gap = dist - e.R - b.R;
      const cl = -vdot(vsub(b.vel, e.vel), d) / Math.max(dist, 1e-6);
      const threat = b.touched || (cl > 1e-9 && gap / cl < 30 * 86400);
      if (!threat) continue;
      const score = cl > 0 ? gap / Math.max(cl, 1e-6) : gap;
      if (score < bestScore) { bestScore = score; best = { b, dist, gap }; }
    }
    if (!best) {
      if (this.contacts.size > 0) {
        return { pos: vadd(this.anchor.pos, e.pos), dist: clamp(e.R * 6, 8, 1e5) };
      }
      return null;
    }
    // Earth close-up only when the action is actually AT Earth — a contact far away
    // (moons colliding out at lunar distance, a grazing thief escaping) keeps the wide
    // two-shot of Earth + the threat, so off-world act-one stays on screen.
    if (best.gap < e.R * 2.2) {
      return { pos: vadd(this.anchor.pos, e.pos), dist: clamp(e.R * 6, 8, 1e5) };
    }
    const mid = vadd(e.pos, vscale(vsub(best.b.pos, e.pos), 0.5));
    return { pos: vadd(this.anchor.pos, mid), dist: clamp(best.dist * 1.5 + e.R * 4, e.R * 4.5, 4.5e5) };
  }

  focusPosFn(id) {
    // FRAME mode: the focus target IS the cinematic frame centre. Dragging/zooming then
    // orbits that centre instead of jerking the camera back to the Earth-follow target.
    if (id === 'frame') return () => {
      const ft = this.autoFrameTarget();
      return ft ? ft.pos : vadd(this.anchor.pos, this.mirror[0].pos);
    };
    if (id === 'sun') return () => [0, 0, 0];
    if (id.startsWith('planet:')) {
      const n = id.slice(7);
      return () => planetPos(n, this.jdNow());
    }
    if (id.startsWith('body:')) {
      const n = id.slice(5);
      return () => {
        const b = this.mirror.find((x) => x.name === n) || this.mirror[0];
        return vadd(this.anchor.pos, b.pos);
      };
    }
    return () => vadd(this.anchor.pos, this.mirror[0].pos);
  }

  buildScene(camera, dtWall) {
    const jd = this.jdNow();
    const focus = camera.focusPos;
    const planets = [];
    const mkModel = (pole, theta, R, posRel) => {
      const ze = vnorm(pole);
      let xe = vnorm(vsub([1, 0, 0], vscale(ze, ze[0])));
      const ye0 = vcross(ze, xe);
      const c = Math.cos(theta), s = Math.sin(theta);
      const xr = vadd(vscale(xe, c), vscale(ye0, s));
      const yr = vcross(ze, xr);
      return new Float32Array([
        xr[0] * R, xr[1] * R, xr[2] * R, 0,
        yr[0] * R, yr[1] * R, yr[2] * R, 0,
        ze[0] * R, ze[1] * R, ze[2] * R, 0,
        posRel[0], posRel[1], posRel[2], 1,
      ]);
    };

    const sunRel = vsub([0, 0, 0], focus);
    planets.push({ texKey: '2k_sun.jpg', model: mkModel([0, 0, 1], (jd - J2000) * 0.4, 695.7, sunRel), emis: 26 });
    const satPole = [0, Math.sin(0.466), Math.cos(0.466)];
    for (const n of PLANET_NAMES) {
      if (n === 'earth') continue;
      const cat = CATALOG[n];
      const pos = vsub(planetPos(n, jd), focus);
      const spin = ((jd - J2000) * 24 / (cat.rotH || 24)) * 2 * Math.PI;
      const pole = n === 'saturn' ? satPole : [0, 0, 1];
      planets.push({ texKey: cat.tex, model: mkModel(pole, spin, cat.R, pos), emis: 0 });
      if (n === 'saturn') {
        planets.push({ texKey: '2k_saturn.jpg', model: mkModel(satPole, 0, 1, pos), emis: 0, ringInner: 74.5, ringOuter: 137.0 });
      }
    }

    // particle hide-mask: bodies fully covered by an opaque shell (or Earth under its pristine
    // globe) draw NO particles — no readback-lag shimmer behind the shell, and a whole planet
    // of fill saved. The moment a shell starts fading (or the globe dissolving), particles return.
    // "show particles" (forceParticles) means EXACTLY that: no globe, no shells, every body's
    // particles visible — so the hide-mask must be empty and the shells skipped below.
    let hideMask = 0;
    if (!this.view.forceParticles && !this.settleMode) {
      if (this.dissolve < 0.02) hideMask |= 1;
      for (const b of this.mirror) {
        if (b.slot > 0 && b.shellTex && b.shellFade > 0.9 && !b.consumed) hideMask |= (1 << b.slot);
      }
    }

    // pristine-blob shells (Moon, rogue Mars/Jupiter/Earth2): textured sphere until ACTUAL
    // violence or near-touch. The sim waking early must NOT strip the shell — wake-relaxation
    // churn trips molten/boundLoss readings, so those only count after a real event.
    const realEventNow = this._heatArmed || this.contacts.size > 0;
    for (const b of this.mirror) {
      if (!b.shellTex || b.slot === 0) continue;
      let gapNear = Infinity, sumRNear = 1;
      for (const o of this.mirror) {
        if (o === b || o.consumed) continue;
        const g = vlen(vsub(o.pos, b.pos)) - o.R - b.R;
        if (g < gapNear) { gapNear = g; sumRNear = o.R + b.R; }
      }
      // same trigger as Earth's globe (gap <= 0): every body in a colliding pair converts on
      // the SAME frame — A's gap to B is B's gap to A. (sumRNear kept for future tuning.)
      const violated = b.touched || gapNear <= 0 ||
        (realEventNow && (this.moltenDelta(b.slot) > 0.01 || this.boundLoss(b.slot) > 0.02));
      if (violated && b.shellFade > 0) b.shellFade = Math.max(0, b.shellFade - dtWall / 0.9);
      if (b.shellFade > 0.012 && !this.view.forceParticles && !this.settleMode) {
        // anchor the shell to the particle blob's actual center of mass (extrapolated forward
        // from the last readback by its COM velocity) so the textured sphere never separates
        // from the particles it represents
        const sb = this.statsCache && this.statsCache.bodies[b.slot];
        let localPos = b.pos;
        // LIVE: anchor the shell to the blob's extrapolated CoM (particles can drift from the
        // mirror). FROZEN: the mirror IS the truth (particles are slaved to it via shifts) —
        // readback extrapolation there only adds 0.7 s stair-step jumps against the trail head.
        if (!this.frozen && sb && sb.count > 2) {
          const dtTag = this.simTime - (this.statsCache.simTimeTag ?? this.simTime);
          localPos = vadd(sb.com, vscale(sb.cov, dtTag));
        }
        const rel = vsub(vadd(this.anchor.pos, localPos), focus);
        planets.push({
          texKey: b.shellTex, emis: 0, fade: b.shellFade,
          model: mkModel([0, 0, 1], this.simTime * 2.66e-6, b.R * 1.06, rel),
        });
      }
    }

    // globe (pristine Earth)
    let globe = null;
    if (this.dissolve < 0.999) {
      const ePosRel = vsub(vadd(this.anchor.pos, this.mirror[0].pos), focus);
      const th = gmst(jd);
      globe = {
        model: mkModel(earthPoleEcliptic(), th, CATALOG.earth.R * 1.048, ePosRel),
        R: CATALOG.earth.R,
        dissolve: this.dissolve,
        cloudRot: (this.simTime / 86400) * 0.02 % 1,
        cityDim: this.view.cityLights ? clamp(this.pop / POP_2026, 0, 1) : 0,
        cloudOp: this.view.clouds ? 0.85 : 0,
        atmoDensity: this.view.atmo ? 1.0 : 0,
      };
    }

    // trails → renderer — rebuilt EVERY frame with a live HEAD vertex pinned to the body's
    // centre, so between sampled points the last segment stretches to the planet itself and
    // the seeded orbit + real path read as one continuous line INTO the body (no dangling gap).
    if (this.view.trails) {
      for (let i = 0; i < this.mirror.length && i < 15; i++) {
        const b = this.mirror[i];
        b.dirty = false;
        const n = b.trail.length;
        if (!n) { this.renderer.writeTrail(i, EMPTY_F32, 0); continue; }
        const head = !(b.touched || b.consumed);          // ended trails draw as-is, no head
        const nv = n + (head ? 1 : 0);
        const arr = new ArrayBuffer(nv * 16);
        const f = new Float32Array(arr), u8 = new Uint8Array(arr);
        // geo-frame trails (Moon) are stored Earth-relative — bake Earth's CURRENT helio
        // position in at upload (trails rebuild every frame, so this is free): the loop
        // travels with Earth. Helio trails bake nothing and stay fixed in space.
        const fo = b.trailFrame === 'geo'
          ? vadd(this.anchor.pos, this.mirror[0].pos) : [0, 0, 0];
        for (let k = 0; k < n; k++) {
          f[k * 4] = b.trail[k][0] + fo[0]; f[k * 4 + 1] = b.trail[k][1] + fo[1]; f[k * 4 + 2] = b.trail[k][2] + fo[2];
          const a = Math.pow(k / Math.max(nv - 1, 1), 1.5) * 0.85;
          const o = k * 16 + 12;
          u8[o] = b.color[0] * 255; u8[o + 1] = b.color[1] * 255; u8[o + 2] = b.color[2] * 255; u8[o + 3] = a * 255;
        }
        if (head) {
          const hp = vadd(this.anchor.pos, b.pos);   // heliocentric, like the trail points
          f[n * 4] = hp[0]; f[n * 4 + 1] = hp[1]; f[n * 4 + 2] = hp[2];
          const o = n * 16 + 12;
          u8[o] = b.color[0] * 255; u8[o + 1] = b.color[1] * 255; u8[o + 2] = b.color[2] * 255; u8[o + 3] = 217;
        }
        this.renderer.writeTrail(i, new Float32Array(arr), nv);
      }
    }

    const partOffset = vsub(this.anchor.pos, focus);
    const showGhosts = this._buildOrbitArcs(focus, jd);   // orbit arcs, vertices already in render-frame coords
    // ejecta lives in an Earth-centred frame; hand the renderer Earth's render-frame position
    if (this.ejecta && this.view.ejecta) this.ejecta.setRenderU(vsub(vadd(this.anchor.pos, this.mirror[0].pos), focus), 0.015);
    return {
      camera,
      timeSec: performance.now() / 1000,
      exposure: this.view.exposure,
      sunPosRel: sunRel,
      sunRel,
      coronaScale: 3200,
      partOffset,
      starBoost: this.view.starBoost,
      planets,
      globe,
      particles: true,
      belt: this.view.belt ? { d2000: jd - J2000, alpha: 0.5, focusOff: vscale(focus, -1) } : null,
      showOrbits: false,                 // static orbit ellipses replaced by the dynamic arc system
      showTrails: this.view.trails,
      showEjecta: this.view.ejecta,
      showGhosts,
      hideMask,
      lineOffsets: { orbits: vscale(focus, -1), trails: vscale(focus, -1), aim: partOffset, ghost: [0, 0, 0] },
      lineAlphas: { orbits: 0.8, trails: 1, aim: 1, ghost: 0.85 },
      bloomStrength: this.view.bloom,
      bloomThresh: 1.4,
    };
  }

  labels(camera) {
    const out = [];
    const focus = camera.focusPos;
    for (const b of this.mirror) {
      const rel = vsub(vadd(this.anchor.pos, b.pos), focus);
      out.push({ rel, name: b.name, kind: b.name === 'Earth' ? 'earth' : 'body', id: 'body:' + b.name });
    }
    out.push({ rel: vsub([0, 0, 0], focus), name: 'Sun', kind: 'planet', id: 'sun' });
    const jd = this.jdNow();
    for (const n of PLANET_NAMES) {
      if (n === 'earth') continue;
      out.push({ rel: vsub(planetPos(n, jd), focus), name: CATALOG[n].name, kind: 'planet', id: 'planet:' + n });
    }
    return out;
  }

  // ---------- custom impactor (Lab) ----------
  aimPreview(params) {
    // integrate a test particle against current mirror field; returns {pts (local), hit, tHit, vHit, E}
    const st = params.stateVector;
    let pos, vel, M;
    if (st) { pos = st.pos.slice(); vel = st.vel.slice(); M = impactorMass(params.d_km, params.recipe); }
    else {
      const v = this._latLonLaunch(params.lat, params.lon, params.speed, params.entryDeg, params.azDeg, params.range);
      pos = v.pos; vel = v.vel; M = impactorMass(params.d_km, params.recipe);
    }
    const bodies = this.mirror.map((b) => ({ pos: b.pos.slice(), vel: b.vel.slice(), M: b.M, R: b.R, name: b.name }));
    const pts = [pos.slice()];
    let hit = null, t = 0, vHit = 0;
    const tMax = 40 * 86400;
    for (let it = 0; it < 20000 && !hit && t < tMax; it++) {
      let r = 1e12;
      for (const b of bodies) r = Math.min(r, vlen(vsub(pos, b.pos)) - b.R);
      const dt = clamp(r / Math.max(vlen(vel), 1e-5) / 30, 1, 1200);
      let acc = [0, 0, 0];
      const sunL = vscale(this.anchor.pos, -1);
      const ds = vsub(sunL, pos);
      const ir = 1 / Math.max(vlen(ds), 1);
      acc = vadd(acc, vscale(ds, GM_SUN * ir * ir * ir));
      for (const b of bodies) {
        const d = vsub(b.pos, pos);
        const rr = Math.max(vlen(d), 0.5);
        acc = vadd(acc, vscale(d, G_SIM * b.M / (rr * rr * rr)));
        // advance bodies crudely (two-body w/ sun only, frozen mutual) — fine for a preview
      }
      vel = vadd(vel, vscale(acc, dt));
      pos = vadd(pos, vscale(vel, dt));
      t += dt;
      if (it % 4 === 0) pts.push(pos.slice());
      for (const b of bodies) {
        const bd = vlen(vsub(pos, b.pos));
        if (bd < b.R * 1.02) { hit = b.name; vHit = vlen(vsub(vel, b.vel)); break; }
      }
      if (vlen(pos) > 60000) break;
    }
    const E = hit ? 0.5 * (M * vHit * vHit) * 1e36 : 0;
    return { pts, hit, tHit: t, vHit, E, M };
  }

  writeAimLine(prev) {
    if (!prev || prev.pts.length < 2) { this.renderer.writeAim(new Float32Array(0), 0); return; }
    const n = Math.min(prev.pts.length, 1000);
    const arr = new ArrayBuffer(n * 16);
    const f = new Float32Array(arr), u8 = new Uint8Array(arr);
    for (let k = 0; k < n; k++) {
      f[k * 4] = prev.pts[k][0]; f[k * 4 + 1] = prev.pts[k][1]; f[k * 4 + 2] = prev.pts[k][2];
      const o = k * 16 + 12;
      const danger = prev.hit ? [255, 80, 40] : [80, 220, 255];
      u8[o] = danger[0]; u8[o + 1] = danger[1]; u8[o + 2] = danger[2]; u8[o + 3] = 200;
    }
    this.renderer.writeAim(new Float32Array(arr), n);
  }

  launchCustom(params) {
    let pos, vel;
    if (params.stateVector) { pos = params.stateVector.pos; vel = params.stateVector.vel; }
    else {
      const v = this._latLonLaunch(params.lat, params.lon, params.speed, params.entryDeg, params.azDeg, params.range);
      pos = v.pos; vel = v.vel;
    }
    this.spawnImpactor({ recipe: params.recipe, d_km: params.d_km, pos, vel, name: params.name || 'Custom impactor' });
    this.renderer.writeAim(new Float32Array(0), 0);
    this.banner = { text: `☄ ${params.name || 'IMPACTOR'} AWAY — ${(params.speed * 1000).toFixed(0)} km/s`, kind: 'warn', until: performance.now() + 5000 };
  }
}

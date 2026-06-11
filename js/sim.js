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
    this.view = {
      clouds: true, atmo: true, cityLights: true, orbits: true, trails: true,
      labels: true, belt: true, forceParticles: false, autoFrame: true,
      bloom: 0.18, exposure: 1.0, coolMul: 2500, heatMul: 1.0, starBoost: 0.5,
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
    this.paused = false;
    this.ps.reset();
    this.mirror = [];
    this.contacts = new Set();
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
    this.banner = null;
    this.headlineQueue = [];
    this.settleUntil = 1800;
    this.renderer.clearTrails();
    this.renderer.writeAim(new Float32Array(0), 0);
    this.trails = [];
    this.settleUntil = 3600;
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
      if (moonMode === 'stopped') mv = [0, 0, 0];
      if (typeof moonMode === 'object' && moonMode.velScale !== undefined) mv = vscale(mv, moonMode.velScale);
      const mb = buildBlob('moon', CATALOG.moon.R, CATALOG.moon.M, Math.round(this.quality * 0.13), {
        textures: { moon: this.texData.moon }, pole: earthPoleEcliptic(), theta: 0,
      });
      this.ps.addBlob(mb, mp, mv, 'Moon');
      this._addMirror('Moon', CATALOG.moon.M, CATALOG.moon.R, mp, mv, 1, [0.8, 0.8, 0.85]);
      const mm = this.mirror[this.mirror.length - 1];
      mm.shellTex = '2k_moon.jpg'; mm.shellFade = 1;
    }

    if (def.build) def.build(this._ctx());

    // decorative orbit lines
    const segs = PLANET_NAMES.map((n) => ({
      pts: orbitPolyline(n, this.jd0, n === 'earth' ? 360 : 220),
      color: n === 'earth' ? [0.3, 0.7, 1, 0.30] : [0.55, 0.6, 0.7, 0.20],
    }));
    this.renderer.writeOrbits(segs);

    this.focusId = def.focus || 'earth';
    this.camHint = { dist: def.camDist || 40 };
    this.statusText = 'SIMULATION READY';
  }

  _addMirror(name, M, R, pos, vel, slot, color) {
    this.mirror.push({
      name, M, R, pos: pos.slice(), vel: vel.slice(), slot, color, trail: [], trailAcc: 0,
      freezePos: pos.slice(), freezeVel: vel.slice(),
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

  qualityMax() { return Math.min(65536, Math.round(this.quality * 1.55)); }

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
    const wTarget = Math.max(0.02, this.warpUser);
    this.warpSmooth = Math.exp(lerp(Math.log(Math.max(0.02, this.warpSmooth)), Math.log(wTarget), Math.min(1, dtWall * 5)));
    if (Math.abs(Math.log(this.warpSmooth / wTarget)) < 0.01) this.warpSmooth = wTarget;

    // proximity from mirror → dt limits + cinematic auto-slow
    let minGap = 1e12, closing = 0, gapSumR = 1;
    for (let i = 0; i < this.mirror.length; i++) {
      for (let j = i + 1; j < this.mirror.length; j++) {
        const a = this.mirror[i], b = this.mirror[j];
        const d = vsub(b.pos, a.pos);
        const dist = vlen(d);
        const gap = dist - a.R - b.R;
        if (gap < minGap) {
          minGap = gap;
          gapSumR = a.R + b.R;
          const vr = vsub(b.vel, a.vel);
          closing = -vdot(vr, d) / Math.max(dist, 1e-6);
        }
      }
    }

    let warp = this.warpSmooth;
    if (this.autoSlow && closing > 1e-9 && minGap > 0 && minGap < 30 * gapSumR) {
      const tClose = minGap / closing;
      warp = Math.min(warp, Math.max(120, tClose / 5));
    }
    if (minGap < 0.5 * gapSumR) warp = Math.min(warp, 1500);   // carnage in slow-mo
    // While a freshly-spawned blob is still settling, hold warp down so it gets real wall-time
    // to relax under gravity with small timesteps — otherwise high warp takes giant steps and
    // the planet shock-heats itself white before anything even hits it.
    const settling = this.simTime < this.settleUntil;
    if (settling) warp = Math.min(warp, 500);
    if (this.warpUser < warp) warp = this.warpUser;            // user slow-mo always wins

    // disturbance assessment → freeze (rigid ride-along) for calm high-warp travel
    const es0 = this.statsCache ? this.statsCache.bodies[0] : null;
    const dSunEarth = vlen(vadd(this.mirror[0].pos, this.anchor.pos));
    const surfT0adj = es0 && this.statsBase ? 288 + (es0.surfT - (this.statsBase.surfT[0] || 288)) : 288;
    const disturbed = this.contacts.size > 0 || this.dissolveGo ||
      surfT0adj > 400 || this.moltenDelta(0) > 0.01 || this.boundLoss(0) > 0.01 ||
      dSunEarth < 42000;
    const calm = !disturbed && minGap > 12 * gapSumR;
    // aftermath sleep: when every body is mechanically at rest (per fresh readback), the
    // expensive dynamics can sleep even though "disturbed" — only heat keeps evolving
    const statsFresh = this.statsCache && (this.simTime - (this.statsCache.simTimeTag || 0)) < Math.max(1800, this.warpEff * 1.2);
    const allQuiet = !!(statsFresh && this.statsCache.bodies.every((sb) => !sb || sb.count < 3 || sb.rmsV < 0.05)) &&
      minGap > 10 * gapSumR;
    if (this._wake && this.frozen) {
      this._wake = false;
      this.frozen = false;
      for (const b of this.mirror) {
        out.shifts.push({ slot: b.slot, dp: vsub(b.pos, b.freezePos), dv: vsub(b.vel, b.freezeVel) });
      }
    }
    // Blobs can only support themselves mechanically at small timesteps: the dt-aware spring
    // clamp makes them soft at large dt, so a planet run at dt 20-45s slowly IMPLODES under its
    // own gravity, then detonates when dt shrinks near an encounter (the "white-hot Earth long
    // before impact" bug). Therefore the live sim NEVER exceeds ~2x the stable dt — any faster
    // time travel uses the frozen rigid ride-along instead. Freeze is decided by the timestep
    // the user's warp would require, not by an arbitrary warp number.
    const dtCap = clamp(this.ps.dtStable * 2, 2, 8);
    const liveDtWanted = (warp * dtWallSim) / Math.max(this.maxSub, 1);
    if (this.frozen) {
      if ((!calm && !allQuiet) || minGap < 10 * gapSumR || liveDtWanted < dtCap * 0.9) {
        this.frozen = false;
        for (const b of this.mirror) {
          out.shifts.push({ slot: b.slot, dp: vsub(b.pos, b.freezePos), dv: vsub(b.vel, b.freezeVel) });
        }
      }
    } else if (!settling && liveDtWanted > dtCap * 1.3 && (calm || allQuiet)) {
      this.frozen = true;
      for (const b of this.mirror) { b.freezePos = b.pos.slice(); b.freezeVel = b.vel.slice(); }
    }

    // relativistic impactors: shrink the timestep so a 0.1c body can't cross Earth inside
    // one step (automatic extreme slow-mo through the lance moment)
    let dtCapEff = dtCap;
    if (closing > 0.05 && minGap < 50 * gapSumR) {
      dtCapEff = clamp((0.5 * this.ps.minRp) / closing, 0.002, dtCap);
    }
    let dtSubMax = this.frozen ? 1200 : dtCapEff;
    let want = warp * dtWallSim;
    let substeps = clamp(Math.ceil(want / dtSubMax), 1, this.frozen ? 12 : this.maxSub);
    let dtSub = Math.min(want / substeps, dtSubMax);
    const simDt = dtSub * substeps;
    this.warpEff = simDt / dtWallSim;
    this.simTime += simDt;
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
        const ds = vsub(sunL, b.pos);
        const r2s = vdot(ds, ds);
        const ir = 1 / Math.sqrt(Math.max(r2s, 1));
        let acc = vsub(vscale(ds, GM_SUN * ir * ir * ir), aAnchor);
        for (const o of this.mirror) {
          if (o === b) continue;
          const d = vsub(o.pos, b.pos);
          const r2 = vdot(d, d) + 0.05;
          const irr = 1 / Math.sqrt(r2);
          acc = vadd(acc, vscale(d, G_SIM * o.M * irr * irr * irr));
        }
        b.vel = vadd(b.vel, vscale(acc, dtSub));
      }
      for (const b of this.mirror) b.pos = vadd(b.pos, vscale(b.vel, dtSub));
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
      if (es && (sT > 430 || this.moltenDelta(0) > 0.012)) this.dissolveGo = true;
      if (sT > 330) this._fireCond('hot:330');
      if (sT > 373) this._fireCond('hot:373');
      if (sT > 1500) this._fireCond('hot:1500');
    }
    this._updateDissolve(dtWall);

    // timed headlines
    for (const ev of this.events) {
      if (!ev.fired && ev.t !== undefined && this.simTime >= ev.t) this._fire(ev);
      if (!ev.fired && ev.cond && ev.cond.startsWith('after:') && this.contacts.size > 0) {
        if (!this._firstContactT) this._firstContactT = this.simTime;
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
        if (vdot(dp, dp) > 1e-10) {
          out.shifts.push({ slot: b.slot, dp, dv: [0, 0, 0] });
          b.freezePos = b.pos.slice();
        }
      }
      this.ps.writeBodyDyn(this.mirror.map((b) => ({
        slot: b.slot, vel: b.vel, pos: b.pos, spin: b.spin,
        atmT: b.slot === 0 ? (this.atmT || 288) : 0,
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
        atmT: b.slot === 0 ? (this.atmT || 288) : 0,
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
        heatGate: this.simTime > this.settleUntil ? 1 : 0,
        settleDrag: (!disturbed && this.simTime < this.settleUntil - 700) ? 1 / 240 : 0,
      });
      out.substeps = substeps;
      out.dtSub = dtSub;
      out.readback = this.frame % 20 === 0;
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
      const step = Math.max(1.5, b.R * 0.5);
      if (!last || vlen(vsub(b.pos, last)) > step) {
        b.trail.push(b.pos.slice());
        if (b.trail.length > 300) b.trail.shift();
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
      b.trail = b.trail.map((p) => vsub(p, rb.dp));
      b.dirty = true;
    }
    if (this.statsCache) for (const sb of this.statsCache.bodies) { sb.com = vsub(sb.com, rb.dp); sb.cov = vsub(sb.cov, rb.dv); }
    this.ps.epoch++;   // invalidate in-flight readbacks (old frame)
  }

  _updateDissolve(dtWall) {
    if (this.view.forceParticles) { this.dissolve = 1; return; }
    if (this.dissolveGo && this.dissolve < 1) this.dissolve = Math.min(1, this.dissolve + dtWall / 1.6);
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
        const sb = stats.bodies[b.slot];
        if (sb && sb.count > 2) { b.pos = sb.com.slice(); b.vel = sb.cov.slice(); }
      }
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
    this.killTarget = Math.max(this.killTarget, kill);
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
    const t = [{ id: 'earth', name: 'Earth', R: CATALOG.earth.R }];
    for (const b of this.mirror) if (b.name !== 'Earth') t.push({ id: 'body:' + b.name, name: b.name, R: Math.max(b.R, 0.3) });
    t.push({ id: 'sun', name: 'Sun', R: 695.7 });
    for (const n of PLANET_NAMES) if (n !== 'earth') t.push({ id: 'planet:' + n, name: CATALOG[n].name, R: CATALOG[n].R });
    return t;
  }

  // Cinematic auto-framing: keep Earth and the most imminent threat in view together.
  autoFrameTarget() {
    if (!this.view.autoFrame || this.mirror.length < 2) return null;
    const e = this.mirror[0];
    let best = null, bestScore = Infinity;
    for (let j = 1; j < this.mirror.length; j++) {
      const b = this.mirror[j];
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
    if (this.contacts.size > 0 || best.gap < e.R * 2.2) {
      return { pos: vadd(this.anchor.pos, e.pos), dist: clamp(e.R * 6, 8, 1e5) };
    }
    const mid = vadd(e.pos, vscale(vsub(best.b.pos, e.pos), 0.5));
    return { pos: vadd(this.anchor.pos, mid), dist: clamp(best.dist * 1.5 + e.R * 4, e.R * 4.5, 4.5e5) };
  }

  focusPosFn(id) {
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

    // pristine-blob shells (Moon, rogue Mars/Jupiter/Earth2): textured sphere until first violence
    for (const b of this.mirror) {
      if (!b.shellTex || b.slot === 0) continue;
      const violated = b.touched || this.moltenDelta(b.slot) > 0.01 || this.boundLoss(b.slot) > 0.02;
      if (violated && b.shellFade > 0) b.shellFade = Math.max(0, b.shellFade - dtWall / 0.9);
      if (b.shellFade > 0.012) {
        // anchor the shell to the particle blob's actual center of mass (extrapolated forward
        // from the last readback by its COM velocity) so the textured sphere never separates
        // from the particles it represents
        const sb = this.statsCache && this.statsCache.bodies[b.slot];
        let localPos = b.pos;
        if (sb && sb.count > 2) {
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

    // trails → renderer
    if (this.view.trails) {
      for (let i = 0; i < this.mirror.length && i < 15; i++) {
        const b = this.mirror[i];
        if (!b.dirty) continue;
        b.dirty = false;
        const n = b.trail.length;
        const arr = new ArrayBuffer(n * 16);
        const f = new Float32Array(arr), u8 = new Uint8Array(arr);
        for (let k = 0; k < n; k++) {
          f[k * 4] = b.trail[k][0]; f[k * 4 + 1] = b.trail[k][1]; f[k * 4 + 2] = b.trail[k][2];
          const a = Math.pow(k / Math.max(n - 1, 1), 1.5) * 0.85;
          const o = k * 16 + 12;
          u8[o] = b.color[0] * 255; u8[o + 1] = b.color[1] * 255; u8[o + 2] = b.color[2] * 255; u8[o + 3] = a * 255;
        }
        this.renderer.writeTrail(i, new Float32Array(arr), n);
      }
    }

    const partOffset = vsub(this.anchor.pos, focus);
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
      showOrbits: this.view.orbits,
      showTrails: this.view.trails,
      lineOffsets: { orbits: vscale(focus, -1), trails: partOffset, aim: partOffset },
      lineAlphas: { orbits: 0.8, trails: 1, aim: 1 },
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

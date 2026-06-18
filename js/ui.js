// ui.js — DOM shell: topbar, sidebar (scenarios/lab/view), HUD, labels, banners, ticker
import { SCENARIOS } from './scenarios.js';
import { IMPACTORS, POP_2026, MAT_TYPES } from './bodies.js';
import { PHYS } from './blob.js';
import { jdToDate, dateToJD } from './orbits.js';
import { clamp, fmtInt, vlen, vsub } from './mathx.js';

const CITIES = [
  { name: 'Yucatán (tradition)', lat: 21.4, lon: -89.5 },
  { name: 'New York', lat: 40.7, lon: -74.0 },
  { name: 'London', lat: 51.5, lon: -0.1 },
  { name: 'Tokyo', lat: 35.7, lon: 139.7 },
  { name: 'San Francisco', lat: 37.8, lon: -122.4 },
  { name: 'Sydney', lat: -33.9, lon: 151.2 },
  { name: 'Middle of Pacific', lat: 0, lon: -160 },
];
const WARP_CHIPS = [
  { label: '0.1×', v: 0.1 }, { label: '1×', v: 1 }, { label: '60×', v: 60 },
  { label: '1h/s', v: 3600 }, { label: '1d/s', v: 86400 }, { label: '1w/s', v: 604800 },
];
const WMIN = 0.02, WMAX = 2.6e6;
const LNMIN = Math.log(WMIN), LNMAX = Math.log(WMAX);

export class UI {
  constructor(sim, camera, renderer, callbacks) {
    this.sim = sim;
    this.camera = camera;
    this.renderer = renderer;
    this.cb = callbacks;   // { loadScenario(id), setQuality(n) }
    this.root = document.getElementById('ui-root');
    this.dispPop = POP_2026;
    this.labelPool = [];
    this._build();
  }

  _el(tag, cls, parent, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    (parent || this.root).appendChild(e);
    return e;
  }

  _build() {
    const r = this.root;
    r.innerHTML = '';
    this.labelsLayer = this._el('div', 'labels-layer');

    // ---- topbar ----
    const tb = this._el('div', 'topbar');
    this._el('div', 'brand', tb, '☄ DOOMSDAY <span>SIMULATOR</span>');
    const dateWrap = this._el('div', 'tb-group', tb);
    this.dateInput = this._el('input', 'date-input', dateWrap);
    this.dateInput.type = 'datetime-local';
    this.dateInput.title = 'Simulation date (UTC)';
    this.dateInput.addEventListener('change', () => {
      if (this.dateInput.value) this.cb.loadScenario(this.sim.scenarioId, this.dateInput.value);
    });
    const nowBtn = this._el('button', 'btn small', dateWrap, 'NOW');
    nowBtn.onclick = () => this.cb.loadScenario(this.sim.scenarioId, new Date().toISOString().slice(0, 16));

    // --- TIME: pause + a VALUE BUTTON that always shows the real (measured) timescale —
    // "9.1× ◂ 60×" when live physics throttles below the requested rate. Clicking opens a
    // dropdown with the preset chips, the cinematic toggle, and the fine log dial.
    const warpWrap = this._el('div', 'tb-group warps', tb);
    this._el('span', 'tb-label', warpWrap, 'TIME');
    this.pauseBtn = this._el('button', 'btn warp', warpWrap, '⏸');
    this.pauseBtn.onclick = () => { this.sim.paused = !this.sim.paused; this._syncWarpBtns(); };
    this.warpBtn = this._el('button', 'btn warp value', warpWrap, '— ▾');
    this.warpBtn.title = 'Timescale: what you are actually seeing (◂ shows the requested rate when live physics runs slower). Click to change.';
    const wPanel = this._drop(warpWrap, this.warpBtn);
    const chipRow = this._el('div', 'drop-row', wPanel);
    this.warpBtns = WARP_CHIPS.map((w) => {
      const b = this._el('button', 'btn warp chip', chipRow, w.label);
      b.onclick = () => { this.sim.paused = false; this.setWarp(w.v); wPanel.style.display = 'none'; };
      return { b, w };
    });
    this.autoSlowBtn = this._el('button', 'btn warp chip toggled', chipRow, '🎬');
    this.autoSlowBtn.title = 'Cinematic auto slow-motion on approach';
    this.autoSlowBtn.onclick = () => {
      this.sim.autoSlow = !this.sim.autoSlow;
      this.autoSlowBtn.classList.toggle('toggled', this.sim.autoSlow);
    };
    this.dial = this._el('input', 'time-dial', wPanel);
    this.dial.type = 'range'; this.dial.min = 0; this.dial.max = 1000; this.dial.step = 1;
    this.dial.title = 'Time dial — drag left for slow motion, right for time warp';
    this.dial.addEventListener('input', () => {
      this.sim.paused = false;
      this.sim.warpUser = Math.exp(LNMIN + (LNMAX - LNMIN) * (this.dial.value / 1000));
      this._syncWarpBtns();
    });
    // While the user holds the dial it's authoritative; otherwise it tracks the MEASURED rate.
    this.dial.addEventListener('pointerdown', () => { this._dialDrag = true; });
    window.addEventListener('pointerup', () => { this._dialDrag = false; });
    const wHint = this._el('div', 'drop-hint', wPanel);
    wHint.innerHTML = '<kbd>,</kbd> slower &nbsp; <kbd>.</kbd> faster &nbsp; <kbd>[</kbd> <kbd>]</kbd> ½× / 2×';

    // --- DENSITY: a VALUE BUTTON showing the real particle count; the dropdown has three
    // machine-calibrated presets (mid = the GPU profiler's pick, ⅓ and 3× around it) and the
    // fine log dial at the bottom. High counts change the sim's whole feel — keep it up top.
    const densWrap = this._el('div', 'tb-group', tb);
    this._el('span', 'tb-label', densWrap, 'DENSITY');
    const fmtQ = (n) => (n >= 1000 ? (n / 1000).toFixed(n < 10000 ? 1 : 0) + 'k' : n);
    this.densBtn = this._el('button', 'btn small value', densWrap, '— ▾');
    this.densBtn.title = 'Particle count — click to change (rebuilds the scenario)';
    const dPanel = this._drop(densWrap, this.densBtn);
    const mp = Math.max(8192, this.cb.machinePick || 16384);
    const rk = (n) => Math.round(clamp(n, 4096, 1048576) / 1000) * 1000;
    this.densTiers = [rk(mp / 3), rk(mp), rk(mp * 3)];
    const dRow = this._el('div', 'drop-row', dPanel);
    this.densBtns = this.densTiers.map((n, i) => {
      const b = this._el('button', 'btn small dens', dRow, fmtQ(n));
      b.title = `${n.toLocaleString()} particles — ${['light & fast', 'calibrated for this GPU', '3× the calibrated pick — the high-density feel'][i]}`;
      b.onclick = () => { this._setQuality(n); dPanel.style.display = 'none'; };
      return { b, n };
    });
    // fine log dial (applied on release; rebuilds bodies). Fast engine scales to 1M;
    // the reference N² engine is capped where it stays interactive.
    const isFast = !!this.sim.ps.isFast;
    const Q_MIN = 4000, Q_MAX = isFast ? 1048576 : 42000;
    const qFromDial = (v) => Math.round(Math.exp(Math.log(Q_MIN) + (Math.log(Q_MAX) - Math.log(Q_MIN)) * (v / 1000)) / 1000) * 1000;
    const dialFromQ = (q) => Math.round(1000 * (Math.log(Math.max(Q_MIN, Math.min(Q_MAX, q))) - Math.log(Q_MIN)) / (Math.log(Q_MAX) - Math.log(Q_MIN)));
    const qSlideRow = this._el('div', 'drop-row', dPanel);
    this.qDial = this._el('input', 'slider', qSlideRow);
    this.qDial.type = 'range'; this.qDial.min = 0; this.qDial.max = 1000; this.qDial.step = 1;
    this.qDial.title = isFast
      ? 'Particle count (log scale, up to 1M) — higher = finer detail, slower. Rebuilds on release.'
      : 'Particle count — the N² reference engine caps at 42k. Remove ?kernel=n2 for up to 1M.';
    const qOut = this._el('span', 'sval', qSlideRow, '');
    this.qDial.addEventListener('input', () => { qOut.textContent = fmtQ(qFromDial(parseInt(this.qDial.value))); });
    this.qDial.addEventListener('change', () => this._setQuality(qFromDial(parseInt(this.qDial.value))));
    // keep value button, dial, readout and preset highlights agreeing with sim.quality
    this._qSync = () => {
      this.qDial.value = dialFromQ(this.sim.quality);
      qOut.textContent = fmtQ(this.sim.quality);
      this.densBtn.textContent = fmtQ(this.sim.quality) + ' ▾';
      for (const { b, n } of this.densBtns) {
        b.classList.toggle('toggled', Math.abs(this.sim.quality - n) <= Math.max(1024, n * 0.06));
      }
    };
    this._qSync();

    const focusWrap = this._el('div', 'tb-group', tb);
    this._el('span', 'tb-label', focusWrap, 'FOCUS');
    this.focusSel = this._el('select', 'focus-sel', focusWrap);
    this.focusSel.addEventListener('change', () => this._focus(this.focusSel.value));
    const sunLockBtn = this._el('button', 'btn small toggled', focusWrap, '☀');
    sunLockBtn.title = 'Sun-locked view: the camera bearing follows the Sun, so the solar system doesn’t appear to orbit the focused body (orbital motion shows as the stars slowly turning). Click for the raw inertial view.';
    sunLockBtn.onclick = () => {
      this.camera.sunLock = !this.camera.sunLock;
      sunLockBtn.classList.toggle('toggled', this.camera.sunLock);
    };

    const resetBtn = this._el('button', 'btn danger', tb, 'RESET EARTH ↵');
    resetBtn.title = 'Restart the current scenario from scratch (hotkey: Return)';
    resetBtn.onclick = () => this.cb.loadScenario(this.sim.scenarioId);
    this.fpsEl = this._el('div', 'fps', tb, '');

    // ---- sidebar ----
    const sb = this._el('div', 'sidebar');
    this.sidebarEl = sb;
    const tabs = this._el('div', 'tabs', sb);
    const collapseBtn = this._el('button', 'tab collapse-btn', tabs, '◀');
    collapseBtn.title = 'Collapse sidebar (full-frame view)';
    collapseBtn.onclick = () => {
      const c = sb.classList.toggle('collapsed');
      this.root.classList.toggle('sb-collapsed', c);
      collapseBtn.textContent = c ? '▶' : '◀';
    };
    const panels = {};
    for (const t of ['SCENARIOS', 'LAB', 'VIEW', 'SYS']) {
      const btn = this._el('button', 'tab', tabs, t === 'LAB' ? '☄ LAB' : t === 'SYS' ? '⚙ SYS' : t);
      const panel = this._el('div', 'panel', sb);
      panels[t] = { btn, panel };
      btn.onclick = () => {
        for (const k in panels) {
          panels[k].btn.classList.toggle('active', k === t);
          panels[k].panel.style.display = k === t ? 'block' : 'none';
        }
      };
    }
    panels.SCENARIOS.btn.click();

    // scenarios
    for (const s of SCENARIOS) {
      const card = this._el('div', 'card', panels.SCENARIOS.panel);
      this._el('div', 'card-title', card, `${s.title} <span class="skulls">${'☠'.repeat(s.skulls)}</span>`);
      this._el('div', 'card-blurb', card, s.blurb);
      card.onclick = () => {
        this.cb.loadScenario(s.id);
        for (const c of panels.SCENARIOS.panel.children) c.classList.remove('active');
        card.classList.add('active');
      };
      if (s.id === 'peaceful') card.classList.add('active');
    }
    this._el('div', 'credits', panels.SCENARIOS.panel,
      'Textures © <a href="https://www.solarsystemscope.com/textures/" target="_blank">Solar System Scope</a> (CC-BY 4.0) · Ephemeris: JPL approx. elements');

    // lab
    const lab = panels.LAB.panel;
    this._el('div', 'lab-head', lab, 'IMPACTOR LAB');
    this.tplSel = this._el('select', 'lab-sel', lab);
    for (const t of IMPACTORS) this.tplSel.add(new Option(`${t.name} (${t.d >= 1 ? t.d + ' km' : t.d * 1000 + ' m'})`, t.id));
    this.tplSel.value = 'chicxXL';
    this.tplSel.addEventListener('change', () => this._applyTemplate());
    this.labInfo = this._el('div', 'lab-info', lab, '');

    const mkSlider = (label, min, max, val, step, fmt) => {
      const row = this._el('div', 'srow', lab);
      this._el('span', 'slabel', row, label);
      const out = this._el('span', 'sval', row, '');
      const inp = this._el('input', 'slider', row);
      inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step; inp.value = val;
      const upd = () => { out.textContent = fmt(parseFloat(inp.value)); };
      inp.addEventListener('input', () => { upd(); this._previewSoon(); });
      upd();
      return { inp, get: () => parseFloat(inp.value) };
    };
    this.sDiam = mkSlider('DIAMETER', 0, 5.15, Math.log10(120 * 10), 0.01, (v) => {
      const km = Math.pow(10, v) / 10;
      return km >= 1 ? km.toFixed(km < 20 ? 1 : 0) + ' km' : (km * 1000).toFixed(0) + ' m';
    });
    this.sSpeed = mkSlider('SPEED', 1, 120, 24, 0.5, (v) => v + ' km/s');
    this.sEntry = mkSlider('ENTRY ANGLE', 5, 90, 45, 1, (v) => v + '°');
    this.sAz = mkSlider('AZIMUTH', 0, 359, 60, 1, (v) => v + '°');
    this.sRange = mkSlider('SPAWN RANGE', 30, 3000, 120, 5, (v) => v + ' Mm');

    const cityRow = this._el('div', 'srow', lab);
    this._el('span', 'slabel', cityRow, 'TARGET');
    this.citySel = this._el('select', 'lab-sel inline', cityRow);
    for (let i = 0; i < CITIES.length; i++) this.citySel.add(new Option(CITIES[i].name, i));
    this.citySel.addEventListener('change', () => {
      const c = CITIES[this.citySel.value];
      this.latInp.value = c.lat; this.lonInp.value = c.lon;
      this._previewSoon();
    });
    const llRow = this._el('div', 'srow', lab);
    this._el('span', 'slabel', llRow, 'LAT / LON');
    this.latInp = this._el('input', 'num', llRow); this.latInp.type = 'number'; this.latInp.value = 21.4; this.latInp.step = 0.1;
    this.lonInp = this._el('input', 'num', llRow); this.lonInp.type = 'number'; this.lonInp.value = -89.5; this.lonInp.step = 0.1;
    this.latInp.addEventListener('input', () => this._previewSoon());
    this.lonInp.addEventListener('input', () => this._previewSoon());

    const adv = this._el('details', 'adv', lab);
    this._el('summary', '', adv, 'ADVANCED: state vector (Mm, Mm/s — Earth-centered ecliptic)');
    this.svInp = this._el('textarea', 'sv', adv);
    this.svInp.placeholder = 'px py pz vx vy vz\ne.g. 200 40 10  -0.02 -0.004 -0.001';

    const btnRow = this._el('div', 'btn-row', lab);
    const prevBtn = this._el('button', 'btn', btnRow, 'PREVIEW PATH');
    prevBtn.onclick = () => this._preview();
    const launchBtn = this._el('button', 'btn danger big', btnRow, '☄ LAUNCH');
    launchBtn.onclick = () => {
      this.sim.launchCustom(this._labParams());
      this.aimResult.textContent = '';
    };
    const clearBtn = this._el('button', 'btn', btnRow, 'CLEAR');
    clearBtn.onclick = () => { this.renderer.writeAim(new Float32Array(0), 0); this.aimResult.textContent = ''; };
    this.aimResult = this._el('div', 'aim-result', lab, '');
    this._applyTemplate();

    // view
    const vw = panels.VIEW.panel;
    const mkToggle = (label, key) => {
      const row = this._el('label', 'trow', vw);
      const cb = this._el('input', '', row);
      cb.type = 'checkbox'; cb.checked = this.sim.view[key];
      this._el('span', '', row, label);
      cb.addEventListener('change', () => {
        this.sim.view[key] = cb.checked;
        // turning cinematic framing back on releases any manual focus lock
        if (key === 'autoFrame' && cb.checked) this.camera.manualFocus = false;
      });
    };
    mkToggle('Clouds', 'clouds');
    mkToggle('Atmosphere', 'atmo');
    mkToggle('City lights', 'cityLights');
    mkToggle('Orbit lines', 'orbits');
    mkToggle('Trails', 'trails');
    mkToggle('Labels', 'labels');
    mkToggle('Asteroid belt', 'belt');
    mkToggle('Cinematic auto-framing', 'autoFrame');
    mkToggle('Impact ejecta', 'ejecta');
    mkToggle('Force particle view', 'forceParticles');
    const mkVS = (label, key, min, max, step) => {
      const row = this._el('div', 'srow', vw);
      this._el('span', 'slabel', row, label);
      const out = this._el('span', 'sval', row, '');
      const inp = this._el('input', 'slider', row);
      inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step; inp.value = this.sim.view[key];
      const upd = () => { out.textContent = parseFloat(inp.value).toFixed(2); this.sim.view[key] = parseFloat(inp.value); };
      inp.addEventListener('input', upd);
      upd();
    };
    mkVS('Bloom', 'bloom', 0, 1.6, 0.05);
    mkVS('Exposure', 'exposure', 0.2, 3, 0.05);
    mkVS('Heat drama', 'heatMul', 0.2, 4, 0.1);
    mkVS('Cooling rate', 'coolMul', 0, 20000, 100);
    mkVS('Star brightness', 'starBoost', 0, 1.5, 0.05);
    // (particle density moved to the topbar DENSITY dropdown)

    // ---- SYSTEM (internal physics knobs) ---------------------------------------------------
    // A huge sim of millions of tiny particles behaves differently than the real-time one, so we
    // expose the internal constants to wiggle until it looks right, then bake the values in.
    // Spawn-time knobs (material, volume) only take effect on a fresh sim → Apply rebuilds.
    // Approach/relativistic knobs read live each tick → they take effect immediately.
    const sysw = panels.SYS.panel;
    const sysHdr = this._el('div', 'sys-hdr', sysw, '⚙ SYSTEM — physics tuning');
    sysHdr.title = 'Internal sim constants. Tune, then bake the good values into the defaults. '
      + 'Material/volume changes need Apply (rebuild); approach/relativistic dials are live.';
    const mkSys = (label, get, set, min, max, step, tip) => {
      const row = this._el('div', 'srow', sysw);
      const lab = this._el('span', 'slabel', row, label); lab.title = tip;
      const out = this._el('span', 'sval', row, '');
      const inp = this._el('input', 'slider', row);
      inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step; inp.value = get(); inp.title = tip;
      const upd = () => { out.textContent = (+inp.value).toFixed(2); set(+inp.value); };
      inp.addEventListener('input', upd); upd();
    };
    const W = MAT_TYPES.WATER, ICE = MAT_TYPES.ICE, CR = MAT_TYPES.CRUST, RK = MAT_TYPES.ROCK;
    // — spawn-time (Apply rebuilds) —
    mkSys('Planet volume', () => PHYS.volFudge, (v) => (PHYS.volFudge = v), -0.3, 0.6, 0.02,
      'Scales the planet’s build size so the particle ball fills the rendered shell (it packs a few '
      + '% inside). Grows the lattice as one intact, touching ball (rp + spacing together) — '
      + 'independent of particle count, and NOT a position-stretch, so it does not loosen the '
      + 'lattice or generate the impact melt. + fills the shell, − shrinks. Rebuild.');
    mkSys('Packing compression', () => PHYS.packComp, (v) => (PHYS.packComp = v), -0.1, 0.4, 0.01,
      'Pre-loads the lattice springs (overlaps neighbours at spawn), independent of size and count. '
      + 'Dial UP until the planet holds its size in Settle (barely moves) — the outward pressure '
      + 'then balances gravity so it never collapses, which is what kills the impact melt. Too high '
      + 'and it visibly expands/flies apart on Settle (stored energy that would detonate on impact); '
      + '0 or − lets it collapse inward. Volume fills the shell, this stops the collapse — tune them '
      + 'together with Settle. Rebuild.');
    mkSys('Water cohesion', () => W.cohF, (v) => { W.cohF = v; ICE.cohF = v; }, 0, 0.5, 0.01,
      'How solid the ocean is. Low = fluid (splashes, absorbs impacts, but jellies/creeps at rest). '
      + 'High = rigid (holds shape but rings/transmits shock on impact). Rebuild.');
    mkSys('Water damping', () => W.dampZ, (v) => { W.dampZ = v; ICE.dampZ = v; }, 0.3, 0.95, 0.05,
      'Ocean shock absorption. High = soaks the impact up locally. Low = transmits the shock around '
      + 'the whole globe (omnidirectional spallation). Rebuild.');
    mkSys('Water density', () => W.densMul, (v) => { W.densMul = v; ICE.densMul = v; }, 0.3, 1.0, 0.02,
      'Ocean particle mass vs crust (crust = 0.88). Lower = buoyant ocean bulging over land; '
      + 'matched = flush sea level. Rebuild.');
    mkSys('Crust cohesion', () => CR.cohF, (v) => (CR.cohF = v), 0, 0.6, 0.01,
      'How solid the land/surface shell is. Higher resists cratering and at-rest collapse; too high '
      + 'and the surface rings like a bell on impact. Rebuild.');
    mkSys('Mantle cohesion', () => RK.cohF, (v) => (RK.cohF = v), 0, 0.6, 0.01,
      'Interior rock cohesion — what holds the planet together at rest, especially at huge particle '
      + 'counts where self-gravity wants to collapse it. Rebuild.');
    mkSys('Mantle damping', () => RK.dampZ, (v) => (RK.dampZ = v), 0.3, 0.95, 0.05,
      'Interior shock absorption. Higher soaks impact energy into heat instead of letting it '
      + 'reverberate back out as global lava eruptions. Rebuild.');
    mkSys('Interior temp', () => PHYS.interiorTempMul, (v) => (PHYS.interiorTempMul = v), 0.4, 1.2, 0.05,
      'Scales the molten core+mantle spawn temperature. The lava you see on impact is the always-hot '
      + 'interior showing through cracks — lower this for more headroom below the 2200K conduction '
      + 'threshold, so a moderate hit cracks the lid without tipping the whole planet into lava. Rebuild.');
    // — approach / relativistic (live) —
    mkSys('Sim-wake dist', () => this.sim.sys.wakeR, (v) => (this.sim.sys.wakeR = v), 1, 8, 0.5,
      'The particle COLLISION sim wakes when the surface gap drops below this × combined radii. '
      + 'Bigger = ignites earlier (safer for fast impactors, costs more compute). Live.');
    mkSys('Fine-step dist', () => this.sim.sys.fineR, (v) => (this.sim.sys.fineR = v), 4, 40, 1,
      'A fast/relativistic impactor switches to small frozen time-steps within this × combined radii '
      + 'so it cannot tunnel through Earth before the sim wakes. The Lance pre-impact dial. Live.');
    mkSys('Fine-step frac', () => this.sim.sys.fineFrac, (v) => (this.sim.sys.fineFrac = v), 0.1, 1.0, 0.05,
      'Max distance a frozen impactor advances per frame during fine-step approach, as a fraction of '
      + 'combined radii. Smaller = smoother and tunnel-proof, but more frames. Live.');
    mkSys('Slow-mo window', () => this.sim.sys.slowWin, (v) => (this.sim.sys.slowWin = v), 10, 400, 10,
      'Relativistic cinematic slow-mo engages within this many seconds of contact (so the Lance’s '
      + 'glint visibly grows on approach). Live.');
    mkSys('Slow-mo strength', () => this.sim.sys.slowDiv, (v) => (this.sim.sys.slowDiv = v), 1, 20, 1,
      'Slow-mo aggressiveness: on-screen warp ≈ time-to-contact / this. Lower = slower, more '
      + 'dramatic approach. Live.');
    const applyBtn = this._el('button', 'btn sys-apply', sysw, 'Apply — rebuild sim');
    applyBtn.title = 'Re-spawn the current scenario from scratch (normal play, shells on) so the '
      + 'spawn-time knobs above take effect. Approach/relativistic dials are already live.';
    const settleBtn = this._el('button', 'btn sys-apply', sysw, 'Settle — watch (shells off)');
    settleBtn.title = 'Rebuild with the current settings, drop the textured shells, and run the planet '
      + 'live and COOL (no heat) so you watch the lattice find its resting shape. Holds its size = '
      + 'stable; collapses inward or flies apart = bad. Heat only comes on later, on a real impact. '
      + 'Click again to stop and restore shells.';
    const setSettle = (on) => {
      this.sim.settleMode = on;
      settleBtn.classList.toggle('active', on);
      settleBtn.textContent = on ? 'Settling… (click to stop)' : 'Settle — watch (shells off)';
    };
    // Both buttons rebuild from scratch first. Apply clears settle (normal play); Settle toggles it.
    applyBtn.onclick = () => { setSettle(false); this.cb.loadScenario(this.sim.scenarioId); };
    settleBtn.onclick = () => { setSettle(!this.sim.settleMode); this.cb.loadScenario(this.sim.scenarioId); };

    // ---- HUD ----
    const hud = this._el('div', 'hud');
    this.statusEl = this._el('div', 'hud-status ok', hud, 'ALL QUIET');
    const mkStat = (label) => {
      const row = this._el('div', 'hud-row', hud);
      this._el('span', 'hud-label', row, label);
      return this._el('span', 'hud-val', row, '—');
    };
    this.popEl = this._el('div', 'hud-pop', hud, fmtInt(POP_2026));
    this._el('div', 'hud-pop-label', hud, 'HUMANS ALIVE');
    this.casEl = mkStat('CASUALTIES');
    this.tempEl = mkStat('SURFACE TEMP');
    this.atmEl = mkStat('ATMOSPHERE');
    this.massEl = mkStat('EARTH MASS');
    this.moonEl = mkStat('MOON');
    this.energyEl = mkStat('ENERGY RELEASED');

    // banner + ticker
    this.bannerEl = this._el('div', 'banner', r, '');
    this.tickerEl = this._el('div', 'ticker', r, '');
    this._tickerUntil = 0;

    // "return to auto camera" pill — shown only while the user holds manual camera control
    this.camPill = this._el('button', 'cam-pill', r, '↺ Return to auto camera <kbd>Space</kbd>');
    this.camPill.style.display = 'none';
    this.camPill.onclick = () => this.camera.returnToAuto();

    this.refreshFocusList();
    this._ready = true;
  }

  _applyTemplate() {
    const t = IMPACTORS.find((x) => x.id === this.tplSel.value);
    if (!t) return;
    this.sDiam.inp.value = Math.log10(Math.max(t.d * 10, 1));
    this.sDiam.inp.dispatchEvent(new Event('input'));
    this.sSpeed.inp.value = t.vDef;
    this.sSpeed.inp.dispatchEvent(new Event('input'));
    this.labInfo.textContent = t.blurb;
    this._previewSoon();
  }

  _labParams() {
    const t = IMPACTORS.find((x) => x.id === this.tplSel.value);
    const d_km = Math.pow(10, this.sDiam.get()) / 10;
    const p = {
      recipe: t ? t.recipe : 'rock',
      d_km,
      speed: this.sSpeed.get() / 1000,    // km/s → Mm/s
      entryDeg: this.sEntry.get(),
      azDeg: this.sAz.get(),
      lat: parseFloat(this.latInp.value) || 0,
      lon: parseFloat(this.lonInp.value) || 0,
      range: this.sRange.get(),
      name: (t ? t.name : 'Custom') + (Math.abs(d_km - (t ? t.d : 0)) > 0.01 * d_km ? ' (custom)' : ''),
    };
    const sv = this.svInp.value.trim();
    if (sv) {
      const nums = sv.split(/[\s,]+/).map(Number).filter((x) => isFinite(x));
      if (nums.length >= 6) p.stateVector = { pos: nums.slice(0, 3), vel: nums.slice(3, 6) };
    }
    return p;
  }

  _previewSoon() {
    if (!this._ready) return;   // no phantom targeting lines during UI construction
    clearTimeout(this._pt);
    this._pt = setTimeout(() => this._preview(), 250);
  }

  _preview() {
    const prev = this.sim.aimPreview(this._labParams());
    this.sim.writeAimLine(prev);
    if (prev.hit) {
      const days = prev.tHit / 86400;
      const eta = days >= 1 ? days.toFixed(1) + ' days' : (prev.tHit / 3600).toFixed(1) + ' h';
      this.aimResult.innerHTML = `<b class="hit">IMPACT: ${prev.hit}</b> · ETA ${eta} · ${(prev.vHit * 1000).toFixed(1)} km/s · ${this.sim._fmtEnergy(prev.E)}`;
    } else {
      this.aimResult.innerHTML = '<b class="miss">CLEAN MISS</b> (or escaped the neighborhood)';
    }
  }

  _focus(id, user = true) {
    const t = this.sim.focusTargets().find((x) => x.id === id) || { R: 6.4 };
    // FRAME is a mode, not a body: never set manualFocus for it, so the cinematic framing
    // keeps owning position+distance until the user actually drags/zooms — and even then
    // the orbit centre stays the frame centre (no jerk back to an Earth-follow camera).
    const isFrame = id === 'frame';
    this.camera.focusOn(this.sim.focusPosFn(id), t.R,
      { dist: isFrame ? this.sim.camHint.dist : t.R * 4.5, user: user && !isFrame, frame: isFrame });
    if (isFrame) this.camera.manual = false;     // re-picking FRAME hands control back
    this.focusSel.value = id;
  }

  refreshFocusList() {
    const cur = this.focusSel.value;
    this.focusSel.innerHTML = '';
    for (const t of this.sim.focusTargets()) this.focusSel.add(new Option(t.name, t.id));
    if ([...this.focusSel.options].some((o) => o.value === cur)) this.focusSel.value = cur;
  }

  updateCamPill() {
    const show = this.camera.canReturnToAuto();
    if (show !== this._pillShown) {
      this._pillShown = show;
      this.camPill.style.display = show ? 'flex' : 'none';
    }
  }

  setWarp(v) {
    this.sim.warpUser = Math.min(WMAX, Math.max(WMIN, v));
    this._syncWarpBtns();
  }

  // step to the next ( +1 ) or previous ( -1 ) TIME chip; snaps cleanly even when the dial has
  // left the warp between presets. Bound to the . and , keys.
  stepWarp(dir) {
    const cur = this.sim.warpUser;
    let target;
    if (dir > 0) {
      const up = WARP_CHIPS.find((w) => w.v > cur * 1.001);
      target = up ? up.v : WARP_CHIPS[WARP_CHIPS.length - 1].v;
    } else {
      const below = WARP_CHIPS.filter((w) => w.v < cur * 0.999);
      target = below.length ? below[below.length - 1].v : WARP_CHIPS[0].v;
    }
    this.sim.paused = false;
    this.setWarp(target);
  }

  // value-button dropdown: panel opens under the button, closes on outside pointerdown,
  // selecting a preset closes it, dragging a slider inside keeps it open
  _drop(wrap, btn) {
    const panel = this._el('div', 'tb-drop', wrap);
    panel.style.display = 'none';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const wasOpen = panel.style.display !== 'none';
      document.querySelectorAll('.tb-drop').forEach((p) => { p.style.display = 'none'; });
      panel.style.display = wasOpen ? 'none' : 'flex';
    });
    if (!this._dropCloser) {
      this._dropCloser = true;
      window.addEventListener('pointerdown', (ev) => {
        document.querySelectorAll('.tb-drop').forEach((p) => {
          if (!p.parentElement.contains(ev.target)) p.style.display = 'none';
        });
      });
    }
    return panel;
  }

  // single path for ALL quality changes (topbar buttons + sidebar dial): rebuild in place
  // when it fits this session's buffers, relaunch with a bigger cap when it doesn't
  _setQuality(n) {
    if (Math.round(n * 1.6) > (this.sim.ps.maxN || 65536)) {
      const u = new URL(location.href);
      u.searchParams.set('q', n);
      u.searchParams.set('scenario', this.sim.scenarioId);
      location.href = u.toString();
      return;
    }
    this.cb.setQuality(n);
    if (this._qSync) this._qSync();
  }

  _syncWarpBtns() {
    this.pauseBtn.classList.toggle('toggled', this.sim.paused);
    const w = this.sim.warpUser;
    this.dial.value = Math.round(1000 * (Math.log(w) - LNMIN) / (LNMAX - LNMIN));
    for (const { b, w: wv } of this.warpBtns) {
      b.classList.toggle('toggled', !this.sim.paused && Math.abs(Math.log(w / wv.v)) < 0.05);
    }
  }

  syncScenario() {
    this.refreshFocusList();
    // scenario load picks the default focus but leaves cinematic framing free to take over
    this._focus(this.sim.focusId === 'earth' ? 'earth' : this.sim.focusId, false);
    const d = jdToDate(this.sim.jd0);
    this.dateInput.value = d.toISOString().slice(0, 16);
    this._syncWarpBtns();
    this.camera._targetDist = this.sim.camHint.dist;
  }

  fmtWarp(w) {
    if (w === 0) return 'PAUSED';
    if (w < 0.95) return w.toFixed(2) + '× SLOW-MO';
    if (w < 90) return w.toFixed(w < 3 ? 1 : 0) + '×';
    if (w < 5400) return (w / 60).toFixed(1) + ' min/s';
    if (w < 129600) return (w / 3600).toFixed(1) + ' h/s';
    return (w / 86400).toFixed(1) + ' d/s';
  }

  update(fps) {
    // 4 Hz numeric refresh
    const now = performance.now();
    if (!this._lastNum || now - this._lastNum > 250) {
      this._lastNum = now;
      const sim = this.sim;
      this.fpsEl.textContent = fps.toFixed(0) + ' fps';
      // The slider STAYS where the user puts it (warpUser). The readout reports the measured
      // rate + state, and flags when the live physics is throttling below the dial setting.
      const physLive = !sim.frozen && sim.ps.activeN > 0;
      if (sim.paused) {
        this.warpBtn.textContent = 'PAUSED ▾';
        this.warpBtn.title = '';
      } else {
        const throttled = sim.warpEff < sim.warpUser * 0.7;
        // the button IS the readout: always the rate you're actually seeing; when live
        // physics throttles below the request, show "actual ◂ requested"
        this.warpBtn.textContent = (throttled
          ? `${this.fmtWarp(sim.warpEff)} ◂ ${this.fmtWarp(sim.warpUser)}`
          : this.fmtWarp(sim.warpEff)) + (physLive ? ' · SIM' : '') + ' ▾';
        this.warpBtn.title = (throttled ? `Dial set to ${this.fmtWarp(sim.warpUser)}; ` : '') +
          (physLive ? 'collision physics is running — time is paced to what the GPU can render.'
                    : 'orbital cruise — particle sim paused; planetary motion at the dial rate until an impactor closes in.');
      }
      const d = jdToDate(sim.jdNow());
      if (document.activeElement !== this.dateInput) this.dateInput.value = d.toISOString().slice(0, 16);

      this.dispPop += (sim.pop - this.dispPop) * 0.35;
      if (Math.abs(this.dispPop - sim.pop) < 500) this.dispPop = sim.pop;
      this.popEl.textContent = fmtInt(this.dispPop);
      this.popEl.classList.toggle('dying', sim.pop < POP_2026 * 0.98 && sim.pop > 0);
      this.popEl.classList.toggle('dead', sim.pop <= 0);
      this.casEl.textContent = fmtInt(POP_2026 - sim.pop);
      const tC = (sim.surfTDisplay || 288) - 273.15;
      this.tempEl.textContent = `${tC.toFixed(1)} °C (${(sim.surfTDisplay || 288).toFixed(0)} K)`;
      this.tempEl.style.color = tC > 50 ? 'var(--danger)' : tC > 25 ? 'var(--accent)' : '';
      const aC = (sim.atmT || 288) - 273.15;
      this.atmEl.textContent = aC > 200 ? `${aC.toFixed(0)} °C ☠` : `${aC.toFixed(1)} °C`;
      this.atmEl.style.color = aC > 60 ? 'var(--danger)' : aC > 25 ? 'var(--accent)' : '';
      const ml = sim.massLossDisplay || 0;
      this.massEl.textContent = (100 * (1 - ml)).toFixed(ml > 0.001 ? 2 : 0) + ' %';
      this.moonEl.textContent = sim.moonStatus();
      this.energyEl.textContent = sim.energyDisplay > 1e12 ? sim._fmtEnergy(sim.energyDisplay) : '—';

      const k = sim.killTarget;
      let status = 'ALL QUIET', cls = 'ok';
      if (sim.pop <= 0) { status = '☠ EARTH STERILIZED ☠'; cls = 'dead'; }
      else if (k > 0.9) { status = 'EXTINCTION LEVEL EVENT'; cls = 'dead'; }
      else if (k > 0.5) { status = 'MASS EXTINCTION IN PROGRESS'; cls = 'danger'; }
      else if (k > 0.08) { status = 'GLOBAL CATASTROPHE'; cls = 'danger'; }
      else if (k > 0.002) { status = 'REGIONAL CATASTROPHE'; cls = 'warn'; }
      else if (sim.dissolveGo) { status = 'PLANETARY EMERGENCY'; cls = 'warn'; }
      this.statusEl.textContent = status;
      this.statusEl.className = 'hud-status ' + cls;

      // banner
      if (sim.banner && now < sim.banner.until) {
        this.bannerEl.textContent = sim.banner.text;
        this.bannerEl.className = 'banner show ' + (sim.banner.kind || '');
      } else {
        this.bannerEl.className = 'banner';
      }
      // ticker
      if (now > this._tickerUntil && sim.headlineQueue.length) {
        this.tickerEl.textContent = '📰 ' + sim.headlineQueue.shift();
        this.tickerEl.classList.add('show');
        this._tickerUntil = now + 7000;
      } else if (now > this._tickerUntil) {
        this.tickerEl.classList.remove('show');
      }
    }

    // labels every frame
    const labels = this.sim.view.labels ? this.sim.labels(this.camera) : [];
    while (this.labelPool.length < labels.length) {
      const el = document.createElement('div');
      el.className = 'blabel';
      this.labelsLayer.appendChild(el);
      this.labelPool.push(el);
    }
    for (let i = 0; i < this.labelPool.length; i++) {
      const el = this.labelPool[i];
      if (i >= labels.length) { el.style.display = 'none'; continue; }
      const L = labels[i];
      const s = this.renderer.projectToScreen(L.rel);
      const dist = vlen(L.rel);
      // no distance-based hiding — a near-zero threshold jitters and strobes the label
      if (!s || s[0] < -50 || s[0] > innerWidth + 50 || s[1] < 0 || s[1] > innerHeight) {
        el.style.display = 'none'; continue;
      }
      el.style.display = 'block';
      el.style.left = s[0] + 'px';
      el.style.top = s[1] + 'px';
      if (el._txt !== L.name + L.kind) {
        el._txt = L.name + L.kind;
        el.textContent = L.name;
        el.className = 'blabel ' + L.kind;
        el.onclick = () => this._focus(L.id);
      }
    }
  }
}

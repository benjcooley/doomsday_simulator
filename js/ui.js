// ui.js — DOM shell: topbar, sidebar (scenarios/lab/view), HUD, labels, banners, ticker
import { SCENARIOS } from './scenarios.js';
import { IMPACTORS, POP_2026 } from './bodies.js';
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

    // --- TIME DIAL: pause + log slider (0.02× slow-mo … 1 month/s) + preset chips ---
    const warpWrap = this._el('div', 'tb-group warps', tb);
    this.pauseBtn = this._el('button', 'btn warp', warpWrap, '⏸');
    this.pauseBtn.onclick = () => { this.sim.paused = !this.sim.paused; this._syncWarpBtns(); };
    this.dial = this._el('input', 'time-dial', warpWrap);
    this.dial.type = 'range'; this.dial.min = 0; this.dial.max = 1000; this.dial.step = 1;
    this.dial.title = 'Time dial — drag left for slow motion, right for time warp';
    this.dial.addEventListener('input', () => {
      this.sim.paused = false;
      this.sim.warpUser = Math.exp(LNMIN + (LNMAX - LNMIN) * (this.dial.value / 1000));
      this._syncWarpBtns();
    });
    this.warpBtns = WARP_CHIPS.map((w) => {
      const b = this._el('button', 'btn warp chip', warpWrap, w.label);
      b.onclick = () => { this.sim.paused = false; this.setWarp(w.v); };
      return { b, w };
    });
    this.autoSlowBtn = this._el('button', 'btn warp toggled', warpWrap, '🎬');
    this.autoSlowBtn.title = 'Cinematic auto slow-motion on approach';
    this.autoSlowBtn.onclick = () => {
      this.sim.autoSlow = !this.sim.autoSlow;
      this.autoSlowBtn.classList.toggle('toggled', this.sim.autoSlow);
    };
    this.warpReadout = this._el('div', 'warp-readout', tb, '—');

    const focusWrap = this._el('div', 'tb-group', tb);
    this._el('span', 'tb-label', focusWrap, 'FOCUS');
    this.focusSel = this._el('select', 'focus-sel', focusWrap);
    this.focusSel.addEventListener('change', () => this._focus(this.focusSel.value));

    const resetBtn = this._el('button', 'btn danger', tb, 'RESET EARTH');
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
    for (const t of ['SCENARIOS', 'LAB', 'VIEW']) {
      const btn = this._el('button', 'tab', tabs, t === 'LAB' ? '☄ LAB' : t);
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
      cb.addEventListener('change', () => { this.sim.view[key] = cb.checked; });
    };
    mkToggle('Clouds', 'clouds');
    mkToggle('Atmosphere', 'atmo');
    mkToggle('City lights', 'cityLights');
    mkToggle('Orbit lines', 'orbits');
    mkToggle('Trails', 'trails');
    mkToggle('Labels', 'labels');
    mkToggle('Asteroid belt', 'belt');
    mkToggle('Cinematic auto-framing', 'autoFrame');
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
    const qRow = this._el('div', 'srow', vw);
    this._el('span', 'slabel', qRow, 'PARTICLES');
    this.qSel = this._el('select', 'lab-sel inline', qRow);
    for (const [label, n] of [['Low (8k)', 8192], ['Medium (16k)', 16384], ['High (32k)', 32768], ['Ultra (42k)', 42000]]) {
      this.qSel.add(new Option(label, n));
    }
    this.qSel.value = String(this.sim.quality);
    this.qSel.addEventListener('change', () => this.cb.setQuality(parseInt(this.qSel.value)));

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

  _focus(id) {
    const t = this.sim.focusTargets().find((x) => x.id === id) || { R: 6.4 };
    this.camera.focusOn(this.sim.focusPosFn(id), t.R, { dist: Math.max(t.R * 4.5, this.camera._targetDist * 0.0 + t.R * 4.5) });
    this.focusSel.value = id;
  }

  refreshFocusList() {
    const cur = this.focusSel.value;
    this.focusSel.innerHTML = '';
    for (const t of this.sim.focusTargets()) this.focusSel.add(new Option(t.name, t.id));
    if ([...this.focusSel.options].some((o) => o.value === cur)) this.focusSel.value = cur;
  }

  setWarp(v) {
    this.sim.warpUser = Math.min(WMAX, Math.max(WMIN, v));
    this._syncWarpBtns();
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
    this._focus(this.sim.focusId === 'earth' ? 'earth' : this.sim.focusId);
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
      if (sim.paused) {
        this.warpReadout.textContent = 'PAUSED';
        this.warpReadout.title = '';
      } else if (sim.warpEff < sim.warpUser * 0.72) {
        // be honest about capping — the dial is set higher than physics can deliver
        this.warpReadout.textContent = this.fmtWarp(sim.warpEff) + ' ⛔' ;
        this.warpReadout.title = `Dial asks ${this.fmtWarp(sim.warpUser)} but the live physics caps at ~${this.fmtWarp(sim.warpEff)} on this GPU ` +
          '(stability needs small timesteps near/after contact). Calm orbital phases run far faster via rigid ride-along.';
      } else {
        this.warpReadout.textContent = this.fmtWarp(sim.warpEff);
        this.warpReadout.title = '';
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

// main.js — boot + frame loop
import { GPU } from './gpu.js';
import { ParticleSystem } from './particles.js';
import { FastParticleSystem } from './particles_fast.js';
import { EjectaSystem } from './ejecta.js';
import { Renderer } from './renderer.js';
import { OrbitCamera } from './camera.js';
import { Sim } from './sim.js';
import { UI } from './ui.js';
import { clamp, vlen, vsub } from './mathx.js';

const boot = document.getElementById('boot-status');
const overlay = document.getElementById('boot-overlay');
function bootMsg(t) { if (boot) boot.textContent = t; }

function fatal(msg, detail) {
  bootMsg('');
  overlay.style.display = 'flex';
  overlay.style.opacity = '1';
  overlay.innerHTML = `<div class="boot-title">☄ DOOMSDAY SIMULATOR</div>
    <div class="boot-sub" style="color:#ff5a4a;max-width:560px;text-align:center;line-height:1.6">${msg}</div>` +
    (detail ? `<div class="boot-sub" style="color:#7d8b9e;max-width:520px;text-align:center;font-size:11px;line-height:1.6;margin-top:4px">${detail}</div>` : '');
}

// Benchmark the REAL physics step (whichever engine is active) to pick the default density.
// N² engine: cost ∝ N², one timing fixes the curve. FAST engine: cost ≈ c0 + b·N, so we
// measure at two sizes and fit the line — its ceiling is far higher.
async function profileGPU(gpu, ps, isFast) {
  const { buildBlob } = await import('./blob.js');
  const measure = async (N) => {
    ps.reset();
    ps.addBlob(buildBlob('rock', 6.0, 5.0, N, {}), [0, 0, 0], [0, 0, 0], 'probe');
    ps.writeParams({ dt: 1, settleBoost: 1, sunPos: [1e7, 0, 0], gmSun: 0, coolMul: 0,
      heatMul: 0, time: 0, solarLum: 1, heatGate: 0, settleDrag: 0 });
    const burst = () => { const e = gpu.device.createCommandEncoder(); ps.step(e, 1); gpu.device.queue.submit([e.finish()]); };
    for (let w = 0; w < 3; w++) burst();
    await gpu.device.queue.onSubmittedWorkDone();
    const K = 12, t0 = performance.now();
    for (let i = 0; i < K; i++) burst();
    await gpu.device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / K;
  };
  const LIVE_SUB = 4, budgetMs = 6.5;
  try {
    if (isFast) {
      const t8 = await measure(8192);
      const t32 = await measure(32768);
      ps.reset();
      const b = Math.max((t32 - t8) / (32768 - 8192), 1e-7);   // ms per particle
      const c0 = Math.max(t8 - b * 8192, 0);                    // fixed per-substep cost
      const ideal = (budgetMs / LIVE_SUB - c0) / b;
      let suggested = 16384;
      for (const t of [16384, 32768, 65536, 131072, 262144, 524288]) if (t <= ideal) suggested = t;
      const tooWeak = (c0 + b * 16384) * LIVE_SUB > 26;
      return { engine: 'fast', t8: +t8.toFixed(2), t32: +t32.toFixed(2), ideal: Math.round(ideal), suggested, tooWeak, perStep: +t8.toFixed(2), N: 8192 };
    }
    const N = 8192;
    const perStep = await measure(N);
    ps.reset();
    const c = perStep / (N * N);
    const ideal = Math.sqrt(budgetMs / (LIVE_SUB * c));
    let suggested = 8192;
    for (const t of [8192, 16384, 32768, 42000]) if (t <= ideal) suggested = t;
    const tooWeak = (LIVE_SUB * c * 7000 * 7000 > 18) || perStep > 60;
    return { engine: 'n2', N, perStep: +perStep.toFixed(2), ideal: Math.round(ideal), suggested, tooWeak };
  } catch (e) {
    ps.reset();
    return { engine: isFast ? 'fast' : 'n2', perStep: 0, ideal: 0, suggested: 16384, tooWeak: false, note: 'probe-failed:' + e.message };
  }
}

async function start() {
  try {
    if (!navigator.gpu) {
      if (!window.isSecureContext) {
        // browsers hide navigator.gpu entirely on insecure (http) pages
        fatal('WebGPU requires a secure connection (HTTPS).',
          `Reload via https://${location.host}${location.pathname} — WebGPU is hidden on plain http even when the browser supports it.`);
      } else {
        fatal('WebGPU is not available in this browser.',
          'DOOMSDAY needs WebGPU. Use a recent Chrome, Edge, or Arc — or Safari 26+ on macOS — with hardware acceleration enabled.');
      }
      return;
    }
    const url = new URL(location.href);
    const qTiers = { low: 8192, med: 16384, high: 32768, ultra: 65536, mega: 131072, insane: 262144, ludicrous: 524288 };
    const qRaw = url.searchParams.get('q');
    // null ?q= MUST yield null (let the profiler pick) — the old fallback chain collapsed
    // "no param" to 4096, silently overriding the profiler for every visitor
    const qOverride = qRaw === null ? null
      : (qTiers[qRaw] || Math.min(1048576, Math.max(4096, parseInt(qRaw) || 16384)));

    bootMsg('acquiring GPU…');
    let gpu;
    try {
      gpu = await new GPU().init(document.getElementById('gpu-canvas'));
    } catch (e) {
      fatal('Could not initialize WebGPU on this machine.',
        (e && e.message ? e.message + ' — ' : '') +
        'Use Chrome, Edge, or Arc (Safari 26+ on macOS), enable hardware acceleration, and prefer a machine with a dedicated GPU.');
      return;
    }

    bootMsg('decoding planetary surfaces…');
    const [earthDay, moon, mars, jupiter, venus, pluto] = await Promise.all([
      gpu.imageDataViaGPU('assets/tex/2k_earth_daymap.jpg', 1024, 512),
      gpu.imageDataViaGPU('assets/tex/2k_moon.jpg', 512, 256),
      gpu.imageDataViaGPU('assets/tex/2k_mars.jpg', 512, 256),
      gpu.imageDataViaGPU('assets/tex/2k_jupiter.jpg', 512, 256),
      gpu.imageDataViaGPU('assets/tex/2k_venus_atmosphere.jpg', 512, 256),
      gpu.imageDataViaGPU('assets/tex/2k_pluto.jpg', 512, 256),
    ]);
    const texData = { earthDay, moon, mars, jupiter, venus, pluto, earth2: earthDay };
    // diagnose the in-browser decode: sample known geography (Pacific should be blue, Sahara tan…)
    {
      const { sampleEquirect } = await import('./gpu.js');
      const D = Math.PI / 180;
      const probes = {
        pacific: [-160, 0], atlantic: [-30, 0], sahara: [10, 23],
        amazon: [-60, -5], antarctica: [0, -80], himalaya: [85, 30],
      };
      window.__texProbe = Object.fromEntries(Object.entries(probes).map(([k, [lo, la]]) =>
        [k, sampleEquirect(earthDay, lo * D, la * D).map((x) => +x.toFixed(2))]));
      window.__texProbe.isNull = !earthDay;
      console.log('texProbe', JSON.stringify(window.__texProbe));
    }

    bootMsg('compiling physics kernels…');
    // The FAST grid engine (validated in test/ stages 0-8 + in-browser A/B) is the DEFAULT.
    // ?kernel=n2 selects the original O(N²) engine as a reference/escape hatch.
    const useFast = url.searchParams.get('kernel') !== 'n2';
    // buffers allocate to this cap — big requests get headroom, small machines stay lean
    const capN = Math.max(262144, Math.ceil((qOverride || 0) * 1.7));
    const ps = await new (useFast ? FastParticleSystem : ParticleSystem)().init(gpu, capN);
    console.log(`physics engine: ${useFast ? 'FAST (grid+monopole, default)' : 'N² (reference)'}`);

    bootMsg('profiling GPU…');
    const prof = await profileGPU(gpu, ps, useFast);
    console.log('GPU profile', JSON.stringify(prof));
    if (prof.tooWeak && !qOverride) {
      fatal('This GPU is too weak for the collision physics.',
        `One physics step on ${prof.N.toLocaleString()} particles took ${prof.perStep} ms — even the lowest density would stutter badly. ` +
        'The orbital view would be fine, but the smashing won’t. Append ?q=low to force it, or use a machine with a dedicated GPU.');
      return;
    }
    const qParam = qOverride || prof.suggested;

    bootMsg('building render pipelines…');
    // strong GPUs (profiler suggests medium density or better, and not weak) get the 8K skybox
    const sky8k = !prof.tooWeak && (qOverride ? qOverride >= 16384 : prof.suggested >= 16384);
    const renderer = await new Renderer().init(gpu, ps, { sky8k });

    bootMsg('seeding impact ejecta…');
    const ejecta = await new EjectaSystem().init(gpu);
    renderer.ejecta = ejecta;

    bootMsg('assembling solar system…');
    const camera = new OrbitCamera(gpu.canvas);
    const sim = new Sim(gpu, ps, renderer, texData, qParam);
    sim.ejecta = ejecta;

    const ui = new UI(sim, camera, renderer, {
      loadScenario(id, date) {
        sim.loadScenario(id, date);
        ui.syncScenario();
      },
      setQuality(n) {
        sim.quality = n;
        sim.loadScenario(sim.scenarioId);
        ui.syncScenario();
      },
    });

    const startScenario = url.searchParams.get('scenario') || 'peaceful';
    sim.loadScenario(startScenario);
    ui.syncScenario();

    new ResizeObserver(() => renderer.resize()).observe(gpu.canvas);
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') {
        // when you've grabbed the camera, Space hands it back to the auto view; otherwise pause
        if (camera.canReturnToAuto()) { camera.returnToAuto(); }
        else { sim.paused = !sim.paused; ui._syncWarpBtns(); }
        e.preventDefault();
      }
      if (e.key === '[') { ui.setWarp(sim.warpUser / 2); }
      if (e.key === ']') { ui.setWarp(sim.warpUser * 2); }
      if (e.key === 'h' || e.key === 'H') {
        const r = document.getElementById('ui-root');
        r.style.display = r.style.display === 'none' ? '' : 'none';
      }
    });
    if (url.searchParams.get('hud') === '0') document.getElementById('ui-root').style.display = 'none';

    // ---- debug helpers (used for automated visual testing) ----
    const film = { active: false };
    window.__film = (shots = 6, hoursStep = 2, scale = 0.24) => new Promise((res) => {
      document.getElementById('film-overlay')?.remove();
      const w = Math.round(gpu.canvas.width * scale), h = Math.round(gpu.canvas.height * scale);
      const cols = Math.ceil(Math.sqrt(shots));
      const c = document.createElement('canvas');
      c.width = cols * w; c.height = Math.ceil(shots / cols) * h;
      Object.assign(film, {
        ctx: c.getContext('2d'), canvas: c, shots, idx: 0,
        stepS: hoursStep * 3600, next: sim.simTime, w, h, cols, active: true, done: res,
      });
    });
    window.__traj = () => {
      const rows = sim.mirror.map((b) => ({
        name: b.name,
        x: +b.pos[0].toFixed(1), y: +b.pos[1].toFixed(1), z: +b.pos[2].toFixed(1),
        speed_kms: +(vlen(b.vel) * 1000).toFixed(2),
        distEarth_Mm: +vlen(vsub(b.pos, sim.mirror[0].pos)).toFixed(1),
      }));
      console.table(rows);
      return {
        simDays: +(sim.simTime / 86400).toFixed(3), warpEff: Math.round(sim.warpEff * 100) / 100,
        frozen: sim.frozen, maxSub: sim.maxSub, fps: Math.round(fps),
        pop: sim.pop, killTarget: +sim.killTarget.toFixed(4),
        contacts: [...sim.contacts], dissolve: +sim.dissolve.toFixed(2), rows,
      };
    };
    window.__sim = sim;
    window.__errs = [];
    window.addEventListener('error', (e) => {
      window.__errs.push(String(e.message).slice(0, 200));
      if (window.__errs.length > 6) window.__errs.shift();
    });
    window.addEventListener('unhandledrejection', (e) => {
      window.__errs.push('rejection: ' + String(e.reason).slice(0, 180));
      if (window.__errs.length > 6) window.__errs.shift();
    });
    window.__state = () => ({
      errs: window.__errs.slice(),
      simDays: +(sim.simTime / 86400).toFixed(4), warpEff: Math.round(sim.warpEff * 100) / 100,
      warpUser: Math.round(sim.warpUser * 100) / 100, paused: sim.paused,
      cmdAck: localStorage.getItem('dd_ack'),
      frozen: sim.frozen, maxSub: sim.maxSub, fps: Math.round(fps),
      gate: { minGapR: +(sim._minGapR ?? 0).toFixed(2), wake: sim._wakeR, sleep: sim._sleepR,
        settling: !!sim._settling, disturbed: !!sim.disturbedNow },
      pop: Math.round(sim.pop), killTarget: +sim.killTarget.toFixed(4),
      livLand: +(sim.livFrac ?? 1).toFixed(4),
      atmT: Math.round(sim.atmT || 288), kDust: +(sim.kDust || 0).toFixed(3),
      crustLiv: sim.statsCache?.crustLiv ?? null, crustTot: sim.statsCache?.crustTot ?? null,
      surfT: Math.round(sim.surfTDisplay || 288), massLoss: +(sim.massLossDisplay || 0).toFixed(4),
      energyJ: sim.energyDisplay ? sim.energyDisplay.toExponential(2) : '0',
      contacts: [...sim.contacts], dissolve: +sim.dissolve.toFixed(2),
      scenario: sim.scenarioId, dist: Math.round(camera.dist),
      engine: ps.isFast ? 'fast' : 'n2',
      bodies: sim.mirror.map((b) => ({
        n: b.name, d: +vlen(vsub(b.pos, sim.mirror[0].pos)).toFixed(1),
        v: +(vlen(b.vel) * 1000).toFixed(2), shell: b.shellFade ?? null,
        rms: +(sim.statsCache?.bodies[b.slot]?.rmsV ?? 0).toFixed(3),
        maxV: +(sim.statsCache?.bodies[b.slot]?.maxV ?? 0).toFixed(2),
        fastN: sim.statsCache?.bodies[b.slot]?.fastN ?? 0,
        nanN: sim.statsCache?.bodies[b.slot]?.nanN ?? 0,
      })),
      matTemps: sim.statsCache?.matTemps || null,
      tHist: sim.statsCache?.tHist || null,
      texProbe: window.__texProbe || null,
      dbg: sim.statsCache?.dbg ? {
        idx: sim.statsCache.dbgIdx, hotT: Math.round(sim.statsCache.hotT || 0),
        contacts: sim.statsCache.dbg[0], minVn_kms: +(sim.statsCache.dbg[1] * 1000).toFixed(2),
        maxPen: +sim.statsCache.dbg[2].toFixed(3), maxHyp: +sim.statsCache.dbg[3].toFixed(3),
        dvContact_kms: sim.statsCache.dbg.slice(4, 7).map((x) => +(x * 1000).toFixed(3)),
        dvLen_kms: +(sim.statsCache.dbg[7] * 1000).toFixed(3),
        accDt_kms: sim.statsCache.dbg.slice(8, 11).map((x) => +(x * 1000).toFixed(3)),
        heat: +sim.statsCache.dbg[11].toFixed(1),
        vOut_kms: sim.statsCache.dbg.slice(12, 15).map((x) => +(x * 1000).toFixed(3)),
        T: Math.round(sim.statsCache.dbg[15]),
        dt: +sim.statsCache.dbg[16].toFixed(2), mi: sim.statsCache.dbg[17],
        radI: +sim.statsCache.dbg[18].toFixed(3), kI: sim.statsCache.dbg[19],
        vIn_kms: sim.statsCache.dbg.slice(20, 23).map((x) => +(x * 1000).toFixed(3)),
      } : null,
    });
    // command bridge: eval runs in a hidden twin context, so the LIVE tab polls
    // localStorage for test commands and publishes its state back.
    // dev telemetry is OPT-IN: add ?dev=1 to enable state posting + remote commands
    const devMode = url.searchParams.get('dev') === '1';
    let lastCmdId = localStorage.getItem('dd_ack') || '';
    const pollBridge = () => {
      try {
        localStorage.setItem('dd_state', JSON.stringify(window.__state()));
        // dev telemetry: POST state to the local server so the build loop can read it
        fetch('/state', { method: 'POST', body: localStorage.getItem('dd_state') }).catch(() => {});
        // dev command channel: the build loop can drive the sim by writing .dd_cmd.json
        fetch('/.dd_cmd.json?' + Date.now()).then((r) => (r.ok ? r.json() : null)).then((c) => {
          if (c && String(c.id) !== lastCmdId) {
            lastCmdId = String(c.id);
            localStorage.setItem('dd_ack', lastCmdId);
            if (c.cmd === 'scenario') { sim.loadScenario(c.args[0]); ui.syncScenario(); }
            else if (c.cmd === 'warp') ui.setWarp(c.args[0]);
            else if (c.cmd === 'pause') { sim.paused = !!c.args[0]; ui._syncWarpBtns(); }
            else if (c.cmd === 'reload') location.reload();
            else if (c.cmd === 'view') Object.assign(sim.view, c.args[0]);
            else if (c.cmd === 'film') window.__film(...c.args);
          }
        }).catch(() => {});
        const raw = localStorage.getItem('dd_cmd');
        if (raw) {
          const c = JSON.parse(raw);
          if (String(c.id) !== lastCmdId) {
            lastCmdId = String(c.id);
            localStorage.setItem('dd_ack', lastCmdId);
            if (c.cmd === 'scenario') { sim.loadScenario(c.args[0]); ui.syncScenario(); }
            else if (c.cmd === 'warp') ui.setWarp(c.args[0]);
            else if (c.cmd === 'pause') { sim.paused = !!c.args[0]; ui._syncWarpBtns(); }
            else if (c.cmd === 'film') window.__film(...c.args);
            else if (c.cmd === 'hud') document.getElementById('ui-root').style.display = c.args[0] ? '' : 'none';
            else if (c.cmd === 'view') Object.assign(sim.view, c.args[0]);
            else if (c.cmd === 'closefilm') document.getElementById('film-overlay')?.remove();
            else if (c.cmd === 'reload') location.reload();
            else if (c.cmd === 'nav') location.href = c.args[0];
          }
        }
      } catch (e) { /* storage may be unavailable; bridge is best-effort */ }
    };
    if (devMode) setInterval(() => { if (!document.hidden) pollBridge(); }, 400);

    overlay.style.opacity = '0';
    setTimeout(() => { overlay.style.display = 'none'; }, 700);

    let last = performance.now();
    let fps = 60;
    let frameCount = 0;

    function frame() {
      const now = performance.now();
      const dtWall = clamp((now - last) / 1000, 0.001, 0.1);
      last = now;
      fps += (1 / Math.max(dtWall, 1e-4) - fps) * 0.06;
      frameCount++;

      try {
        gpu.frameId = frameCount;
        sim.fps = fps;
        const plan = sim.tick(dtWall);
        camera.autoEnabled = sim.view.autoFrame;
        camera.autoTarget = sim.autoFrameTarget();
        camera.update(dtWall);
        ui.updateCamPill();

        // compute pass
        const enc = gpu.device.createCommandEncoder();
        if (plan.rebase) ps.rebase(enc, plan.rebase.dp, plan.rebase.dv);
        for (const s of plan.shifts) ps.rebase(enc, s.dp, s.dv, s.slot, s.rot);
        if (plan.substeps > 0) ps.step(enc, plan.substeps);
        if (plan.thermal) ps.thermalStep(enc);
        if (plan.readback) ps.requestReadback(enc, sim.simTime);
        if (sim.view.ejecta) { try { ejecta.step(enc, sim._lastSimDt || 0, sim.view.coolMul / 2500); } catch (e) { console.error('ejecta step', e); } }
        gpu.device.queue.submit([enc.finish()]);
        ps.afterSubmit();

        const stats = ps.pollStats();
        if (stats) sim.applyStats(stats);

        // render
        const scene = sim.buildScene(camera, dtWall);
        const rEnc = renderer.render(scene);
        gpu.device.queue.submit([rEnc.finish()]);

        ui.update(fps);

        if (film.active && sim.simTime >= film.next) {
          const x = (film.idx % film.cols) * film.w, y = Math.floor(film.idx / film.cols) * film.h;
          film.ctx.drawImage(gpu.canvas, x, y, film.w, film.h);
          film.ctx.fillStyle = '#ff7a1a';
          film.ctx.font = '12px monospace';
          film.ctx.fillText('T+' + (sim.simTime / 3600).toFixed(1) + 'h', x + 5, y + 14);
          film.ctx.strokeStyle = 'rgba(255,122,26,0.4)';
          film.ctx.strokeRect(x + 0.5, y + 0.5, film.w - 1, film.h - 1);
          film.idx++;
          film.next = sim.simTime + film.stepS;
          if (film.idx >= film.shots) {
            film.active = false;
            const div = document.createElement('div');
            div.id = 'film-overlay';
            div.style.cssText = 'position:fixed;inset:0;z-index:300;background:rgba(0,0,0,.93);display:flex;align-items:center;justify-content:center;cursor:pointer';
            film.canvas.style.cssText = 'max-width:97vw;max-height:97vh';
            div.appendChild(film.canvas);
            div.onclick = () => div.remove();
            document.body.appendChild(div);
            film.done?.('film ready: ' + film.idx + ' shots');
          }
        }
      } catch (err) {
        console.error('frame error:', err);
        if (frameCount < 10) { fatal('Startup error: ' + err.message); return; }
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    fatal('Failed to start: ' + err.message);
  }
}

start();

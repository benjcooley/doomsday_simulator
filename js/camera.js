// camera.js — orbit camera. ONE focal point; smoothing is framerate-INDEPENDENT
// (1 - exp(-rate*dt), exact at any framerate). Two modes:
//   AUTO   — follows the focused body, or the sim's cinematic framing target.
//   MANUAL — the moment you drag/zoom, the camera is yours and STAYS yours (it only keeps
//            the focused body centered as it moves). It never grabs control back on a timer.
//            Press Space (or click the pill) to return to the auto view.
import { clamp, vlerp, vadd, vsub, vlen } from './mathx.js';

export class OrbitCamera {
  constructor(canvas) {
    this.canvas = canvas;
    this.yaw = 0.6;
    this.pitch = 0.32;
    this.dist = 40;                 // smoothed distance actually used to render
    this._targetDist = 40;          // where zoom wants to be
    this.minDist = 8;
    this.maxDist = 8e6;
    this.fov = 50 * Math.PI / 180;
    this.focusPos = [0, 0, 0];      // heliocentric Mm

    this.getFocusPos = null;        // () => helio pos of the focused body
    this.autoTarget = null;         // {pos, dist} from sim's cinematic framing (set each frame)
    this.autoEnabled = false;       // sim.view.autoFrame (set each frame)
    this.manualFocus = false;       // a user focus pick overrides cinematic auto-framing
    this.manual = false;            // user has taken control
    this.idle = true;               // pre-first-input cinematic drift

    const takeControl = () => { this.manual = true; this.idle = false; };
    canvas.addEventListener('pointerdown', (e) => {
      takeControl();
      this._drag = { x: e.clientX, y: e.clientY, id: e.pointerId };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this._drag || e.pointerId !== this._drag.id) return;
      const dx = e.clientX - this._drag.x, dy = e.clientY - this._drag.y;
      this._drag.x = e.clientX; this._drag.y = e.clientY;
      this.yaw -= dx * 0.0048;
      this.pitch = clamp(this.pitch + dy * 0.0048, -1.45, 1.45);
    });
    const endDrag = (e) => { if (this._drag && e.pointerId === this._drag.id) this._drag = null; };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      takeControl();
      this._targetDist = clamp(this._targetDist * Math.exp(e.deltaY * 0.0012), this.minDist, this.maxDist);
    }, { passive: false });
  }

  // can the user hand control back to an automatic view right now?
  canReturnToAuto() { return this.manual && (this.autoEnabled || !!this.getFocusPos); }
  returnToAuto() { this.manual = false; }

  focusOn(getPos, radius, opts = {}) {
    this.getFocusPos = getPos;
    this.minDist = Math.max(0.4, radius * 1.25);
    this.manual = false;                       // an explicit pick returns to auto-follow
    this.manualFocus = !!opts.user;            // a user pick overrides cinematic auto-framing
    if (opts.dist) this._targetDist = clamp(opts.dist, this.minDist, this.maxDist);
    if (this._targetDist < this.minDist) this._targetDist = this.minDist * 3;
    if (opts.jump || opts.user) {              // user picks SNAP straight to the body
      this.dist = this._targetDist;
      this.focusPos = getPos().slice();
    }
  }

  update(dt) {
    dt = Math.min(dt, 0.05);                   // a hitch must not teleport the camera
    let desiredPos, desiredDist, key;
    if (!this.manual && !this.manualFocus && this.autoEnabled && this.autoTarget) {
      desiredPos = this.autoTarget.pos;        // cinematic framing owns both
      desiredDist = clamp(this.autoTarget.dist, this.minDist, this.maxDist);
      key = 'auto';
    } else {
      desiredPos = this.getFocusPos ? this.getFocusPos() : this.focusPos;  // keep body centered
      desiredDist = this._targetDist;          // user-controlled zoom (or held in manual)
      key = this.getFocusPos || 'self';
    }
    // RIDE the target: add the target's own frame-to-frame motion to the camera FIRST, so a
    // body's motion (orbits, time warp) never drags the view — the smoothing below eases only
    // the transition offset (focus switches, cinematic re-aims), never the follow itself.
    // Without this, the filter lags by v/rate: invisible at 60x, thousands of Mm at 1Mx warp.
    if (this._lastKey === key && this._lastDesired) {
      const d = vsub(desiredPos, this._lastDesired);
      // a same-key jump far beyond the view scale is a RE-AIM (cinematic picked a new subject),
      // not motion — let the smoothing ease it instead of hard-cutting
      if (key !== 'auto' || vlen(d) < 6 * Math.max(desiredDist, this.dist)) {
        this.focusPos = vadd(this.focusPos, d);
      }
    }
    this._lastKey = key;
    this._lastDesired = desiredPos.slice();
    // exact framerate-independent exponential smoothing
    this.focusPos = vlerp(this.focusPos, desiredPos, 1 - Math.exp(-5 * dt));
    this.dist += (desiredDist - this.dist) * (1 - Math.exp(-6 * dt));
    if (this.idle && !this.manual) this.yaw += dt * 0.02;   // gentle pre-input drift only
  }

  // camera position relative to focus (render frame origin = focus)
  eyeRel() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    return [
      this.dist * cp * Math.cos(this.yaw),
      this.dist * cp * Math.sin(this.yaw),
      this.dist * sp,
    ];
  }
}

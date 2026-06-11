// camera.js — orbit camera around a focus target with smooth transitions
import { clamp, vadd, vsub, vlerp } from './mathx.js';

export class OrbitCamera {
  constructor(canvas) {
    this.canvas = canvas;
    this.yaw = 0.6;
    this.pitch = 0.32;
    this.dist = 40;          // Mm from focus
    this.minDist = 8;
    this.maxDist = 8e6;
    this.fov = 50 * Math.PI / 180;
    this.focusPos = [0, 0, 0];      // heliocentric Mm (double) — set externally per frame
    this._targetDist = this.dist;
    this._vyaw = 0; this._vpitch = 0;
    this.idle = true;
    this._idleT = 0;

    this.userTouched = 0;        // last manual interaction (ms) — pauses auto-framing
    this.autoTarget = null;      // {pos: helio, dist} fed by sim's cinematic framing
    canvas.addEventListener('pointerdown', (e) => {
      this.idle = false;
      this.userTouched = performance.now();
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
      this.idle = false;
      this.userTouched = performance.now();
      const f = Math.exp(e.deltaY * 0.0012);
      this._targetDist = clamp(this._targetDist * f, this.minDist, this.maxDist);
    }, { passive: false });
    canvas.addEventListener('dblclick', () => { this.idle = true; this._idleT = 0; });
  }

  focusOn(getPos, radius, opts = {}) {
    this.getFocusPos = getPos;
    this.minDist = Math.max(0.4, radius * 1.25);
    // explicit body selection: follow that body for a while even if auto-framing exists
    this.followUntil = performance.now() + 12000;
    if (opts.dist) this._targetDist = clamp(opts.dist, this.minDist, this.maxDist);
    if (this._targetDist < this.minDist) this._targetDist = this.minDist * 3;
    if (opts.jump) { this.dist = this._targetDist; this.focusPos = getPos().slice(); }
  }

  update(dt) {
    // ONE focal point, ONE smoothing law — and when the USER moves the camera during
    // auto-framing, the focal point simply HOLDS (orbit where you are). No fallback target,
    // no destination fight, no glide-back ping-pong. Body-follow happens only when there is
    // no framing target, or right after an explicit focus selection.
    const now = performance.now();
    const explicitFollow = now < (this.followUntil || 0);
    const autoActive = this.autoTarget && !explicitFollow && now - this.userTouched > 4500;
    let desired = this.focusPos;                      // default: HOLD (user-owned point)
    let desiredDist = this._targetDist;
    if (autoActive) {
      desired = this.autoTarget.pos;
      desiredDist = clamp(this.autoTarget.dist, this.minDist, this.maxDist);
    } else if (this.getFocusPos && (explicitFollow || !this.autoTarget)) {
      desired = this.getFocusPos();
    }
    const k = Math.min(1, dt * 6);
    this.focusPos = vlerp(this.focusPos, desired, k);
    this._targetDist += (desiredDist - this._targetDist) * Math.min(1, dt * 1.8);
    this.dist += (this._targetDist - this.dist) * Math.min(1, dt * 7);
    if (this.idle) {
      this._idleT += dt;
      this.yaw += dt * 0.018;        // slow cinematic drift until first input
    }
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

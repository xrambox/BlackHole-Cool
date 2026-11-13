// Run with:  node --test
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const BH = require('../physics.js');
const { PI, B_CRIT } = BH;
const { dot, norm, scale, add, normalize, cross } = BH.vec;

const close = (a, b, tol, msg) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg || ''} expected ${b}, got ${a} (diff ${Math.abs(a - b)}, tol ${tol})`);

/* ---------- an independent 3D integrator ----------
 * Null geodesics of Schwarzschild in Cartesian-like coordinates, from the
 * Hamiltonian H = 1/2 ( -E^2/f + p.p - (1-f)(n.p)^2 ),  f = 1 - 1/r.
 * It never uses the orbit equation, the orbital plane, or the angle phi.
 */
function hamDeriv(s, E) {
  const [x, y, z, px, py, pz] = s;
  const r = Math.hypot(x, y, z), n = [x / r, y / r, z / r];
  const f = 1 - 1 / r, A = 1 / r;
  const sp = n[0] * px + n[1] * py + n[2] * pz;
  const c = E * E / (2 * r * r * f * f) + sp * sp / (2 * r * r);
  const p = [px, py, pz];
  return [
    px - A * sp * n[0], py - A * sp * n[1], pz - A * sp * n[2],
    ...[0, 1, 2].map(i => -(c * n[i] - sp * (p[i] - sp * n[i]) / (r * r))),
  ];
}
function hamStep(s, h, E) {
  const k1 = hamDeriv(s, E);
  const k2 = hamDeriv(s.map((v, i) => v + 0.5 * h * k1[i]), E);
  const k3 = hamDeriv(s.map((v, i) => v + 0.5 * h * k2[i]), E);
  const k4 = hamDeriv(s.map((v, i) => v + h * k3[i]), E);
  return s.map((v, i) => v + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}
// Integrate until the ray leaves past rEnd, or until stop(prev, next) is true.
function ham3d(s, E, { rEnd = 1e6, stop = null } = {}) {
  for (let i = 0; i < 400000; i++) {
    const r = Math.hypot(s[0], s[1], s[2]);
    const h = 0.004 * r;
    const next = hamStep(s, h, E);
    if (stop && stop(s, next)) return { state: s, stopped: true, h };
    s = next;
    const rn = Math.hypot(s[0], s[1], s[2]);
    // Inside the photon sphere an ingoing photon cannot turn back; stop before
    // these coordinates go singular at the horizon.
    if (rn < 1.05) return { state: s, captured: true };
    if (rn > rEnd && (s[0] * s[3] + s[1] * s[4] + s[2] * s[5]) > 0) return { state: s, escaped: true };
  }
  throw new Error('3D integrator did not finish');
}
// Momentum for a camera hovering at pos, looking along unit dir (its own frame).
function cameraMomentum(pos, dir) {
  const r = norm(pos), n = scale(pos, 1 / r), f = 1 - 1 / r;
  const dn = dot(dir, n);
  const p = add(add(dir, scale(n, -dn)), scale(n, dn / Math.sqrt(f)));
  return { state: [...pos, ...p], E: Math.sqrt(f) };
}
function velocity(s) {
  const r = Math.hypot(s[0], s[1], s[2]), n = [s[0] / r, s[1] / r, s[2] / r];
  const sp = n[0] * s[3] + n[1] * s[4] + n[2] * s[5];
  return normalize([0, 1, 2].map(i => s[3 + i] - sp * n[i] / r));
}

/* ---------- 1. the photon sphere ---------- */
test('photon sphere at 1.5 rs and critical impact parameter 3*sqrt(3)/2 rs', () => {
  close(B_CRIT, 2.598076211353316, 1e-14);
  // u = 2/3 is a fixed point of u'' = -u + 3/2 u^2
  close(-BH.U_PHOTON + 1.5 * BH.U_PHOTON ** 2, 0, 1e-15);
  // its first integral is exactly (1/b_c)^2
  close(BH.firstIntegral(BH.U_PHOTON, 0), 1 / (B_CRIT * B_CRIT), 1e-15);
});

/* ---------- 2. turning points ---------- */
test('turning-point roots solve u^3 - u^2 + 1/b^2 = 0, and vanish below b_c', () => {
  for (const b of [2.6, 3, 5, 20, 1e3, 1e5]) {
    const { u1, u2, u3 } = BH.cubicRoots(b);
    assert.ok(u1 < 0 && u1 < u2 && u2 <= u3, `ordering at b=${b}`);
    for (const u of [u1, u2, u3]) close(u * u * u - u * u + 1 / (b * b), 0, 2e-16);
    close(u1 + u2 + u3, 1, 1e-14);
  }
  assert.equal(BH.cubicRoots(2.5), null);
  assert.equal(BH.deflectionExact(2.598), Infinity);
});

/* ---------- 3. elliptic integral vs quadrature ---------- */
test('elliptic-integral bending agrees with direct numerical quadrature', () => {
  for (const b of [2.7, 3, 4, 7, 15, 60]) {
    const { u1, u2, u3 } = BH.cubicRoots(b);
    // u = u2 (1 - t^2) removes the square-root singularity at the turning point
    const g = t => { const u = u2 * (1 - t * t); return 2 * Math.sqrt(u2) / Math.sqrt((u - u1) * (u3 - u)); };
    const n = 20000; let s = g(0) + g(1);
    for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * g(i / n);
    const quad = 2 * s / (3 * n) - PI;
    close(BH.deflectionExact(b), quad, 1e-11, `b=${b}`);
  }
});

/* ---------- 4. weak field ---------- */
test('far from the hole, bending matches the weak-field series 2/b + 15pi/16b^2 + 16/3b^3', () => {
  const series = b => 2 / b + 15 * PI / (16 * b * b) + 16 / (3 * b ** 3);
  close(BH.deflectionExact(100), series(100), 2e-7);
  close(BH.deflectionExact(1000), series(1000), 2e-11);
  close(BH.deflectionExact(1e5), 2e-5, 1e-9);   // Einstein's 4GM/(c^2 b)
});

/* ---------- 5. strong field ---------- */
test('near b_c, bending diverges logarithmically (Bozza strong-field limit)', () => {
  const bbar = Math.log(216 * (7 - 4 * Math.sqrt(3))) - PI;  // -0.4002
  for (const eps of [1e-5, 1e-7, 1e-9]) {
    const b = B_CRIT * (1 + eps);
    close(BH.deflectionExact(b), -Math.log(eps) + bbar, 30 * eps * Math.abs(Math.log(eps)) + 1e-6, `eps=${eps}`);
  }
});

/* ---------- 6. fine-step integration ---------- */
test('integrating the orbit equation with fine steps reproduces the exact bending to 1e-10', () => {
  for (const b of [2.65, 3, 5, 12, 40]) {
    const res = BH.traceFromInfinity(b, 1.5, 0.005, 1e6);
    assert.equal(res.status, BH.ESCAPED);
    close(res.phi - PI, BH.deflectionExact(b), 1e-10, `b=${b}`);
  }
});

/* ---------- 7. the renderer's step size ---------- */
test('the step sizes the shader uses are accurate to 5 parts per million before rounding', () => {
  for (const b of [2.62, 2.8, 3, 4, 6, 10, 20, 50, 100]) {
    const res = BH.traceFromInfinity(b);
    assert.equal(res.status, BH.ESCAPED, `b=${b}`);
    const exact = BH.deflectionExact(b);
    close(res.phi - PI, exact, (b < 2.7 ? 5e-6 : 1e-6) * exact, `b=${b}`);
  }
});

/* ---------- 8. conservation ---------- */
test('the first integral (u\')^2 + u^2 - u^3 stays constant along a traced ray', () => {
  const b = 3.3;
  const beta2 = 1 / (b * b);
  let u = 0, w = 1 / b, maxErr = 0;
  for (let i = 0; i < 40; i++) {
    [u, w] = BH.rk4(u, w, BH.STEP, 1.5);
    maxErr = Math.max(maxErr, Math.abs(BH.firstIntegral(u, w) - beta2) / beta2);
  }
  assert.ok(maxErr < 5e-7, `relative drift ${maxErr}`);
});

/* ---------- 9. flat space ---------- */
test('with gravity switched off, rays are straight lines', () => {
  for (const b of [0.5, 2, 5, 30]) {
    const res = BH.traceFromInfinity(b, 0);
    if (b < 1) { assert.equal(res.status, BH.CAPTURED); continue; }
    assert.equal(res.status, BH.ESCAPED);
    close(res.phi, PI, 1e-9, `b=${b}`);
  }
  // camera ray: escapes in exactly the direction it was aimed
  const cam = [10, 3, -4], dir = normalize([0.3, 0.5, 0.81]);
  const res = BH.traceCameraRay(cam, dir, { G: 0, disk: false });
  close(dot(res.escapeDir, dir), 1, 1e-12);
});

/* ---------- 10. capture threshold ---------- */
test('the tracer separates captured from escaping rays at b_c', () => {
  let lo = 2.5, hi = 2.7;
  for (let i = 0; i < 40; i++) {
    const mid = 0.5 * (lo + hi);
    if (BH.traceFromInfinity(mid).status === BH.ESCAPED) hi = mid; else lo = mid;
  }
  close(0.5 * (lo + hi), B_CRIT, 1e-6 * B_CRIT);
});

/* ---------- 11. shadow size formula ---------- */
test('the shadow covers half the sky at 1.5 rs and tends to b_c / r far away', () => {
  close(BH.shadowAngle(1.5), PI / 2, 1e-12);
  close(BH.shadowAngle(1e6), B_CRIT / 1e6, 1e-11);
  assert.ok(BH.shadowAngle(1.2) > PI / 2);
  close(BH.shadowAngle(22) * 360 / PI, 13.2510, 1e-3);     // diameter in degrees
  close(BH.shadowAngle(22, 0), Math.asin(1 / 22), 1e-15);  // flat space: a ball of radius rs
});

/* ---------- 12. shadow edge from traced rays ---------- */
test('rays traced from a hovering camera find the shadow edge the formula predicts', () => {
  for (const r of [22, 6, 2]) {
    const u = 1 / r;
    let lo = 0, hi = PI / 2;   // angle from the inward direction
    for (let i = 0; i < 40; i++) {
      const a = 0.5 * (lo + hi);
      const [u0, w0] = BH.initState(u, -Math.cos(a), Math.sin(a));
      if (BH.traceOrbit(u0, w0, 1.5).status === BH.ESCAPED) hi = a; else lo = a;
    }
    close(0.5 * (lo + hi), BH.shadowAngle(r), 2e-6, `r=${r}`);
  }
});

/* ---------- 13. Einstein ring ---------- */
test('the traced Einstein ring matches the exact elliptic-integral solution', () => {
  const r = 22, u = 1 / r;
  const aE = BH.einsteinRingAngle(r);
  const phiAt = a => { const [u0, w0] = BH.initState(u, -Math.cos(a), Math.sin(a)); return BH.traceOrbit(u0, w0, 1.5).phi; };
  let lo = BH.shadowAngle(r) * 1.01, hi = PI / 2;
  for (let i = 0; i < 40; i++) {
    const m = 0.5 * (lo + hi);
    if (phiAt(m) > PI) lo = m; else hi = m;
  }
  close(0.5 * (lo + hi), aE, 2e-6 * aE);
  assert.ok(aE > BH.shadowAngle(r) && aE < PI / 2);
});

/* ---------- 14. 3D integrator: bending in a tilted plane ---------- */
test('an independent 3D geodesic integrator gives the same bending, in a tilted plane', () => {
  const R = 1e6, inc = 0.6;
  for (const b of [3, 5, 10]) {
    // momentum along +x with H = 0, E = 1; offset y0 so that L/E is exactly b
    let k = 1, y0 = b, pos;
    for (let i = 0; i < 5; i++) {
      pos = [-R, y0 * Math.cos(inc), y0 * Math.sin(inc)];
      const r = norm(pos), nx = pos[0] / r;
      k = 1 / Math.sqrt((1 - 1 / r) * (1 - nx * nx / r));
      y0 = b / k;
    }
    const out = ham3d([...pos, k, 0, 0], 1);
    assert.ok(out.escaped);
    const v = velocity(out.state);
    close(Math.acos(Math.max(-1, Math.min(1, v[0]))), BH.deflectionExact(b), 1e-9, `b=${b}`);
    // it never leaves its plane
    const nPlane = [0, -Math.sin(inc), Math.cos(inc)];
    close(dot(out.state.slice(0, 3), nPlane) / norm(out.state.slice(0, 3)), 0, 1e-9);
  }
});

/* ---------- 15. camera rays vs 3D ---------- */
test('a camera pixel escapes in the same direction in the plane tracer and in 3D', () => {
  const cam = BH.cameraBasis(22, 9, 0.7);
  for (const [px, py] of [[700, 380], [480, 300], [900, 120], [530, 290]]) {
    const dir = BH.pixelDir(cam, px, py, 1280, 720, Math.tan(40 * PI / 180));
    const plane = BH.traceCameraRay(cam.pos, dir, { step: 0.002, maxSteps: 1e6, disk: false });
    const { state, E } = cameraMomentum(cam.pos, dir);
    const out = ham3d(state, E, { rEnd: 1e7 });
    if (plane.status !== BH.ESCAPED) { assert.ok(out.captured, `pixel ${px},${py}`); continue; }
    assert.ok(out.escaped);
    // at r = 1e7 the position direction equals the escape direction to ~1e-6
    const p3 = normalize(out.state.slice(0, 3));
    close(dot(p3, plane.escapeDir), 1, 1e-11, `pixel ${px},${py}`);
  }
});

/* ---------- 16. disk crossings vs 3D ---------- */
test('the disk-plane crossings land where the 3D integrator crosses y = 0', () => {
  const cam = BH.cameraBasis(22, 9, 0.3);
  for (const [px, py] of [[640, 300], [400, 350], [900, 400], [640, 250]]) {
  const dir = BH.pixelDir(cam, px, py, 1280, 720, Math.tan(40 * PI / 180));
  const plane = BH.traceCameraRay(cam.pos, dir, { step: 0.002, maxSteps: 1e6 });
  assert.ok(plane.hits.length >= 1, `pixel ${px},${py}`);
  for (const h of plane.hits) close(h.x[1] / h.r, 0, 1e-12);
  const { state, E } = cameraMomentum(cam.pos, dir);
  const out = ham3d(state, E, { stop: (a, b) => a[1] * b[1] <= 0 && a[1] !== 0 });
  // land exactly on y = 0
  let s = out.state, lo = 0, hi = out.h;
  for (let i = 0; i < 60; i++) {
    const m = 0.5 * (lo + hi);
    if (hamStep(s, m, E)[1] * s[1] > 0) lo = m; else hi = m;
  }
  const x = hamStep(s, 0.5 * (lo + hi), E);
  close(norm(x.slice(0, 3)), plane.hits[0].r, 1e-8);
  close(dot(normalize(x.slice(0, 3)), normalize(plane.hits[0].x)), 1, 1e-12);
  }
});

/* ---------- 17. redshift ---------- */
test('the gas redshift formula matches -p.u from the 3D integrator and a local Doppler calculation', () => {
  const cam = BH.cameraBasis(22, 12, 0.0);
  let checked = 0;
  for (const [px, py] of [[400, 360], [880, 360], [640, 330], [640, 420], [560, 300]]) {
    const dir = BH.pixelDir(cam, px, py, 1280, 720, Math.tan(40 * PI / 180));
    const plane = BH.traceCameraRay(cam.pos, dir, { step: 0.002, maxSteps: 1e6 });
    const hit = plane.hits.find(h => h.r >= BH.DISK_IN && h.r <= BH.DISK_OUT);
    if (!hit) continue;
    const gFormula = BH.redshift(hit.r, plane.lambda, 22);

    const { state, E } = cameraMomentum(cam.pos, dir);
    let crossings = 0;
    const out = ham3d(state, E, { stop: (a, b) => a[1] * b[1] <= 0 && ++crossings === hit.index + 1 });
    let s = out.state, lo = 0, hi = out.h;
    for (let i = 0; i < 60; i++) { const m = 0.5 * (lo + hi); if (hamStep(s, m, E)[1] * s[1] > 0) lo = m; else hi = m; }
    s = hamStep(s, 0.5 * (lo + hi), E);
    const x = s.slice(0, 3), p = s.slice(3), r = norm(x);
    close(r, hit.r, 1e-7);
    // (a) covariant: physical photon k = (-E, -p); gas u = u^t (1, Omega yhat x X)
    const Om = BH.keplerOmega(r), ut = 1 / Math.sqrt(1 - 1.5 / r);
    const v = scale(cross([0, 1, 0], x), Om);
    const g3d = 1 / (ut * (E + dot(v, p)));
    close(gFormula, g3d, 1e-8, `pixel ${px},${py}`);
    // (b) local: gravitational shift times special-relativistic Doppler in the static frame
    const n = scale(x, 1 / r), f = 1 - 1 / r;
    const kr = -dot(p, n), kt = add(scale(p, -1), scale(n, -kr));     // physical photon, covariant
    const kLocal = add(kt, scale(n, kr * Math.sqrt(f)));               // orthonormal static frame
    const speed = Math.sqrt(0.5 / (r - 1));
    const cosChi = dot(kLocal, normalize(v)) / norm(kLocal);
    const gamma = 1 / Math.sqrt(1 - speed * speed);
    const gLocal = Math.sqrt(f) / (Math.sqrt(1 - 1 / 22) * gamma * (1 - speed * cosChi));
    close(gFormula, gLocal, 1e-8, `pixel ${px},${py}`);
    checked++;
  }
  assert.ok(checked >= 3, `only ${checked} rays hit the disk`);
});

/* ---------- 18. colour and heat ---------- */
test('blackbody colours, the shifted-blackbody identity, and the disk temperature profile', () => {
  // CIE illuminant A is a 2856 K blackbody at (0.4476, 0.4074)
  const [xa, ya] = BH.chromaticity(2856);
  close(xa, 0.4476, 0.003); close(ya, 0.4074, 0.003);
  const [x65, y65] = BH.chromaticity(6500);
  close(x65, 0.3135, 0.003); close(y65, 0.3237, 0.003);
  // g^3 B_nu(nu/g, T) = B_nu(nu, g T): a shifted blackbody is a blackbody
  for (const g of [0.6, 1.4]) for (const nu of [3e14, 6e14])
    close(g ** 3 * BH.planckNu(nu / g, 5000) / BH.planckNu(nu, g * 5000), 1, 1e-12);
  // hottest at 49/36 of the inner edge, cold at the inner edge, cooler outward
  close(BH.diskTemperature(BH.R_PEAK, 4700), 4700, 1e-9);
  assert.equal(BH.diskTemperature(3, 4700), 0);
  assert.ok(BH.diskTemperature(BH.R_PEAK * 1.01, 4700) < 4700 && BH.diskTemperature(BH.R_PEAK * 0.99, 4700) < 4700);
  assert.ok(BH.diskTemperature(12, 4700) < BH.diskTemperature(6, 4700));
});

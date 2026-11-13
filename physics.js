/*
 * Black Hole Lab — physics core.
 *
 * Units: lengths in Schwarzschild radii (rs = 1, so M = 1/2 with G = c = 1).
 * Light orbits obey  u'' + u = (3/2) u^2,  with u = rs / r and ' = d/dphi.
 *
 * The ray tracer here (traceOrbit / traceCameraRay) is a line-for-line mirror of
 * the GLSL in index.html, so the CPU can explain any pixel the GPU draws and the
 * tests can check the algorithm the shader runs.
 *
 * Works as a browser global (window.BH) and as a CommonJS module.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.BH = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const PI = Math.PI;
  const B_CRIT = 1.5 * Math.sqrt(3);        // critical impact parameter, 2.598 rs
  const U_PHOTON = 2 / 3;                    // photon sphere, r = 1.5 rs
  const DISK_IN = 3;                         // innermost stable circular orbit
  const DISK_OUT = 12;
  const STEP = 0.1;                          // radians of swept angle per RK4 step, far out
  const MAX_STEPS = 500;
  // Steps shrink near the photon sphere, where rays are most sensitive.
  const stepAt = (u, base) => base / (1 + 3 * u * u);
  const ESCAPED = 1, CAPTURED = 2, TIMEOUT = 3;

  /* ---------- the orbit equation ---------- */

  // One fourth-order Runge–Kutta step of u'' + u = K u^2 (K = 1.5 with
  // gravity, 0 for flat space). Lawson's integrating-factor form: the straight-
  // line part u'' + u = 0 is a rotation of (u, u') and is applied exactly, so
  // RK4 only has to approximate the gravity term. Flat space comes out exact.
  function rk4(u, w, h, K) {
    const c = Math.cos(0.5 * h), s = Math.sin(0.5 * h);
    const C = Math.cos(h), S = Math.sin(h);
    // stage 1
    const n1 = K * u * u;                               // K1 = (0, n1)
    // stage 2: rotate y + h/2 K1 forward by h/2, evaluate, rotate back
    let a = u * c + (w + 0.5 * h * n1) * s;
    const n2 = K * a * a, k2u = -n2 * s, k2w = n2 * c;
    // stage 3
    a = (u + 0.5 * h * k2u) * c + (w + 0.5 * h * k2w) * s;
    const n3 = K * a * a, k3u = -n3 * s, k3w = n3 * c;
    // stage 4
    a = (u + h * k3u) * C + (w + h * k3w) * S;
    const n4 = K * a * a, k4u = -n4 * S, k4w = n4 * C;
    const vu = u + h / 6 * (2 * k2u + 2 * k3u + k4u);
    const vw = w + h / 6 * (n1 + 2 * k2w + 2 * k3w + k4w);
    return [vu * C + vw * S, -vu * S + vw * C];
  }

  // (u')^2 + u^2 - u^3 = (rs/b)^2 is conserved along every light ray.
  function firstIntegral(u, w) { return w * w + u * u - u * u * u; }

  /* ---------- exact results ---------- */

  // Roots of u^3 - u^2 + beta^2 = 0, beta = 1/b. Three real roots u1 < 0 < u2 <= u3
  // exist only when b > B_CRIT; u2 is the ray's closest approach.
  function cubicRoots(b) {
    const beta2 = 1 / (b * b);
    if (b <= B_CRIT) return null;
    const q = beta2 - 2 / 27;
    const th = Math.acos(Math.max(-1, Math.min(1, -13.5 * q)));
    let r = [2 / 3 * Math.cos(th / 3 + 2 * PI / 3) + 1 / 3,
             2 / 3 * Math.cos(th / 3 - 2 * PI / 3) + 1 / 3,
             2 / 3 * Math.cos(th / 3) + 1 / 3];
    r = r.map(x => {
      for (let i = 0; i < 3; i++) {
        const d = 3 * x * x - 2 * x;
        if (d === 0) break;
        x -= (x * x * x - x * x + beta2) / d;
      }
      return x;
    });
    return { u1: r[0], u2: r[1], u3: r[2] };
  }

  // Carlson's symmetric elliptic integral R_F (duplication algorithm).
  function carlsonRF(x, y, z) {
    let ave, dx, dy, dz;
    for (let i = 0; i < 200; i++) {
      const sx = Math.sqrt(x), sy = Math.sqrt(y), sz = Math.sqrt(z);
      const lam = sx * (sy + sz) + sy * sz;
      x = 0.25 * (x + lam); y = 0.25 * (y + lam); z = 0.25 * (z + lam);
      ave = (x + y + z) / 3;
      dx = (ave - x) / ave; dy = (ave - y) / ave; dz = (ave - z) / ave;
      if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) < 1e-4) break;
    }
    const e2 = dx * dy - dz * dz, e3 = dx * dy * dz;
    return (1 + (e2 / 24 - 0.1 - 3 * e3 / 44) * e2 + e3 / 14) / Math.sqrt(ave);
  }

  // Incomplete elliptic integral of the first kind F(phi, k), modulus k.
  function ellipticF(phi, k) {
    const s = Math.sin(phi), c = Math.cos(phi);
    return s * carlsonRF(c * c, 1 - k * k * s * s, 1);
  }

  // Angle swept while u climbs from x to the turning point u2 (Gradshteyn 3.131.5).
  function sweptToTurningPoint(b, x) {
    const R = cubicRoots(b);
    if (!R) return Infinity;
    const { u1, u2, u3 } = R;
    const k = Math.sqrt((u2 - u1) / (u3 - u1));
    const s2 = Math.min(1, (u3 - u1) * (u2 - x) / ((u2 - u1) * (u3 - x)));
    return 2 / Math.sqrt(u3 - u1) * ellipticF(Math.asin(Math.sqrt(s2)), k);
  }

  // Total bending of a ray arriving from infinity with impact parameter b.
  function deflectionExact(b) {
    if (b <= B_CRIT) return Infinity;
    return 2 * sweptToTurningPoint(b, 0) - PI;
  }

  // Angular radius of the shadow for a camera hovering at radius r.
  function shadowAngle(r, G = 1) {
    if (!G) return Math.asin(Math.min(1, 1 / r));
    const s = Math.min(1, B_CRIT / r * Math.sqrt(1 - 1 / r));
    return r >= 1.5 ? Math.asin(s) : PI - Math.asin(s);
  }

  // Angular radius of the Einstein ring (image of the point directly behind
  // the hole) for a camera hovering at radius r: solved exactly.
  function einsteinRingAngle(r) {
    const uc = 1 / r;
    const bmax = r / Math.sqrt(1 - uc);
    const total = b => sweptToTurningPoint(b, uc) + sweptToTurningPoint(b, 0);
    let lo = B_CRIT * (1 + 1e-12), hi = bmax * (1 - 1e-15);
    for (let i = 0; i < 200; i++) {
      const mid = 0.5 * (lo + hi);
      if (total(mid) > PI) lo = mid; else hi = mid;
    }
    const b = 0.5 * (lo + hi);
    return Math.asin(b * uc * Math.sqrt(1 - uc));
  }

  /* ---------- the tracer the shader runs ---------- */

  // Start state for a camera hovering at u = 1/r. psi is the angle between the
  // ray and the outward radial direction, measured in the camera's own frame.
  function initState(u, cosPsi, sinPsi, G = 1) {
    const s = Math.max(sinPsi, 1e-7);
    return [u, -u * Math.sqrt(1 - G * u) * cosPsi / s];
  }

  // March u(phi) from (u, w). The disk plane is crossed at firstCross, then
  // every PI after. onCross(u, phi, index) may return false to stop the ray.
  function traceOrbit(u, w, K, firstCross = Infinity, onCross = null, step = STEP, maxSteps = MAX_STEPS) {
    let phi = 0, nextC = firstCross, steps = 0, crossings = 0;
    for (let i = 0; i < maxSteps; i++) {
      let h = stepAt(u, step), crossing = false;
      if (phi + h >= nextC) { h = nextC - phi; crossing = true; }
      const [u1, w1] = rk4(u, w, h, K);
      steps++;
      if (u1 <= 0) {
        // Escaped to infinity inside this step: land on u = 0 with Newton steps.
        for (let k = 0; k < 3; k++) {
          const d = -u / Math.min(w, -1e-12);
          [u, w] = rk4(u, w, d, K);
          phi += d;
        }
        return { status: ESCAPED, phi, steps, crossings, u, w };
      }
      if (u1 >= 1) return { status: CAPTURED, phi: phi + h, steps, crossings, u: u1, w: w1 };
      u = u1; w = w1; phi += h;
      if (crossing) {
        crossings++;
        if (onCross && onCross(u, phi, crossings - 1) === false)
          return { status: 4, phi, steps, crossings, u, w };
        nextC += PI;
      }
    }
    return { status: TIMEOUT, phi, steps, crossings, u, w };
  }

  function traceFromInfinity(b, K = 1.5, step = STEP, maxSteps = MAX_STEPS) {
    return traceOrbit(0, 1 / b, K, Infinity, null, step, maxSteps);
  }

  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = a => Math.hypot(a[0], a[1], a[2]);
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const normalize = a => scale(a, 1 / norm(a));

  // Reduce a 3D camera ray to its orbital plane and trace it.
  // pos: camera position; dir: unit ray direction in the camera's frame.
  function traceCameraRay(pos, dir, opts = {}) {
    const G = opts.G === undefined ? 1 : opts.G;
    const r = norm(pos);
    const e1 = scale(pos, 1 / r);
    const cp = dot(dir, e1);
    const perp = add(dir, scale(e1, -cp));
    const sp = norm(perp);
    const e2 = sp > 1e-7 ? scale(perp, 1 / sp)
      : normalize(cross(e1, Math.abs(e1[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]));
    const [u0, w0] = initState(1 / r, cp, sp, G);
    const a = e1[1], c = e2[1];
    let first = Infinity;
    if (opts.disk !== false && Math.abs(a) + Math.abs(c) > 1e-6) {
      first = ((Math.atan2(-a, c) % PI) + PI) % PI;
      if (first < 1e-5) first += PI;
    }
    // Photon's conserved L_z / E about the disk axis (+y).
    const lambda = -(r / Math.sqrt(1 - G / r)) * (e1[2] * dir[0] - e1[0] * dir[2]);
    const at = (phi, u) => scale(add(scale(e1, Math.cos(phi)), scale(e2, Math.sin(phi))), 1 / u);
    const hits = [];
    const res = traceOrbit(u0, w0, 1.5 * G, first, (u, phi, index) => {
      hits.push({ r: 1 / u, x: at(phi, u), phi, index });
      if (opts.stopAtDisk && 1 / u >= DISK_IN && 1 / u <= DISK_OUT) return false;
    }, opts.step || STEP, opts.maxSteps || MAX_STEPS);
    const out = Object.assign({}, res, { hits, lambda, e1, e2, psi: Math.atan2(sp, cp), b: r * sp / Math.sqrt(1 - G / r) });
    if (res.status === ESCAPED)
      out.escapeDir = add(scale(e1, Math.cos(res.phi)), scale(e2, Math.sin(res.phi)));
    return out;
  }

  /* ---------- camera ---------- */

  function cameraBasis(dist, elevDeg, az) {
    const el = elevDeg * PI / 180;
    const pos = [dist * Math.cos(el) * Math.cos(az), dist * Math.sin(el), dist * Math.cos(el) * Math.sin(az)];
    const fwd = scale(pos, -1 / dist);
    const right = normalize(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);
    return { pos, fwd, right, up };
  }

  // px, py in pixels from the bottom-left; tanHalf of the horizontal field of view.
  function pixelDir(cam, px, py, w, h, tanHalf) {
    const s = 2 * tanHalf / w;
    const x = (px - 0.5 * w) * s, y = (py - 0.5 * h) * s;
    return normalize(add(cam.fwd, add(scale(cam.right, x), scale(cam.up, y))));
  }

  /* ---------- gas ---------- */

  const R_PEAK = 49 / 36 * DISK_IN;   // hottest radius of a zero-torque thin disk
  const tProfile = r => r <= DISK_IN ? 0 : Math.pow(r, -0.75) * Math.pow(1 - Math.sqrt(DISK_IN / r), 0.25);
  function diskTemperature(r, tPeak) { return tPeak * tProfile(r) / tProfile(R_PEAK); }
  function keplerOmega(r) { return Math.sqrt(0.5 / (r * r * r)); }

  // Observed / emitted frequency for gas on a circular orbit at r, photon with
  // L_z/E = lambda, camera hovering at rCam. G = 0 gives flat space.
  function redshift(r, lambda, rCam, G = 1) {
    return Math.sqrt(1 - (0.5 + G) / r) / (Math.sqrt(1 - G / rCam) * (1 - keplerOmega(r) * lambda));
  }

  /* ---------- colour ---------- */

  // CIE 1931 2-degree observer, multi-lobe fit of Wyman, Sloan & Shirley (2013).
  const lobe = (x, mu, s1, s2) => { const t = (x - mu) / (x < mu ? s1 : s2); return Math.exp(-0.5 * t * t); };
  function cieXYZ(nm) {
    return [1.056 * lobe(nm, 599.8, 37.9, 31.0) + 0.362 * lobe(nm, 442.0, 16.0, 26.7) - 0.065 * lobe(nm, 501.1, 20.4, 26.2),
            0.821 * lobe(nm, 568.8, 46.9, 40.5) + 0.286 * lobe(nm, 530.9, 16.3, 31.1),
            1.217 * lobe(nm, 437.0, 11.8, 36.0) + 0.681 * lobe(nm, 459.0, 26.0, 13.8)];
  }
  const C2 = 1.438777e7;               // second radiation constant, nm K
  function planckLambda(nm, T) { return Math.pow(nm, -5) / Math.expm1(C2 / (nm * T)); }
  const H_OVER_K = 4.799243073e-11;    // s K
  function planckNu(nu, T) { return nu * nu * nu / Math.expm1(H_OVER_K * nu / T); }

  function blackbodyXYZ(T) {
    let X = 0, Y = 0, Z = 0;
    for (let nm = 360; nm <= 830; nm += 1) {
      const p = planckLambda(nm, T), c = cieXYZ(nm);
      X += p * c[0]; Y += p * c[1]; Z += p * c[2];
    }
    return [X * 1e12, Y * 1e12, Z * 1e12];
  }
  function chromaticity(T) { const [X, Y, Z] = blackbodyXYZ(T); const s = X + Y + Z; return [X / s, Y / s]; }
  function xyzToLinearSRGB([X, Y, Z]) {
    return [3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0.0557 * X - 0.2040 * Y + 1.0570 * Z];
  }

  // n texels, log-spaced temperature. RGB: linear sRGB at luminance 1 (pulled
  // into gamut toward white). A: log10 of luminance.
  const T_MIN = 300, T_MAX = 100000;
  function blackbodyTable(n) {
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const T = T_MIN * Math.pow(T_MAX / T_MIN, i / (n - 1));
      const xyz = blackbodyXYZ(T);
      let rgb = xyzToLinearSRGB(xyz.map(v => v / xyz[1]));
      const m = Math.min(rgb[0], rgb[1], rgb[2]);
      if (m < 0) { const t = 1 / (1 - m); rgb = rgb.map(v => 1 + (v - 1) * t); }
      out.set([rgb[0], rgb[1], rgb[2], Math.log10(xyz[1])], i * 4);
    }
    return out;
  }

  return {
    PI, B_CRIT, U_PHOTON, DISK_IN, DISK_OUT, STEP, MAX_STEPS, stepAt, ESCAPED, CAPTURED, TIMEOUT, R_PEAK, T_MIN, T_MAX,
    rk4, firstIntegral, cubicRoots, carlsonRF, ellipticF, sweptToTurningPoint, deflectionExact,
    shadowAngle, einsteinRingAngle, initState, traceOrbit, traceFromInfinity, traceCameraRay,
    cameraBasis, pixelDir, diskTemperature, keplerOmega, redshift,
    cieXYZ, planckLambda, planckNu, blackbodyXYZ, chromaticity, xyzToLinearSRGB, blackbodyTable,
    vec: { dot, norm, scale, add, cross, normalize },
  };
});

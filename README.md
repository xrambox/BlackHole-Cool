# Black Hole Lab

A non-rotating (Schwarzschild) black hole and its disk of hot gas, ray traced live in the browser. For every pixel, the GPU follows a ray of light backward through curved spacetime with the relativistic orbit equation

    u'' + u = (3/2) u²        (u = rₛ / r, primes are d/dφ)

and asks where that light came from: the disk, the background sky, or the event horizon.

## What's in the page

- **Live view** (WebGL 2): Doppler and gravitational frequency shifts, blackbody colour from Planck's law and the CIE observer, a camera hovering in place, and four preset views, including one at the photon sphere where the shadow fills half the sky.
- **Single-ray explorer**: aim one ray and compare its exact bending angle with what your GPU computes.
- **Self-check**: on load, the shader traces about 5,000 test rays and compares them with exact general-relativity results (bending angles, the critical impact parameter, shadow and Einstein-ring sizes).

## Files

| File | Purpose |
| --- | --- |
| `index.html` | The page: layout, GLSL shaders, interaction |
| `physics.js` | Physics core: exact solutions (Carlson elliptic integrals), and a CPU mirror of the shader's ray tracer |
| `test/physics.test.js` | 18 unit tests, including an independent 3D geodesic integrator |
| `scripts/build.js` | Copies the site files into `dist/` |
| `netlify.toml` | Netlify build settings |

## Run locally

Open `index.html` in a browser. No build step or server is needed.

```sh
npm test         # run the 18 physics tests (Node 18+)
npm run build    # tests, then copy the site into dist/
```

## Deploy to Netlify

Import this repository in Netlify. `netlify.toml` already sets the build command (`npm run build`) and the publish directory (`dist`). A deploy fails if any physics test fails.

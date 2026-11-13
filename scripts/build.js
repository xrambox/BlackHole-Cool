// Copies the static site into dist/ for deployment. Run via `npm run build`,
// which runs the physics tests first.
'use strict';
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const dist = path.join(root, 'dist');
const SITE_FILES = ['index.html', 'physics.js'];

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(dist);
for (const f of SITE_FILES) fs.copyFileSync(path.join(root, f), path.join(dist, f));
console.log(`Built ${SITE_FILES.length} files into dist/`);

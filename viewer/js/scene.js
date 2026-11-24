// scene.js — tout ce qui se dessine en 3D. Aucune logique d'interface ici.
//
// Plan de la scène (axe X = sens du calcul, de gauche à droite) :
//   - la colonne dorée : le flux du mot en cours, qui traverse les 12 couches
//   - autour, sur un arc : un flux fin par mot précédent (la « mémoire » du texte)
//   - avant chaque sphère, une « couronne » : c'est là que l'attention fait venir
//     l'information des autres mots (filaments bleus) vers la colonne
//   - chaque sphère : les 3 072 neurones du MLP de la couche
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const GAP = 4.2;            // distance entre deux couches
const R_STREAM = 3.6;       // rayon de l'arc des mots précédents
const R_HEAD = 2.7;         // rayon de la couronne des têtes d'attention
const CROWN = -2.1;         // la couronne d'attention se trouve juste avant la sphère
const RADII = [1.25, 2.1, 1.25];
const FLOW_P = 36, DIVE_P = 56, TRAIL = 48;

const BEZIER = /* glsl */`
  vec3 bez(vec3 a, vec3 b, vec3 c, vec3 d, float t) {
    float u = 1.0 - t;
    return u*u*u*a + 3.0*u*u*t*b + 3.0*u*t*t*c + t*t*t*d;
  }`;
const ROUND_POINT = /* glsl */`
  uniform float uShow;
  float roundPoint() { float r = length(gl_PointCoord - 0.5); if (r > 0.5) discard; return r; }`;

export function createScene(canvas, { L, M, H, Tmax }) {
  const layerX = w => (w - (L - 1) / 2) * GAP;
  const crownX = l => layerX(l) + CROWN;
  const entryX = layerX(-1), exitX = layerX(L);
  // Place de chaque mot précédent sur l'arc : suite du nombre d'or, pour que les premiers mots
  // soient déjà bien répartis et que les suivants comblent les vides
  const PHI0 = -2.1, SPAN = 4.2, GOLD = 0.6180339887;
  const slotAngle = j => PHI0 + ((j * GOLD) % 1) * SPAN;
  const slotDir = j => { const f = slotAngle(j); return [Math.sin(f), -Math.cos(f)]; };   // (y, z)
  const streamPos = (j, x) => { const [y, z] = slotDir(j); return new THREE.Vector3(x, y * R_STREAM, z * R_STREAM); };
  const headPos = (l, h) => { const a = Math.PI / 2 + (2 * Math.PI * h) / H; return new THREE.Vector3(crownX(l), Math.sin(a) * R_HEAD, Math.cos(a) * R_HEAD); };

  // ------------------------------------------------------------------
  // Rendu, caméra, lueur
  // ------------------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x03030a);
  const net = new THREE.Group();        // tout le réseau : on peut le réduire à un point (intro)
  scene.add(net);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.minDistance = 3;
  controls.maxDistance = 140;
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.9, 0.5, 0.12);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  const U = {
    uTime: { value: 0 }, uWave: { value: -2 }, uPix: { value: renderer.getPixelRatio() },
    uReveal: { value: 0 }, uFocus: { value: -10 }, uFocusAmt: { value: 0 }, uHover: { value: -10 },
  };
  // Intensité de chaque famille d'éléments (0 = invisible), pilotée par l'introduction
  const SHOW = { neurons: { value: 1 }, spine: { value: 1 }, memory: { value: 1 }, flows: { value: 1 }, rings: { value: 1 } };
  const additive = (vertexShader, fragmentShader, extra = {}) => new THREE.ShaderMaterial({
    uniforms: { ...U, uShow: { value: 1 }, ...extra }, vertexShader, fragmentShader,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });

  let seed = 42;
  const rnd = () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());

  // ------------------------------------------------------------------
  // 1. Les neurones : une sphère ovale par couche, un point par neurone
  // ------------------------------------------------------------------
  const NN = L * M;
  const nPos = new Float32Array(NN * 3), nLayer = new Float32Array(NN), nRand = new Float32Array(NN * 3), nAct = new Float32Array(NN);
  for (let l = 0, k = 0; l < L; l++) for (let i = 0; i < M; i++, k++) {
    let x = gauss(), y = gauss(), z = gauss(); const d = Math.hypot(x, y, z) || 1;
    const r = 0.45 + 0.55 * Math.cbrt(rnd());                     // plus dense vers la surface : effet de coque
    nPos.set([layerX(l) + x / d * r * RADII[0], y / d * r * RADII[1], z / d * r * RADII[2]], k * 3);
    nLayer[k] = l;
    nRand.set([rnd(), rnd(), rnd()], k * 3);
  }
  const nGeo = new THREE.BufferGeometry();
  nGeo.setAttribute('position', new THREE.BufferAttribute(nPos, 3));
  nGeo.setAttribute('aLayer', new THREE.BufferAttribute(nLayer, 1));
  nGeo.setAttribute('aRand', new THREE.BufferAttribute(nRand, 3));
  const nActAttr = new THREE.BufferAttribute(nAct, 1).setUsage(THREE.DynamicDrawUsage);
  nGeo.setAttribute('aAct', nActAttr);
  net.add(new THREE.Points(nGeo, additive(/* glsl */`
    attribute float aLayer, aAct; attribute vec3 aRand;
    uniform float uTime, uWave, uPix, uReveal, uFocus, uFocusAmt, uHover;
    varying float vI, vA;
    void main() {
      vec3 p = position + 0.05 * vec3(sin(uTime*.7 + aRand.x*6.28), sin(uTime*.6 + aRand.y*6.28), sin(uTime*.8 + aRand.z*6.28));
      float d = uWave - aLayer;                                   // > 0 : le signal a traversé cette couche
      float reveal = max(uReveal, smoothstep(-0.3, 0.1, d));
      float front = exp(-d*d*6.0) * (1.0 - uReveal);              // éclair au passage du signal
      float focus = abs(aLayer - uFocus) < 0.5 ? 1.0 : mix(1.0, 0.1, uFocusAmt);
      float hover = abs(aLayer - uHover) < 0.5 ? 1.7 : 1.0;
      vA = aAct;
      vI = (0.05 + aAct * (0.2 + 0.8*reveal) + front * (0.06 + 0.45*aAct)) * focus * hover;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_PointSize = min((1.1 + aAct*3.2 + front*0.8) * mix(1.0, 0.6, uFocusAmt) * uPix * (30.0 / -mv.z), 16.0 * uPix);
      gl_Position = projectionMatrix * mv;
    }`, ROUND_POINT + /* glsl */`
    varying float vI, vA;
    void main() {
      float r = roundPoint();
      vec3 col = mix(vec3(.15, .5, 1.0), vec3(1.0, .25, .6), smoothstep(.12, .6, vA));
      col = mix(col, vec3(1.0), smoothstep(.75, 1.0, vA));
      gl_FragColor = vec4((col * vI * pow(1.0 - r*2.0, 1.6)) * uShow, 1.0);
    }`, { uShow: SHOW.neurons })));

  // ------------------------------------------------------------------
  // 2. La colonne dorée : le flux résiduel du mot en cours
  // ------------------------------------------------------------------
  const NS = 3000;
  const sT = new Float32Array(NS), sR = new Float32Array(NS * 3);
  for (let i = 0; i < NS; i++) { sT[i] = rnd(); sR.set([rnd(), rnd(), rnd()], i * 3); }
  const sGeo = new THREE.BufferGeometry();
  sGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NS * 3), 3));
  sGeo.setAttribute('aT', new THREE.BufferAttribute(sT, 1));
  sGeo.setAttribute('aRand', new THREE.BufferAttribute(sR, 3));
  const spineU = { uNorm: { value: new Array(L + 1).fill(0) }, uX0: { value: entryX }, uX1: { value: exitX } };
  const spine = new THREE.Points(sGeo, additive(/* glsl */`
    attribute float aT; attribute vec3 aRand;
    uniform float uTime, uWave, uPix, uX0, uX1, uNorm[${L + 1}];
    varying float vI;
    void main() {
      float t = fract(aT + uTime * 0.025);
      float lc = -1.0 + t * ${L + 1}.0;
      float f = clamp(lc + 1.0, 0.0, ${L}.0);
      int i0 = int(floor(f)); int i1 = min(i0 + 1, ${L});
      float n = mix(uNorm[i0], uNorm[i1], fract(f));              // taille réelle du vecteur à cet endroit
      float ang = aRand.x * 6.2832 + uTime * 0.5 * (aRand.y - 0.5);
      float rad = (0.04 + 0.32 * n) * sqrt(aRand.z);
      vec3 p = vec3(mix(uX0, uX1, t), cos(ang) * rad, sin(ang) * rad);
      float d = uWave - lc;
      float front = exp(-d*d*3.0);
      vI = 0.04 + smoothstep(-0.2, 0.2, d) * 0.45 * n + front * 0.9;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_PointSize = (1.0 + front * 2.5) * uPix * (30.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }`, ROUND_POINT + /* glsl */`
    varying float vI;
    void main() { float r = roundPoint(); gl_FragColor = vec4((vec3(1.0, .85, .55) * vI * pow(1.0 - r*2.0, 1.5)) * uShow, 1.0); }`, { ...spineU, uShow: SHOW.spine }));
  spine.frustumCulled = false;   // positions calculées dans le shader : three.js ne peut pas deviner la boîte englobante
  net.add(spine);

  // ------------------------------------------------------------------
  // 3. Les mots précédents : un flux fin chacun, sur un arc autour de la colonne
  // ------------------------------------------------------------------
  const SLOTS = Math.max(1, Tmax - 1), PER = 220, NM = SLOTS * PER;
  const mSlot = new Float32Array(NM), mT = new Float32Array(NM), mR = new Float32Array(NM);
  for (let j = 0, k = 0; j < SLOTS; j++) for (let i = 0; i < PER; i++, k++) { mSlot[k] = j; mT[k] = rnd(); mR[k] = rnd(); }
  const mGeo = new THREE.BufferGeometry();
  mGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NM * 3), 3));
  mGeo.setAttribute('aSlot', new THREE.BufferAttribute(mSlot, 1));
  mGeo.setAttribute('aT', new THREE.BufferAttribute(mT, 1));
  mGeo.setAttribute('aR', new THREE.BufferAttribute(mR, 1));
  const memU = { uN: { value: 0 }, uPhi0: { value: PHI0 }, uSpan: { value: SPAN }, uR: { value: R_STREAM }, uX0: { value: entryX }, uX1: { value: exitX }, uGap: { value: GAP }, uHalf: { value: (L - 1) / 2 } };
  const memory = new THREE.Points(mGeo, additive(/* glsl */`
    attribute float aSlot, aT, aR;
    uniform float uTime, uWave, uPix, uN, uPhi0, uSpan, uR, uX0, uX1, uGap, uHalf, uFocusAmt;
    varying float vI;
    void main() {
      float on = step(aSlot + 0.5, uN);                          // seuls les mots déjà écrits existent
      float f = uPhi0 + fract(aSlot * 0.6180339887) * uSpan;
      float t = fract(aT + uTime * 0.02);
      float x = mix(uX0, uX1, t);
      float rad = uR + (aR - 0.5) * 0.07;
      vec3 p = vec3(x, sin(f) * rad, -cos(f) * rad);
      float lc = x / uGap + uHalf;
      float near = exp(-pow(uWave - lc, 2.0) * 1.5);
      vI = on * (0.06 + 0.22 * near) * mix(1.0, 0.6, uFocusAmt);
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_PointSize = on * 1.1 * uPix * (30.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }`, ROUND_POINT + /* glsl */`
    varying float vI;
    void main() { float r = roundPoint(); gl_FragColor = vec4((vec3(.7, .85, 1.0) * vI * (1.0 - r*2.0)) * uShow, 1.0); }`, { ...memU, uShow: SHOW.memory }));
  memory.frustumCulled = false;
  net.add(memory);

  // ------------------------------------------------------------------
  // 4. Filaments : des particules qui coulent le long de courbes de Bézier
  // ------------------------------------------------------------------
  function flowSystem(curves, perCurve, gateUniform, waveDriven) {
    const n = curves * perCurve;
    const g = new THREE.BufferGeometry();
    const attr = (name, size) => { const a = new THREE.BufferAttribute(new Float32Array(n * size), size).setUsage(THREE.DynamicDrawUsage); g.setAttribute(name, a); return a; };
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const A = { p0: attr('aP0', 3), p1: attr('aP1', 3), p2: attr('aP2', 3), p3: attr('aP3', 3), w: attr('aW', 1), layer: attr('aLayer', 1), hue: attr('aHue', 1), t: attr('aT', 1) };
    for (let i = 0; i < n; i++) A.t.array[i] = (i % perCurve) / perCurve + rnd() * 0.02;
    const mat = additive(BEZIER + /* glsl */`
      attribute vec3 aP0, aP1, aP2, aP3; attribute float aW, aLayer, aHue, aT;
      uniform float uTime, uWave, uPix, uReveal, uGate, uWaveDriven;
      varying float vI, vT, vHue;
      void main() {
        float t = fract(aT + uTime * 0.55);
        vec3 p = bez(aP0, aP1, aP2, aP3, t);
        float d = uWave - aLayer;
        float burst = exp(-d*d*12.0) * (1.0 - uReveal);
        float after = max(uReveal, smoothstep(-0.1, 0.25, d));
        float lit = mix(1.0, burst * 1.6 + after * 0.4, uWaveDriven);
        vI = aW * lit * uGate;
        vT = t; vHue = aHue;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_PointSize = aW < 0.01 ? 0.0 : (1.0 + aW * 2.2) * uPix * (30.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`, ROUND_POINT + /* glsl */`
      varying float vI, vT, vHue;
      void main() {
        float r = roundPoint();
        vec3 col = mix(vec3(.2, .75, 1.0), vec3(1.0, .35, .7), vHue);
        col = mix(col, vec3(1.0), smoothstep(.7, 1.0, vT) * .5);   // plus clair à l'arrivée
        gl_FragColor = vec4((col * vI * pow(1.0 - r*2.0, 1.4)) * uShow, 1.0);
      }`, { uGate: gateUniform, uWaveDriven: { value: waveDriven ? 1 : 0 }, uShow: SHOW.flows });
    const pts = new THREE.Points(g, mat);
    pts.frustumCulled = false;
    net.add(pts);
    function setCurve(c, a, b, cc, d, w, layer, hue) {
      for (let i = c * perCurve; i < (c + 1) * perCurve; i++) {
        A.p0.setXYZ(i, a.x, a.y, a.z); A.p1.setXYZ(i, b.x, b.y, b.z); A.p2.setXYZ(i, cc.x, cc.y, cc.z); A.p3.setXYZ(i, d.x, d.y, d.z);
        A.w.array[i] = w; A.layer.array[i] = layer; A.hue.array[i] = hue;
      }
    }
    function setWeight(c, w) { A.w.array.fill(w, c * perCurve, (c + 1) * perCurve); }
    const flush = () => Object.values(A).forEach(a => a.needsUpdate = true);
    return { setCurve, setWeight, flush, curves };
  }

  // Vue d'ensemble : chaque mot précédent → la colonne, à chaque couche
  const gateOverview = { value: 1 }, gateDive = { value: 0 };
  const overview = flowSystem(L * SLOTS, FLOW_P, gateOverview, true);
  for (let l = 0; l < L; l++) for (let j = 0; j < SLOTS; j++) {
    const x = crownX(l), [y, z] = slotDir(j);
    overview.setCurve(l * SLOTS + j,
      streamPos(j, x - 1.1),
      new THREE.Vector3(x - 0.5, y * R_STREAM * 0.8, z * R_STREAM * 0.8),
      new THREE.Vector3(x - 0.1, y * R_STREAM * 0.25, z * R_STREAM * 0.25),
      new THREE.Vector3(x + 0.35, 0, 0), 0, l - 0.5, 0);
  }
  overview.flush();

  // Plongée : mots → tête choisie, puis chaque tête → colonne
  const dive = flowSystem(SLOTS + H, DIVE_P, gateDive, false);

  // ------------------------------------------------------------------
  // 5. Têtes d'attention et neurones marqués (plongée uniquement)
  // ------------------------------------------------------------------
  function markerSystem(n, ring) {
    const g = new THREE.BufferGeometry();
    const pos = new THREE.BufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const size = new THREE.BufferAttribute(new Float32Array(n), 1).setUsage(THREE.DynamicDrawUsage);
    const hue = new THREE.BufferAttribute(new Float32Array(n), 1).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', pos); g.setAttribute('aSize', size); g.setAttribute('aHue', hue);
    const pts = new THREE.Points(g, additive(/* glsl */`
      attribute float aSize, aHue; uniform float uPix, uFocusAmt, uTime; varying float vHue, vOn;
      void main() {
        vHue = aHue; vOn = uFocusAmt;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * uFocusAmt * (1.0 + 0.08 * sin(uTime * 3.0)) * uPix * (30.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`, ROUND_POINT + /* glsl */`
      varying float vHue, vOn;
      void main() {
        float r = roundPoint();
        float a = ${ring ? 'smoothstep(0.32, 0.4, r) * (1.0 - smoothstep(0.44, 0.5, r))' : 'pow(1.0 - r*2.0, 1.2)'};
        vec3 col = mix(vec3(.75, .92, 1.0), vec3(1.0, .35, .7), vHue);
        gl_FragColor = vec4((col * a * vOn * 1.2) * uShow, 1.0);
      }`));
    pts.frustumCulled = false;
    net.add(pts);
    return { pos, size, hue, n };
  }
  const heads = markerSystem(H, false);
  const marks = markerSystem(10, true);

  // ------------------------------------------------------------------
  // 6. La boucle : le token choisi repart vers l'entrée, et deux anneaux entrée / sortie
  // ------------------------------------------------------------------
  const tIdx = new Float32Array(TRAIL).map((_, i) => i);
  const oGeo = new THREE.BufferGeometry();
  oGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
  oGeo.setAttribute('aIdx', new THREE.BufferAttribute(tIdx, 1));
  const orbU = { uLoop: { value: -1 }, uA: { value: new THREE.Vector3(exitX, 0, 0) }, uB: { value: new THREE.Vector3(exitX + 3, 8, -2) }, uC: { value: new THREE.Vector3(entryX - 3, 8, -2) }, uD: { value: new THREE.Vector3(entryX, 0, 0) } };
  const orb = new THREE.Points(oGeo, additive(BEZIER + /* glsl */`
    attribute float aIdx; uniform float uLoop, uPix; uniform vec3 uA, uB, uC, uD; varying float vI;
    void main() {
      float s = uLoop - aIdx * 0.007;
      float on = step(0.0, s) * step(s, 1.0) * step(0.0, uLoop);
      vec3 p = bez(uA, uB, uC, uD, clamp(s, 0.0, 1.0));
      vI = on * (1.0 - aIdx / ${TRAIL}.0);
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_PointSize = on * (aIdx < 0.5 ? 9.0 : 4.0 * vI + 1.0) * uPix * (30.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }`, ROUND_POINT + /* glsl */`
    varying float vI;
    void main() { float r = roundPoint(); gl_FragColor = vec4((vec3(1.0, .4, .72) * vI * 1.6 * pow(1.0 - r*2.0, 1.3)) * uShow, 1.0); }`, { ...orbU, uShow: SHOW.spine }));
  orb.frustumCulled = false;
  net.add(orb);

  function ring(x, color, lit) {
    const n = 260, g = new THREE.BufferGeometry(), p = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2, r = 0.95 + (rnd() - 0.5) * 0.12; p.set([x, Math.sin(a) * r, Math.cos(a) * r], i * 3); }
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    const u = { uLit: { value: 0 } };
    const pts = new THREE.Points(g, additive(/* glsl */`
      uniform float uPix, uLit, uTime; varying float vI;
      void main() { vI = 0.12 + uLit; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_PointSize = 1.6 * uPix * (30.0 / -mv.z); gl_Position = projectionMatrix * mv; }`,
      ROUND_POINT + /* glsl */`
      varying float vI; void main() { float r = roundPoint(); gl_FragColor = vec4((vec3(${color}) * vI * (1.0 - r*2.0)) * uShow, 1.0); }`, { ...u, uShow: SHOW.rings }));
    net.add(pts);
    return { pts, u, lit };
  }
  const rings = [ring(entryX, '1.0, .85, .55', w => w < -0.4 ? 0.9 : 0.0), ring(exitX, '1.0, .4, .72', w => w > L - 0.6 ? 0.9 : 0.0)];

  // ------------------------------------------------------------------
  // 7. L'échelle (intro) : chaque point du « cosmos » vaut un GPT-2 entier
  //    Les points sont rangés du centre vers l'extérieur : en afficher n, c'est remplir une boule de n GPT-2.
  // ------------------------------------------------------------------
  const COSMOS = 22700, CELL = 1.9;
  const cosmosR = n => CELL * Math.cbrt(n * 3 / (4 * Math.PI) / 0.62);
  const RC = cosmosR(COSMOS), cList = [];
  for (let i = 0; i < COSMOS; i++) {
    let x = gauss(), y = gauss(), z = gauss(); const d = Math.hypot(x, y, z) || 1;
    const r = RC * Math.cbrt(rnd());
    cList.push([x / d * r, y / d * r * 0.62, z / d * r, r]);
  }
  cList.sort((a, b) => a[3] - b[3]);
  cList[0] = [0, 0, 0, 0];                                         // le point 0 est le vrai GPT-2, au centre
  const cGeo = new THREE.BufferGeometry();
  cGeo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(cList.flatMap(c => c.slice(0, 3))), 3));
  cGeo.setAttribute('aIdx', new THREE.BufferAttribute(Float32Array.from(cList.map((_, i) => i)), 1));
  cGeo.setAttribute('aRand', new THREE.BufferAttribute(Float32Array.from(cList.map(() => rnd())), 1));
  const cosmosU = { uCount: { value: 0 }, uCosmos: { value: 0 } };
  const cosmos = new THREE.Points(cGeo, additive(/* glsl */`
    attribute float aIdx, aRand; uniform float uCount, uCosmos, uPix, uTime; varying float vI, vR;
    void main() {
      float on = step(0.5, aIdx) * step(aIdx, uCount - 1.0) * step(0.001, uCosmos);
      float fresh = 1.0 - clamp((uCount - aIdx) / (uCount * 0.12 + 40.0), 0.0, 1.0);   // les derniers arrivés brillent
      vR = aRand;
      vI = on * uCosmos * (0.5 + 0.9 * fresh) * (0.85 + 0.15 * sin(uTime * 2.0 + aRand * 40.0));
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = on * max(1.7 * uPix, (3.0 + aRand * 1.6) * uPix * (30.0 / -mv.z));
      gl_Position = projectionMatrix * mv;
    }`, ROUND_POINT + /* glsl */`
    varying float vI, vR;
    void main() {
      float r = roundPoint();
      vec3 col = mix(vec3(1.0, .4, .72), vec3(.35, .8, 1.0), vR);
      col = mix(col, vec3(1.0), pow(1.0 - r * 2.0, 3.0) * .6);
      gl_FragColor = vec4((col * vI * pow(1.0 - r*2.0, 1.5)) * uShow, 1.0);
    }`, cosmosU));
  cosmos.frustumCulled = false;
  scene.add(cosmos);

  const NEB = 60000, nbP = new Float32Array(NEB * 3), nbR = new Float32Array(NEB);
  for (let i = 0; i < NEB; i++) { nbP.set([gauss() * 120, gauss() * 80, gauss() * 120], i * 3); nbR[i] = rnd(); }
  const nbGeo = new THREE.BufferGeometry();
  nbGeo.setAttribute('position', new THREE.BufferAttribute(nbP, 3));
  nbGeo.setAttribute('aRand', new THREE.BufferAttribute(nbR, 1));
  const nebulaU = { uNebula: { value: 0 } };
  const nebula = new THREE.Points(nbGeo, additive(/* glsl */`
    attribute float aRand; uniform float uNebula, uPix, uTime; varying float vI, vR;
    void main() {
      float on = step(0.001, uNebula);
      vR = aRand;
      vI = uNebula * (0.12 + 0.22 * aRand) * (0.8 + 0.2 * sin(uTime * 0.7 + aRand * 30.0));
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = on * max(1.2 * uPix, (2.0 + aRand * 2.5) * uPix * (30.0 / -mv.z));
      gl_Position = projectionMatrix * mv;
    }`, ROUND_POINT + /* glsl */`
    varying float vI, vR;
    void main() {
      float r = roundPoint();
      vec3 col = mix(vec3(.45, .3, 1.0), vec3(1.0, .35, .75), vR);
      gl_FragColor = vec4((col * vI * (1.0 - r*2.0)) * uShow, 1.0);
    }`, nebulaU));
  nebula.frustumCulled = false;
  scene.add(nebula);

  // ------------------------------------------------------------------
  // État par étape et par plongée
  // ------------------------------------------------------------------
  function setStep(raw, a, scales) {
    for (let l = 0; l < L; l++) {
      const row = raw.mlp[l], sc = scales.actScale[l];
      for (let i = 0; i < M; i++) nAct[l * M + i] = Math.pow(Math.min(1, Math.max(0, row[i]) / sc), 0.7);
    }
    nActAttr.needsUpdate = true;
    spineU.uNorm.value = raw.res_norm.map(v => v / scales.normMax);
    memU.uN.value = a.n;
    for (let l = 0; l < L; l++) for (let j = 0; j < SLOTS; j++) overview.setWeight(l * SLOTS + j, j < a.n ? a.layers[l].flow[j] : 0);
    overview.flush();
  }

  function setDive(l, h, raw, a, scales) {
    const n = a.n, row = raw.attention[l][h];
    const layerMax = Math.max(...raw.head_norm[l]) || 1;
    const x = crownX(l), hp = headPos(l, h);
    for (let j = 0; j < SLOTS; j++) {
      const s = streamPos(j, x - 1.25), e = hp;
      dive.setCurve(j, s, new THREE.Vector3(s.x + 0.7, s.y, s.z), new THREE.Vector3(e.x - 0.5, e.y, e.z), e,
        j < n ? Math.sqrt(row[j]) : 0, 0, 0);
    }
    for (let k = 0; k < H; k++) {
      const s = headPos(l, k), e = new THREE.Vector3(x + 0.7, 0, 0), w = raw.head_norm[l][k] / layerMax;
      dive.setCurve(SLOTS + k, s, new THREE.Vector3(s.x + 0.35, s.y * 0.7, s.z * 0.7), new THREE.Vector3(e.x - 0.3, s.y * 0.15, s.z * 0.15), e,
        (k === h ? 1 : 0.45) * Math.sqrt(w), 0, 1);
    }
    dive.flush();
    for (let k = 0; k < H; k++) {
      const p = headPos(l, k);
      heads.pos.setXYZ(k, p.x, p.y, p.z);
      heads.size.array[k] = (k === h ? 11 : 4) + 8 * raw.head_norm[l][k] / layerMax;
      heads.hue.array[k] = k === h ? 1 : 0;
    }
    heads.pos.needsUpdate = heads.size.needsUpdate = heads.hue.needsUpdate = true;
    a.layers[l].top.slice(0, marks.n).forEach((t, k) => {
      const i = (l * M + t.i) * 3;
      marks.pos.setXYZ(k, nPos[i], nPos[i + 1], nPos[i + 2]);
      marks.size.array[k] = k < 5 ? 9 : 6;
      marks.hue.array[k] = k === 0 ? 1 : 0;
    });
    marks.pos.needsUpdate = marks.size.needsUpdate = marks.hue.needsUpdate = true;
  }

  // ------------------------------------------------------------------
  // Caméra : vol fluide, léger balancement au repos, décalage pour les panneaux
  // ------------------------------------------------------------------
  let tween = null, sway = true, swayT = 0, focusTarget = 0, followX = null;
  const off = { x: 0, y: 0, tx: 0, ty: 0 };
  controls.addEventListener('start', () => { sway = false; tween = null; controls.enabled = true; });
  const ease = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

  function flyTo(pos, target, dur = 1.4) {
    tween = { p0: camera.position.clone(), t0: controls.target.clone(), p1: pos, t1: target, k: 0, dur };
    controls.enabled = false;
  }
  function overviewView() {
    if (camera.aspect < 1) return { pos: new THREE.Vector3(-34, 11, 40), target: new THREE.Vector3(0, 0, 0) };
    const half = 32, t = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const dist = Math.max(40, half / (t * camera.aspect));
    // Légèrement de biais et en surplomb : on devine la profondeur des fils sans perdre la lecture de gauche à droite
    return { pos: new THREE.Vector3(-dist * 0.14, dist * 0.24, dist * 0.97), target: new THREE.Vector3(0, 0, 0) };
  }
  function diveView(l) {
    const target = new THREE.Vector3(layerX(l) - 1.0, 0, 0);
    const d = camera.aspect < 1 ? new THREE.Vector3(2.2, 3.2, 15) : new THREE.Vector3(2.8, 2.4, 9.8);
    return { pos: target.clone().add(d), target };
  }
  function home(dur = 1.4) {
    const v = overviewView();
    flyTo(v.pos, v.target, dur);
    tween.then = () => { sway = true; swayT = 0; };
  }
  // Travelling : caméra proche qui accompagne le signal. L'utilisateur peut tourner et zoomer autour.
  function followView(x) {
    const d = camera.aspect < 1 ? new THREE.Vector3(-8, 6, 26) : new THREE.Vector3(-7, 5, 17);
    const target = new THREE.Vector3(x, 0, 0);
    return { pos: target.clone().add(d), target };
  }
  function follow(x, dur = 1.2) {
    followX = x;
    const v = followView(x);
    flyTo(v.pos, v.target, dur);
    tween.then = () => { sway = true; swayT = 0; };
  }

  function resize(w, h) {
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    bloom.setSize(w, h);
    camera.aspect = w / h;
    applyOffset();
  }
  function setOffset(x, y) { off.tx = x; off.ty = y; }
  function applyOffset() {
    const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
    if (Math.abs(off.x) < 0.5 && Math.abs(off.y) < 0.5) camera.clearViewOffset();
    else camera.setViewOffset(w, h, off.x, off.y, w, h);
    camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------------
  // Sélection d'une sphère à la souris : intersection rayon / ellipsoïde, calculée à la main
  // ------------------------------------------------------------------
  const ray = new THREE.Raycaster();
  function pickLayer(ndcX, ndcY) {
    ray.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
    const o = ray.ray.origin, d = ray.ray.direction, R = [RADII[0] + 0.15, RADII[1] + 0.15, RADII[2] + 0.15];
    let best = -1, bestT = Infinity;
    for (let l = 0; l < L; l++) {
      const ox = (o.x - layerX(l)) / R[0], oy = o.y / R[1], oz = o.z / R[2];
      const dx = d.x / R[0], dy = d.y / R[1], dz = d.z / R[2];
      const A = dx * dx + dy * dy + dz * dz, B = 2 * (ox * dx + oy * dy + oz * dz), C = ox * ox + oy * oy + oz * oz - 1;
      const disc = B * B - 4 * A * C;
      if (disc < 0) continue;
      const t = (-B - Math.sqrt(disc)) / (2 * A);
      if (t > 0 && t < bestT) { bestT = t; best = l; }
    }
    return best;
  }

  const v = new THREE.Vector3();
  function project(p) {
    v.copy(p).project(camera);
    const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
    return { x: (v.x + 1) / 2 * w, y: (1 - v.y) / 2 * h, on: v.z < 1 && Math.abs(v.x) < 1.15 && Math.abs(v.y) < 1.15 };
  }

  // ------------------------------------------------------------------
  // Boucle
  // ------------------------------------------------------------------
  function frame(dt) {
    U.uTime.value += dt;
    U.uFocusAmt.value += (focusTarget - U.uFocusAmt.value) * Math.min(1, dt * 4);
    gateOverview.value = 1 - 0.85 * U.uFocusAmt.value;
    gateDive.value = U.uFocusAmt.value;
    rings.forEach(r => { r.u.uLit.value += (r.lit(U.uWave.value) - r.u.uLit.value) * Math.min(1, dt * 6); });

    if (Math.abs(off.tx - off.x) > 0.5 || Math.abs(off.ty - off.y) > 0.5) {
      off.x += (off.tx - off.x) * Math.min(1, dt * 5); off.y += (off.ty - off.y) * Math.min(1, dt * 5);
      applyOffset();
    } else if (off.x !== off.tx || off.y !== off.ty) { off.x = off.tx; off.y = off.ty; applyOffset(); }

    if (tween) {
      tween.k = Math.min(1, tween.k + dt / tween.dur);
      const e = ease(tween.k);
      camera.position.lerpVectors(tween.p0, tween.p1, e);
      controls.target.lerpVectors(tween.t0, tween.t1, e);
      if (tween.k >= 1) { const then = tween.then; tween = null; controls.enabled = true; then && then(); }
    } else if (followX !== null && Math.abs(followX - controls.target.x) > 0.001) {
      // on translate caméra et cible ensemble : l'angle choisi par l'utilisateur est conservé
      const dx = (followX - controls.target.x) * Math.min(1, dt * 3);
      camera.position.x += dx; controls.target.x += dx;
    }
    if (!tween && sway) {
      swayT += dt;
      const o = camera.position.clone().sub(controls.target), s = new THREE.Spherical().setFromVector3(o);
      s.theta += Math.cos(swayT * 0.07) * 0.07 * 0.22 * dt;     // dérivée d'un balancement de ±0,22 rad
      camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(s));
    }
    controls.update();
    composer.render();
  }

  return {
    L, layerX, crownX, entryX, exitX, streamPos, headPos,
    neuronPos: (l, i) => new THREE.Vector3(nPos[(l * M + i) * 3], nPos[(l * M + i) * 3 + 1], nPos[(l * M + i) * 3 + 2]),
    camera, controls,
    setStep, setDive,
    setWave: w => U.uWave.value = w,
    setReveal: r => U.uReveal.value = r,
    setFocus: l => { U.uFocus.value = l < 0 ? -10 : l; focusTarget = l < 0 ? 0 : 1; },
    setHover: l => U.uHover.value = l < 0 ? -10 : l,
    setLoop: p => orbU.uLoop.value = p,
    setBloom: on => bloom.enabled = on,
    get bloom() { return bloom.enabled; },
    flyTo, home, follow, overviewView, diveView, resize, setOffset, pickLayer, project, frame,
    setFollowX: x => { if (followX !== null) followX = x; },
    stopFollow: () => { followX = null; },
    get following() { return followX !== null; },
    orbX: p => { const u = 1 - p; return u*u*u*exitX + 3*u*u*p*(exitX + 3) + 3*u*p*p*(entryX - 3) + p*p*p*entryX; },
    stopSway: () => { sway = false; },
    setShow: (name, v) => { SHOW[name].value = v; },
    setNetScale: k => net.scale.setScalar(k),
    setCosmos: (count, alpha) => { cosmosU.uCount.value = count; cosmosU.uCosmos.value = alpha; },
    setNebula: a => { nebulaU.uNebula.value = a; },
  };
}

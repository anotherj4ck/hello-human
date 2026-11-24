// main.js — chargement des données, chronologie, interface, plongée dans une couche.
import { Vector3 } from 'three';
import { createScene } from './scene.js';
import { createIntro } from './intro.js';
import {
  analyze, pct, num, clean, word, q, paperName, HEAD_TYPES,
  describeHead, certainty, decideText, attentionText, mlpText,
} from './analysis.js';

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const say = t => { $('msg').textContent = t; $('msg').hidden = !t; };
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

// ------------------------------------------------------------------
// 1. Données produites par extract.py
// ------------------------------------------------------------------
const DATA = '../data/';
let manifest, raw;
try {
  manifest = await (await fetch(DATA + 'manifest.json')).json();
} catch (e) {
  say("Impossible de charger les données (data/manifest.json). En local, lance le serveur depuis le dossier du projet, et non depuis viewer/ : voir le README.");
  throw e;
}
if (manifest.format !== 2) {
  say("Ces données viennent de l'ancienne version d'extract.py. Remplace-le par la nouvelle, puis relance : python extract.py --tokens 17 --prompt \"I am an AI and I\"");
  throw new Error('format de données v1');
}
let loaded = 0;
raw = await Promise.all(manifest.steps.map(s => fetch(DATA + s.file).then(r => r.json())
  .then(j => { say(`Chargement des données : ${++loaded} / ${manifest.steps.length}`); return j; })));
say('');

const A = analyze(manifest, raw);
const { L, H, M, N } = A;
const S = createScene($('c'), { L, M, H, Tmax: raw[N - 1].tokens.length });
$('meta').textContent = `${manifest.model} · ${(manifest.params / 1e6).toFixed(0)} M de paramètres · ${(L * M).toLocaleString('fr-FR')} neurones affichés · données réelles enregistrées par extract.py`;

// ------------------------------------------------------------------
// 2. Chronologie : une étape = le signal traverse les couches, puis le choix et la boucle
// ------------------------------------------------------------------
// Durées d'une étape à la vitesse « × 1 », en secondes : la traversée des couches (WAVE), puis le choix et la boucle (HOLD)
const WAVE = 16.8, HOLD = 5.6, D = WAVE + HOLD, END = N * D - 0.001;
// La boucle rose garde la même part de la phase HOLD, quelle que soit sa durée :
// elle démarre après 5/28 de HOLD et dure 19/28 de HOLD (0,5 s puis 1,9 s quand HOLD valait 2,8 s).
const LOOP_START = HOLD * 5 / 28, LOOP_LEN = HOLD * 19 / 28;
const W0 = -1.6, W1 = L + 0.6;
let T = 0, playing = true, speed = 1, cur = -1, dive = null, dragging = false, wasFollowing = true, intro = null;

function state() {
  T = clamp(T, 0, END);
  const k = Math.min(N - 1, Math.floor(T / D)), local = T - k * D;
  const w = W0 + (W1 - W0) * Math.min(1, local / WAVE);
  let phase, lw = -1, sub = '';
  if (local >= WAVE) phase = 'choix';
  else if (w < -0.5) phase = 'entree';
  else if (w > L - 0.5) phase = 'sortie';
  else { phase = 'couche'; lw = clamp(Math.round(w), 0, L - 1); sub = w < lw - 0.1 ? 'attention' : 'mlp'; }
  const loop = local >= WAVE + LOOP_START ? Math.min(1, (local - WAVE - LOOP_START) / LOOP_LEN) : -1;
  return { k, local, w, phase, lw, sub, loop };
}

function applyStep(k) {
  cur = k;
  S.setStep(raw[k], A.step(k), A);
  buildStrip(k);
  if (dive) { if (!dive.manual) dive.h = A.step(k).layers[dive.l].best; refreshDive(); }
}

const setPlaying = p => { playing = p; $('play').textContent = p ? 'Pause' : 'Lecture'; };
function gotoStep(k) {
  k = clamp(k, 0, N - 1);
  T = dive ? k * D + WAVE - 0.001 : k * D;     // en plongée : l'étape est figée, calcul terminé
}

// ------------------------------------------------------------------
// 3. Bandeau du haut : la phrase et la piste des couches
// ------------------------------------------------------------------
let seqKey = '';
function renderSeq(st) {
  const choix = st.phase === 'choix' && !dive;
  const key = `${st.k}|${choix}`;
  if (key === seqKey) return; seqKey = key;
  const a = A.step(st.k), P = manifest.prompt_len;
  $('kicker').textContent = `GPT-2 écrit un token à la fois · étape ${st.k + 1} sur ${N}`;
  // Deux parties toujours étiquetées : le texte écrit par l'humain, puis la suite écrite par GPT-2.
  // L'espace initial du premier token de GPT-2 est retiré : l'étiquette fait déjà la séparation.
  const tok = (t, i) => `<span class="${i < P ? 'p' : 'g'}${i === a.n ? ' cur' : ''}">${esc(i === P ? clean(t).trimStart() : clean(t))}</span>`;
  const gen = a.tokens.slice(P).map((t, j) => tok(t, P + j));
  if (choix) gen.push(`<span class="new">${esc(gen.length ? clean(a.chosen) : clean(a.chosen).trimStart())}</span>`);
  $('seq').innerHTML = `<span class="part"><span class="who h">Humain</span>${a.tokens.slice(0, P).map(tok).join('')}</span>`
    + (gen.length ? `<span class="part"><span class="who ai">GPT-2</span>${gen.join('')}</span>` : '');
}

function buildStrip(k) {
  const a = A.step(k);
  $('strip').innerHTML = a.layers.map((ly, l) =>
    `<button type="button" class="cell${ly.fav.isChosen ? ' hit' : ''}" data-l="${l}"
      title="Après la couche ${l + 1}, le modèle prédirait ${esc(q(ly.fav.t))} (${pct(ly.fav.p, 0)}). ${esc(q(a.chosen))}, le mot finalement choisi : ${pct(ly.lens.p)}. Cliquer pour plonger.">
      <b>C${l + 1}</b><code>${esc(word(ly.fav.t))}</code><i style="width:${Math.max(2, ly.lens.p * 100)}%"></i></button>`).join('');
  stripKey = '';
}
let stripKey = '';
function renderStrip(st) {
  const passed = dive ? L : st.phase === 'choix' || st.phase === 'sortie' ? L : Math.max(0, Math.ceil(st.w - 0.3));
  const key = `${st.k}|${passed}|${st.lw}`;
  if (key === stripKey) return; stripKey = key;
  [...$('strip').children].forEach((c, l) => {
    c.classList.toggle('on', l < passed);
    c.classList.toggle('now', l === st.lw);
  });
}
$('strip').addEventListener('click', e => { const c = e.target.closest('[data-l]'); if (c) enterDive(+c.dataset.l); });

// ------------------------------------------------------------------
// 4. Explication en direct (bas de l'écran)
// ------------------------------------------------------------------
let statusKey = '';
function bars(list, chosen) {
  return `<div class="bars">${list.map(([t, p]) => `<div class="bar${t === chosen ? ' win' : ''}"><code>${esc(clean(t))}</code><div><i style="width:${p * 100}%"></i></div><span>${pct(p, 0)}</span></div>`).join('')}</div>`;
}
function renderStatus(st) {
  const key = `${st.k}|${st.phase}|${st.lw}|${st.sub}`;
  if (key === statusKey) return; statusKey = key;
  const a = A.step(st.k), curTok = q(a.tokens[a.n]);
  $('where').textContent = `Étape ${st.k + 1} sur ${N} · mot en cours ${curTok}`;
  let title, text, fin = '';
  switch (st.phase) {
    case 'entree':
      title = 'Entrée';
      text = `${curTok} entre dans le réseau sous forme d'une liste de ${manifest.d_model} nombres : la colonne dorée. Les fils fins autour sont les ${a.n} mots déjà écrits.`;
      break;
    case 'couche':
      title = `Couche ${st.lw + 1} sur ${L} · ${st.sub === 'attention' ? 'attention' : 'neurones'}`;
      text = st.sub === 'attention' ? attentionText(a, st.lw) : mlpText(a, st.lw, M);
      break;
    case 'sortie':
      title = 'Sortie';
      text = `En bout de réseau, le modèle donne une probabilité à chacun des ${manifest.vocab.toLocaleString('fr-FR')} tokens de son vocabulaire.`;
      break;
    default:
      title = `Choix : ${q(a.chosen)}`;
      text = `Il retient ${q(a.chosen)} (${pct(a.p)}) : il est ${certainty(a.p)}. ${decideText(a, L)} Le mot rejoint le texte, et tout recommence pour le suivant.`;
      fin = bars(a.next.slice(0, 5).map(n => [n.token, n.p]), a.chosen);
  }
  $('phase').textContent = title;
  $('explain').textContent = text;
  $('final').innerHTML = fin;
}

// ------------------------------------------------------------------
// 5. Plongée dans une couche
// ------------------------------------------------------------------
function enterDive(l, h) {
  setPlaying(false);
  if (cur < 0) applyStep(0);
  T = cur * D + WAVE - 0.001;
  const a = A.step(cur);
  dive = { l, h: h ?? a.layers[l].best, manual: h !== undefined };
  document.body.classList.add('diving');
  $('dive').hidden = false;
  S.setFocus(l);
  S.setReveal(1);
  S.stopSway();
  wasFollowing = S.following;
  S.stopFollow();
  const v = S.diveView(l);
  S.flyTo(v.pos, v.target, 1.5);
  refreshDive();
  updateOffsets();
}
function exitDive() {
  if (!dive) return;
  dive = null;
  document.body.classList.remove('diving');
  $('dive').hidden = true;
  S.setFocus(-1);
  S.setReveal(0);
  $('home').textContent = wasFollowing ? "Vue d'ensemble" : 'Suivre le signal';
  if (wasFollowing) S.follow(S.layerX(clamp(state().w, -1, L))); else S.home();
  statusKey = seqKey = stripKey = '';
  updateOffsets();
}
function setLayer(l) {
  l = clamp(l, 0, L - 1);
  if (!dive || l === dive.l) return;
  dive.l = l; dive.h = A.step(cur).layers[l].best; dive.manual = false;
  S.setFocus(l);
  const v = S.diveView(l);
  S.flyTo(v.pos, v.target, 1.0);
  refreshDive();
}
function setHead(h) { if (!dive) return; dive.h = h; dive.manual = true; refreshDive(); }

function refreshDive() {
  const { l, h } = dive, a = A.step(cur), rw = raw[cur], ly = a.layers[l], hd = ly.heads[h];
  S.setDive(l, h, rw, a, A);
  const row = rw.attention[l][h], maxNorm = Math.max(...rw.head_norm[l]);
  const lensRank = ly.lens.rank === 1 ? 'en tête' : `${ly.lens.rank.toLocaleString('fr-FR')}ᵉ`;
  $('dive').innerHTML = `
    <div class="dhead">
      <div><p class="kicker">Étape ${cur + 1} sur ${N} · mot en cours ${esc(q(a.tokens[a.n]))}</p><h2>Couche ${l + 1} <span>sur ${L}</span></h2></div>
      <div class="dnav">
        <button type="button" data-act="lprev" title="Couche précédente (↑)" aria-label="Couche précédente" ${l === 0 ? 'disabled' : ''}>▲</button>
        <button type="button" data-act="lnext" title="Couche suivante (↓)" aria-label="Couche suivante" ${l === L - 1 ? 'disabled' : ''}>▼</button>
        <button type="button" data-act="close" class="close" title="Retour à la vue d'ensemble (Échap)">Retour</button>
      </div>
    </div>
    <section>
      <h3>Ce que fait cette couche</h3>
      <p>Elle modifie le vecteur du mot de ${pct(ly.delta, 0)}. Part approximative de ce changement :</p>
      <div class="split"><i style="width:${ly.attnShare * 100}%"></i></div>
      <div class="legendline"><span>attention ${pct(ly.attnShare, 0)}</span><span>neurones ${pct(1 - ly.attnShare, 0)}</span></div>
    </section>
    <section>
      <h3>Attention · ${H} têtes (la barre indique leur influence réelle)</h3>
      <div class="heads">${ly.heads.map(x => `<button type="button" class="hd t-${x.type}${x.h === h ? ' sel' : ''}" data-h="${x.h}" title="Tête ${x.h + 1} (notation des articles : ${paperName(l, x.h)})"><b>T${x.h + 1}</b><span>${HEAD_TYPES[x.type].short}</span><i style="width:${x.norm / maxNorm * 100}%"></i></button>`).join('')}</div>
      <p><b>Tête ${h + 1}</b> · ${HEAD_TYPES[hd.type].label} (type probable) · ${hd.rank === 1 ? 'la plus influente' : `${hd.rank}ᵉ plus influente`} de la couche</p>
      <p>${esc(describeHead(hd, a.tokens))}</p>
      <p class="note">Où elle regarde, mot par mot :</p>
      <p class="heat">${a.tokens.map((t, j) => `<span class="${j === a.n ? 'cur' : ''}" style="--a:${Math.sqrt(row[j]).toFixed(3)}" title="${pct(row[j])}">${esc(clean(t))}</span>`).join('')}</p>
    </section>
    <section>
      <h3>Neurones · ${M.toLocaleString('fr-FR')} dans cette couche</h3>
      <p>${ly.active} sont nettement allumés (valeur supérieure à 1). Les plus forts :</p>
      <ol class="neurons">${ly.top.slice(0, 8).map(t => `<li><code>n° ${t.i}</code><b>${num(t.v)}</b><span>${esc(t.note)}</span></li>`).join('')}</ol>
      <p class="note">Ce qu'un neurone « signifie » reste en grande partie inconnu : on ne voit que quand il s'allume, sur les ${N} étapes enregistrées.</p>
    </section>
    <section>
      <h3>Prédiction provisoire</h3>
      <p>Si le modèle s'arrêtait après cette couche :</p>
      ${bars(ly.lens.top, a.chosen)}
      <p style="margin-top:8px">${esc(q(a.chosen))}, le mot finalement choisi, est ici ${lensRank} (${pct(ly.lens.p)}).</p>
      <p class="note">Technique de la « lentille logit » : on applique la sortie du modèle directement à ce stade du calcul. C'est une approximation, très parlante sur GPT-2.</p>
    </section>`;
}
$('dive').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.h !== undefined) setHead(+b.dataset.h);
  else if (b.dataset.act === 'close') exitDive();
  else if (b.dataset.act === 'lprev') setLayer(dive.l - 1);
  else if (b.dataset.act === 'lnext') setLayer(dive.l + 1);
});

// ------------------------------------------------------------------
// 6. Étiquettes accrochées à la 3D
// ------------------------------------------------------------------
const labelEls = new Map();
function updateLabels(st) {
  const out = [], a = A.step(st.k);
  const add = (key, pos, text, cls, alpha = 1, h) => out.push({ key, pos, text, cls, alpha, h });
  add('cur', new Vector3(S.entryX - 0.6, 0.6, 0), word(a.tokens[a.n]), 'cur', dive ? 0.5 : 1);
  if (dive) {
    const { l, h } = dive, row = raw[st.k].attention[l][h];
    for (let j = 0; j < a.n; j++) add(`ctx${j}`, S.streamPos(j, S.crownX(l) - 1.4), word(a.tokens[j]), 'ctx', 0.3 + 0.7 * Math.sqrt(row[j]));
    for (let k = 0; k < H; k++) add(`head${k}`, S.headPos(l, k), `T${k + 1}`, 'head' + (k === h ? ' sel' : ''), 1, k);
    a.layers[l].top.slice(0, 5).forEach((t, r) => add(`neu${r}`, S.neuronPos(l, t.i), `n° ${t.i}`, 'neu', r === 0 ? 1 : 0.75));
  } else if (st.phase === 'couche' && st.sub === 'attention') {
    a.layers[st.lw].sources.slice(0, 3).filter(s => s.share > 0.06)
      .forEach(s => add(`src${s.j}`, S.streamPos(s.j, S.crownX(st.lw) - 1.15), `${word(a.tokens[s.j])} ${pct(s.share, 0)}`, 'src'));
  }
  const passedTo = dive ? L : st.phase === 'choix' || st.phase === 'sortie' ? L : Math.floor(st.w - 0.25) + 1;
  for (let l = 0; l < Math.min(L, passedTo); l++) {
    const f = a.layers[l].fav;
    if (dive && l !== dive.l) continue;                 // en plongée, seule la couche observée
    const alpha = dive ? 1 : (l === passedTo - 1 ? 1 : 0.55);
    add(`lens${l}`, new Vector3(S.layerX(l), 2.75, 0), `${word(f.t)} ${pct(f.p, 0)}`, 'lens' + (f.isChosen ? ' hit' : ''), alpha);
  }
  if (!dive && st.phase === 'choix') add('out', new Vector3(S.exitX + 0.5, 1.2, 0), word(a.chosen), 'out');

  const used = new Set();
  for (const e of out) {
    let el = labelEls.get(e.key);
    if (!el) { el = document.createElement('div'); $('labels').appendChild(el); labelEls.set(e.key, el); }
    const p = S.project(e.pos);
    used.add(e.key);
    if (el._t !== e.text) { el.textContent = e.text; el._t = e.text; }
    const cls = 'lab ' + e.cls;
    if (el.className !== cls) el.className = cls;
    if (e.h !== undefined) el.dataset.h = e.h; else delete el.dataset.h;
    el.style.opacity = p.on ? e.alpha : 0;
    el.style.transform = `translate(${p.x.toFixed(1)}px, ${p.y.toFixed(1)}px) translate(-50%, -50%)`;
  }
  for (const [key, el] of labelEls) if (!used.has(key)) el.style.opacity = 0;
}
$('labels').addEventListener('click', e => { const t = e.target.closest('[data-h]'); if (t && dive) setHead(+t.dataset.h); });

// ------------------------------------------------------------------
// 7. Souris : survol et clic sur une sphère (un clic n'est pas un glissé)
// ------------------------------------------------------------------
const canvas = $('c');
let pointer = null, down = null, hover = -1;
canvas.addEventListener('pointermove', e => { pointer = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener('pointerleave', () => { pointer = null; });
canvas.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener('pointerup', e => {
  if (intro || !down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) { down = null; return; }
  down = null;
  const l = pick(e.clientX, e.clientY);
  if (l >= 0) dive ? setLayer(l) : enterDive(l);
});
function pick(x, y) {
  const r = canvas.getBoundingClientRect();
  return S.pickLayer(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
}
function updateHover() {
  const l = pointer && !down ? pick(pointer.x, pointer.y) : -1;
  if (l !== hover) { hover = l; S.setHover(l); canvas.style.cursor = l >= 0 ? 'pointer' : ''; }
  const tip = $('tip');
  if (l >= 0 && !(dive && dive.l === l)) {
    tip.hidden = false;
    tip.textContent = `Couche ${l + 1} · cliquer pour plonger`;
    tip.style.left = pointer.x + 'px'; tip.style.top = pointer.y + 'px';
  } else tip.hidden = true;
}

// ------------------------------------------------------------------
// 8. Commandes et clavier
// ------------------------------------------------------------------
$('play').addEventListener('click', () => { if (!playing && T >= END) T = 0; setPlaying(!playing); });
$('prev').addEventListener('click', () => { const st = state(); gotoStep(dive || st.local < 0.8 ? st.k - 1 : st.k); });
$('next').addEventListener('click', () => gotoStep(state().k + 1));
$('speed').addEventListener('change', e => speed = +e.target.value);
$('glow').addEventListener('click', () => { S.setBloom(!S.bloom); $('glow').textContent = `Lueur : ${S.bloom ? 'oui' : 'non'}`; });
$('home').addEventListener('click', () => {
  if (dive) { wasFollowing = false; exitDive(); return; }
  if (S.following) { S.stopFollow(); S.home(); } else S.follow(S.layerX(clamp(state().w, -1, L)));
  $('home').textContent = S.following ? "Vue d'ensemble" : 'Suivre le signal';
});
$('help').addEventListener('click', () => { $('legend').hidden = !$('legend').hidden; });
$('legendClose').addEventListener('click', () => { $('legend').hidden = true; });
const range = $('time');
range.max = END;
range.addEventListener('input', () => { T = +range.value; });
// Partie déjà jouée de la piste, en rose : le CSS la dessine à partir de --p (0 à 1)
let rangeFill = -1;
function setRangeFill() {
  const p = Math.round(T / END * 1000) / 1000;
  if (p !== rangeFill) { rangeFill = p; range.style.setProperty('--p', p); }
}
range.addEventListener('pointerdown', () => dragging = true);
addEventListener('pointerup', () => dragging = false);
addEventListener('keydown', e => {
  if (intro || e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = e.key;
  if (k === ' ') { e.preventDefault(); $('play').click(); }
  else if (k === 'ArrowLeft') { e.preventDefault(); $('prev').click(); }
  else if (k === 'ArrowRight') { e.preventDefault(); $('next').click(); }
  else if (k === 'ArrowUp' && dive) { e.preventDefault(); setLayer(dive.l - 1); }
  else if (k === 'ArrowDown' && dive) { e.preventDefault(); setLayer(dive.l + 1); }
  else if (k === 'Escape') exitDive();
  else if (k === 'Enter' && !dive) { const st = state(); enterDive(st.lw >= 0 ? st.lw : 0); }
});
$('replay').addEventListener('click', () => { $('legend').hidden = true; startIntro(); });

// ------------------------------------------------------------------
// Introduction : jouée à l'ouverture, sauf si l'adresse se termine par #experience
// ------------------------------------------------------------------
function startIntro() {
  if (dive) exitDive();
  setPlaying(false);
  T = 0;
  document.body.classList.add('intro');
  for (const el of labelEls.values()) el.style.opacity = 0;
  $('tip').hidden = true;
  intro = createIntro({ S, A, raw, manifest, onDone: endIntro });
}
function endIntro() {
  intro = null;
  document.body.classList.remove('intro');
  T = 0; cur = -1;
  statusKey = seqKey = stripKey = '';
  wasFollowing = true;
  $('home').textContent = "Vue d'ensemble";
  S.follow(S.entryX, 1.2);
  setPlaying(true);
  updateOffsets();
}
if (location.hash === '#experience') {
  if (innerWidth < 760) $('legend').hidden = true;        // sur téléphone, la légende reste sur demande (?)
  setTimeout(() => { $('legend').hidden = true; }, 16000);
} else {
  $('legend').hidden = true;
}

// ------------------------------------------------------------------
// 9. Mise en page : la 3D se centre dans l'espace laissé libre par les panneaux
// ------------------------------------------------------------------
function updateOffsets() {
  const w = innerWidth, h = innerHeight;
  if (dive) {
    if (w > 760) S.setOffset($('dive').offsetWidth / 2 + 12, 0);
    else S.setOffset(0, $('dive').offsetHeight / 2);
  } else {
    const top = $('topbar').getBoundingClientRect().bottom;
    const bottom = Math.min($('status').getBoundingClientRect().top, $('ctrl').getBoundingClientRect().top);
    S.setOffset(0, h / 2 - (top + bottom) / 2);
  }
}
function resize() { S.resize(innerWidth, innerHeight); updateOffsets(); }
addEventListener('resize', resize);
resize();
S.follow(S.entryX, 0.01);                                   // départ : caméra proche, à l'entrée du réseau

// ------------------------------------------------------------------
// 10. Boucle principale
// ------------------------------------------------------------------
let last = performance.now(), layoutTimer = 0;
function loop(now) {
  const dt = clamp((now - last) / 1000, 0, 0.1); last = now;   // le premier horodatage peut précéder « last »
  if (intro) { intro.update(dt); S.frame(dt); requestAnimationFrame(loop); return; }
  if (playing) { T += dt * speed; if (T >= END) { T = END; setPlaying(false); } }
  const st = state();
  if (st.k !== cur) applyStep(st.k);
  S.setWave(dive ? W1 : st.w);
  S.setLoop(dive ? -1 : st.loop);
  // la caméra suit le front du signal, puis le mot choisi pendant sa boucle de retour
  if (!dive) S.setFollowX(st.loop >= 0 ? S.orbX(st.loop) : st.phase === 'choix' ? S.exitX : S.layerX(clamp(st.w, -1, L)));
  renderSeq(st);
  if (!dive) { renderStrip(st); renderStatus(st); }
  if (!dragging) range.value = T;
  setRangeFill();
  if ((layoutTimer += dt) > 0.5) { layoutTimer = 0; updateOffsets(); }   // la hauteur des panneaux varie avec le texte
  updateHover();
  S.frame(dt);
  updateLabels(st);
  requestAnimationFrame(loop);
}
if (location.hash !== '#experience') startIntro();
requestAnimationFrame(loop);

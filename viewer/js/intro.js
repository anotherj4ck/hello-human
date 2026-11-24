// intro.js — l'introduction pour les visiteurs qui découvrent :
// une accroche, trois idées simples, le vertige d'échelle, puis l'entrée dans l'expérience.
//
// Tous les textes sont dans la fonction TEXTES ci-dessous : c'est la partie à réécrire à ta main.
import { Vector3 } from 'three';
import { pct, word } from './analysis.js';

// Références d'échelle, en paramètres (« réglages »)
//   GPT-3   : 175 milliards (OpenAI, 2020)
//   Kimi K3 : 2 800 milliards au total, 104 milliards actifs par token (Moonshot AI, juillet 2026)
const GPT3 = 175e9;
const KIMI_K3 = 2.8e12;

const fr = n => Math.round(n).toLocaleString('fr-FR');
const clamp01 = x => Math.min(1, Math.max(0, x));
const smooth = x => { x = clamp01(x); return x * x * (3 - 2 * x); };
const ramp = (t, t0, dur) => smooth((t - t0) / dur);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

function TEXTES(m, first, N, ratio) {
  const humain = m.prompt === 'I am an AI and I' && m.output.trim().startsWith('am not a scientist. I am a human being.');
  const params = Math.round(m.params / 1e6);
  return {
    traduction: humain ? 'Je suis une IA et je… ne suis pas un scientifique. Je suis un être humain. Je suis un être humain.' : '',
    accroche: [humain ? 'Une IA vient d’écrire qu’elle est humaine.' : 'Une IA vient d’écrire cette phrase.', 'Voici ce qui s’est passé dans sa tête.'],
    pourquoi: humain ? 'GPT-2 n’est pas un assistant : il prolonge le texte le plus probable. Et sur internet, « I am a human being » est presque toujours écrit par des humains.' : '',
    qui: ['L’humain a écrit :', 'GPT-2 a continué :'],   // au-dessus des deux lignes de l'accroche
    etiquettes: ['Humain', 'GPT-2'],                       // devant les deux parties de la phrase, pendant le montage
    tokens: [`Une IA ne lit pas des mots.`, `Elle découpe le texte en morceaux, les tokens. Chacun reçoit un numéro, puis devient une liste de ${m.d_model} nombres.`],
    traversee: [`Ces nombres traversent ${m.n_layers} couches de neurones artificiels.`, `${params} millions de réglages en tout${m.model === 'gpt2' ? ', appris en lisant 8 millions de pages web' : ''}.`],
    devine: [`Elle ne « répond » pas : elle devine.`, `Elle calcule la probabilité de chaque mot possible et en choisit un. Ici, « ${word(first.chosen)} », à ${pct(first.next[0].p, 0)}.`],
    boucle: [`Puis elle recommence, mot après mot.`, `Pour écrire cette phrase : ${N} passages complets dans le réseau.`],
    gpt2: [`Ça, c’est GPT-2. Une IA de 2019.`, `Tout ce que tu viens de voir tient dans ce point lumineux.`],
    gpt3: [`GPT-3, un an plus tard\u00a0: 175\u00a0milliards de réglages.`, `Chaque point est un GPT-2 entier. Il en faut ${fr(ratio.gpt3)}.`],
    kimi: [`Kimi K3 (2026), l’un des plus grands modèles ouverts : 2\u202f800 milliards de réglages.`, `Environ ${fr(ratio.kimi)} GPT-2. Seuls 104 milliards de réglages travaillent pour chaque token : c’est un modèle « à experts ».`],
    secret: [`ChatGPT, Claude, Gemini : leur taille n’est pas publiée.`, `Même principe, autre échelle.`],
    entree: [`Maintenant, entre dans GPT-2 pendant qu’il écrit.`, `Clique sur une sphère pour plonger dans une couche. Espace pour mettre en pause.`],
  };
}

export function createIntro({ S, A, raw, manifest, onDone }) {
  const L = manifest.n_layers, N = raw.length, first = raw[0];
  const ratio = { gpt3: Math.round(GPT3 / manifest.params), kimi: Math.round(KIMI_K3 / manifest.params / 100) * 100 };
  const TX = TEXTES(manifest, first, N, ratio);
  const prompt = manifest.prompt;
  const gen = manifest.output.trimStart();   // sur sa propre ligne, la réponse n'a pas besoin de son espace initial

  // ------------------------------------------------------------------
  // Calque HTML de l'intro
  // ------------------------------------------------------------------
  const root = document.createElement('div');
  root.className = 'intro black';
  root.innerHTML = `
    <div class="veil"></div>
    <div class="hook">
      <div class="type">
        <p class="line"><span class="who h"></span><span class="tp"></span><span class="caret"></span></p>
        <p class="line gl"><span class="who ai"></span><span class="tg"></span></p>
      </div>
      <p class="trans"></p>
      <p class="l1"></p><p class="why"></p><p class="l2"></p>
    </div>
    <p class="sentence"></p>
    <p class="counter"></p>
    <div class="chips"></div>
    <div class="probs"></div>
    <p class="big"></p>
    <div class="caption"><p class="c1"></p><p class="c2"></p></div>
    <span class="tag">GPT-2</span>
    <nav class="inav"><div class="dots"></div><button type="button" class="next">Entrer</button></nav>
    <button type="button" class="skip">Passer l’intro</button>`;
  document.body.appendChild(root);
  const $ = sel => root.querySelector(sel);
  const setCaption = (c) => { $('.c1').textContent = c ? c[0] : ''; $('.c2').textContent = c ? c[1] : ''; $('.caption').classList.toggle('on', !!c); };
  const show = (sel, on) => $(sel).classList.toggle('on', on);
  $('.trans').textContent = TX.traduction;
  $('.l1').textContent = TX.accroche[0];
  $('.why').textContent = TX.pourquoi;
  $('.l2').textContent = TX.accroche[1];
  $('.who.h').textContent = TX.qui[0];
  $('.who.ai').textContent = TX.qui[1];

  // La phrase du montage : le texte de l'humain, puis la suite écrite par GPT-2, chacun avec son étiquette
  function sentenceHTML(human, ai, chosen) {
    if (ai) ai = ai.trimStart(); else chosen = chosen.trimStart();   // l'étiquette fait déjà la séparation
    return `<span class="part"><span class="who h">${esc(TX.etiquettes[0])}</span><span>${esc(human)}</span></span>`
      + `<span class="part"><span class="who ai">${esc(TX.etiquettes[1])}</span>${esc(ai)}<em>${esc(chosen)}</em></span>`;
  }

  // ------------------------------------------------------------------
  // État de départ de la scène 3D : tout est éteint, caméra à l'entrée
  // ------------------------------------------------------------------
  const ALL = ['neurons', 'spine', 'memory', 'flows', 'rings'];
  const showAll = v => ALL.forEach(n => S.setShow(n, v));
  S.setStep(first, A.step(0), A);
  showAll(0);
  S.setWave(-1.6); S.setLoop(-1); S.setNetScale(1); S.setCosmos(0, 0); S.setNebula(0);
  S.setOffset(0, 0);
  S.follow(S.entryX, 0.01);
  const origin = new Vector3(0, 0, 0);

  // ------------------------------------------------------------------
  // Les scènes. Chacune : enter() à l'arrivée, update(t) à chaque image, end = durée de son animation.
  // ------------------------------------------------------------------
  const typing = {
    p0: 0.5, cp: 0.055, cg: 0.038,
    get g0() { return this.p0 + prompt.length * this.cp + 0.8; },
    get done() { return this.g0 + gen.length * this.cg; },
  };
  let flown = false, montage = null;

  const SCENES = [
    { // 1. L'accroche
      button: 'Entrer', end: typing.done + (TX.pourquoi ? 3.9 : 2.4),
      enter() { root.classList.add('black'); show('.hook', true); },
      update(t) {
        $('.tp').textContent = prompt.slice(0, Math.floor(clamp01((t - typing.p0) / (prompt.length * typing.cp)) * prompt.length));
        $('.tg').textContent = gen.slice(0, Math.floor(clamp01((t - typing.g0) / (gen.length * typing.cg)) * gen.length));
        // La ligne de GPT-2 apparaît quand il commence à écrire ; le curseur passe alors sur sa ligne
        const ai = t >= typing.g0 - 0.3, line = $(ai ? '.gl' : '.line');
        show('.gl', ai);
        if ($('.caret').parentNode !== line) line.appendChild($('.caret'));
        show('.trans', t > typing.done + 0.5 && !!TX.traduction);
        show('.l1', t > typing.done + 1.3);
        show('.why', t > typing.done + 2.2 && !!TX.pourquoi);       // même règle que la traduction
        show('.l2', t > typing.done + (TX.pourquoi ? 3.6 : 2.1));
      },
    },
    { // 2. Des mots aux nombres
      button: 'Suivant', end: 4.8,
      enter() {
        root.classList.remove('black'); show('.hook', false); flown = false;
        $('.chips').innerHTML = first.tokens.map((t, i) =>
          `<span class="chip" style="transition-delay:${i * 0.08}s"><b>${esc(word(t))}</b><i>${first.ids ? first.ids[i] : ''}</i></span>`).join('');
        requestAnimationFrame(() => root.querySelectorAll('.chip').forEach(c => c.classList.add('in')));
        setCaption(TX.tokens);
      },
      update(t) {
        if (t > 1.4) root.querySelectorAll('.chip').forEach(c => c.classList.add('num'));
        if (t > 3.0 && !flown) {
          flown = true;
          const p = S.project(new Vector3(S.entryX, 0, 0));
          root.querySelectorAll('.chip').forEach((c, i) => {
            const r = c.getBoundingClientRect();
            c.animate([{ transform: 'translate(0,0) scale(1)', opacity: 1 },
              { transform: `translate(${p.x - r.left - r.width / 2}px, ${p.y - r.top - r.height / 2}px) scale(.25)`, opacity: 0 }],
            { duration: 900, delay: i * 90, easing: 'cubic-bezier(.5,0,.3,1)', fill: 'forwards' });
          });
        }
        S.setShow('spine', ramp(t, 3.4, 0.9));
        S.setShow('rings', ramp(t, 3.4, 0.9));
        S.setShow('memory', 0.8 * ramp(t, 3.7, 1.0));
      },
      leave() { $('.chips').innerHTML = ''; ['spine', 'rings'].forEach(n => S.setShow(n, 1)); S.setShow('memory', 0.8); },
    },
    { // 3. La traversée
      button: 'Suivant', end: 7.0,
      enter() { setCaption(TX.traversee); show('.counter', true); },
      update(t) {
        S.setShow('neurons', 1);
        S.setShow('flows', ramp(t, 0.2, 0.6));
        const w = -1.6 + (L + 2.2) * smooth((t - 0.6) / 6.2);
        S.setWave(w);
        S.setFollowX(S.layerX(Math.max(-1, Math.min(L, w))));
        $('.counter').textContent = w < -0.5 ? 'Entrée' : w > L - 0.5 ? 'Sortie' : `Couche ${Math.round(w) + 1} sur ${L}`;
      },
      leave() { show('.counter', false); ['neurons', 'flows', 'memory'].forEach(n => S.setShow(n, 1)); },
    },
    { // 4. Elle devine, puis la boucle et le montage de toute la phrase
      button: 'Suivant', end: 6.4 + (N - 1) * 0.6 + 0.6,
      enter() {
        montage = { k: 0, home: false };
        setCaption(TX.devine);
        $('.probs').innerHTML = first.next.slice(0, 5).map((n, i) =>
          `<div class="pr${i === 0 ? ' win' : ''}"><code>${esc(word(n.token))}</code><div><i data-w="${n.p * 100}"></i></div><span>${pct(n.p, 0)}</span></div>`).join('');
        requestAnimationFrame(() => root.querySelectorAll('.pr i').forEach(i => i.style.width = i.dataset.w + '%'));
        show('.probs', true);
      },
      update(t) {
        if (t < 3.3) { S.setWave(L + 0.6); S.setFollowX(S.exitX); show('.probs', t > 0.3); root.querySelector('.pr')?.classList.toggle('lit', t > 1.3); return; }
        show('.probs', false);
        if (t < 5.6) {                                                        // la boucle : le mot repart à l'entrée
          const p = (t - 3.3) / 2.2;
          S.setLoop(p); S.setFollowX(S.orbX(p));
          setCaption(TX.boucle);
          $('.sentence').innerHTML = sentenceHTML(prompt, '', first.chosen);
          show('.sentence', true);
          return;
        }
        S.setLoop(-1);
        if (!montage.home) { montage.home = true; S.stopFollow(); S.home(1.2); show('.counter', true); }
        const k = Math.min(N - 1, 1 + Math.floor((t - 6.4) / 0.6));
        if (t >= 6.4 && k !== montage.k) {                                    // montage : les autres passages en accéléré
          montage.k = k;
          S.setStep(raw[k], A.step(k), A);
          const toks = raw[k].tokens, P = manifest.prompt_len ?? first.tokens.length;
          $('.sentence').innerHTML = sentenceHTML(toks.slice(0, P).join(''), toks.slice(P).join(''), raw[k].chosen);
        }
        const local = t < 6.4 ? 0 : ((t - 6.4) % 0.6) / 0.55;
        S.setWave(t >= 6.4 + (N - 1) * 0.6 ? L + 0.6 : -1.6 + (L + 2.2) * Math.min(1, local));
        $('.counter').textContent = `Passage ${Math.min(N, k + 1)} sur ${N}`;
      },
      leave() { S.setLoop(-1); show('.probs', false); show('.sentence', false); show('.counter', false); },
    },
    { // 5. Le vertige d'échelle, en quatre temps
      button: 'Suivant', subs: 4, end: 4.6,
      enter(sub) {
        S.stopFollow();
        const views = [[0, 9, 62, 3.2], [0, 14, 54, 3.5], [0, 36, 122, 4.5], [24, 46, 178, 4.5]];
        const [x, y, z, d] = views[sub];
        S.flyTo(new Vector3(x, y, z), origin.clone(), d);
        setCaption([TX.gpt2, TX.gpt3, TX.kimi, TX.secret][sub]);
        this.from = this.count ?? 1;
        if (sub >= 1) { S.setNetScale(0.03); ALL.forEach(n => S.setShow(n, 0.5)); }
      },
      update(t, sub) {
        if (sub === 0) {                                                      // GPT-2 se réduit à un point
          const k = ramp(t, 0.3, 3.0);
          S.setNetScale(1 - 0.97 * k);
          ALL.forEach(n => S.setShow(n, 1 - 0.5 * k));
          $('.big').textContent = '';
          this.count = 1;
        } else if (sub === 1 || sub === 2) {                                  // puis les autres modèles, en GPT-2
          const target = sub === 1 ? ratio.gpt3 : ratio.kimi;
          this.count = this.from + (target - this.from) * ramp(t, 0.4, sub === 1 ? 3.4 : 4.2);
          S.setCosmos(this.count, 1);
          $('.big').textContent = `× ${fr(this.count)}`;
        } else {                                                              // et l'inconnu
          S.setNebula(ramp(t, 0.3, 3.5));
          $('.big').textContent = '?';
        }
        show('.big', sub > 0);
        show('.tag', sub === 0 ? t > 2.8 : sub === 1 && t < 1.2);
      },
      leave() { show('.big', false); show('.tag', false); },
    },
    { // 6. L'entrée dans l'expérience
      button: 'Commencer', end: 2.6,
      enter() {
        setCaption(TX.entree);
        S.setStep(first, A.step(0), A);
        S.setWave(-1.6);
        S.follow(S.entryX, 2.4);
        this.c0 = SCENES[4].count ?? 0;
      },
      update(t) {
        const k = ramp(t, 0, 2.2);
        S.setNebula(1 - k > 0.001 ? (1 - k) * 0.9 : 0);
        S.setCosmos(this.c0 * (1 - k), 1 - k);
        S.setNetScale(0.03 + 0.97 * k);
        ALL.forEach(n => S.setShow(n, 0.5 + 0.5 * k));
      },
    },
  ];

  // ------------------------------------------------------------------
  // Navigation
  // ------------------------------------------------------------------
  let si = 0, sub = 0, t = 0, finished = false;
  $('.dots').innerHTML = SCENES.map(() => '<i></i>').join('');
  function enter(i, s = 0) {
    if (i !== si && SCENES[si].leave) SCENES[si].leave();
    si = i; sub = s; t = 0;
    SCENES[si].enter(sub);
    $('.next').textContent = SCENES[si].button;
    root.querySelectorAll('.dots i').forEach((d, k) => d.classList.toggle('on', k === si));
  }
  function next() {
    const sc = SCENES[si];
    if (t < sc.end && si === 0) { t = sc.end; return; }                       // un clic pendant la frappe : tout afficher
    if (sc.subs && sub < sc.subs - 1) { enter(si, sub + 1); return; }
    if (si < SCENES.length - 1) enter(si + 1); else finish();
  }
  function finish() {
    if (finished) return;
    finished = true;
    if (SCENES[si].leave) SCENES[si].leave();
    showAll(1); S.setNetScale(1); S.setCosmos(0, 0); S.setNebula(0); S.setLoop(-1);
    removeEventListener('keydown', onKey, true);
    root.remove();
    onDone();
  }
  function onKey(e) {
    if ([' ', 'Enter', 'ArrowRight', 'PageDown'].includes(e.key)) { e.preventDefault(); e.stopImmediatePropagation(); next(); }
    else if (e.key === 'Escape') { e.stopImmediatePropagation(); finish(); }
  }
  addEventListener('keydown', onKey, true);
  $('.next').addEventListener('click', e => { e.stopPropagation(); next(); });
  $('.skip').addEventListener('click', e => { e.stopPropagation(); finish(); });
  root.addEventListener('click', () => { if (root.classList.contains('black')) next(); });

  enter(0);

  return {
    update(dt) {
      if (finished) return;
      t += dt;
      SCENES[si].update.call(SCENES[si], t, sub);
      $('.next').classList.toggle('ready', t >= SCENES[si].end);
      if ($('.tag').classList.contains('on')) {
        const p = S.project(origin);
        $('.tag').style.transform = `translate(${p.x}px, ${p.y - 26}px) translate(-50%, -100%)`;
      }
    },
    skip: finish,
  };
}

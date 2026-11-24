// analysis.js — tout ce qu'on peut déduire des données enregistrées, sans 3D ni page web.
// Chaque phrase affichée par le site est générée ici, à partir des vrais chiffres.

export const pct = (p, digits = 1) =>
  (p * 100).toLocaleString('fr-FR', { maximumFractionDigits: digits }) + ' %';
export const num = (x, digits = 2) => x.toLocaleString('fr-FR', { maximumFractionDigits: digits });
export const clean = t => t.replace(/\n/g, '↵');                     // token affichable tel quel
export const word = t => clean(t).trim() || '␣';                        // sans l'espace initial
export const q = t => `« ${word(t)} »`;
export const headName = (l, h) => `C${l + 1}·T${h + 1}`;               // couche et tête, comptées à partir de 1
export const paperName = (l, h) => `${l}.${h}`;                          // notation des articles (à partir de 0)

export const HEAD_TYPES = {
  puits:     { label: 'Au repos',      short: 'repos' },
  induction: { label: 'Induction',     short: 'induction' },
  doublon:   { label: 'Doublon',       short: 'doublon' },
  precedent: { label: 'Mot précédent', short: 'précédent' },
  soi:       { label: 'Lui-même',      short: 'soi' },
  diffuse:   { label: 'Diffuse',       short: 'diffuse' },
  ciblee:    { label: 'Ciblée',        short: 'ciblée' },
};
const NOTABLE = new Set(['induction', 'doublon', 'precedent']);

// ------------------------------------------------------------------
// Type d'une tête d'après où elle regarde (heuristique, d'où « probable » dans l'interface)
// ------------------------------------------------------------------
export function classifyHead(row, tokens) {
  const n = row.length - 1;
  let j = 0;
  for (let i = 1; i <= n; i++) if (row[i] > row[j]) j = i;
  const a = row[j];
  const earlier = [];
  for (let i = 0; i < n; i++) if (tokens[i] === tokens[n]) earlier.push(i);

  if (j === 0 && a > 0.5) return { type: 'puits', j, a };
  if (a > 0.3 && j < n && earlier.includes(j - 1)) return { type: 'induction', j, a, src: j - 1 };
  if (a > 0.3 && earlier.includes(j)) return { type: 'doublon', j, a };
  if (j === n - 1 && a > 0.4) return { type: 'precedent', j, a };
  if (j === n && a > 0.4) return { type: 'soi', j, a };
  if (a < 0.25) return { type: 'diffuse', j, a };
  return { type: 'ciblee', j, a };
}

export function describeHead(c, tokens) {
  const n = tokens.length - 1, cur = q(tokens[n]), at = q(tokens[c.j]), p = pct(c.a, 0);
  switch (c.type) {
    case 'puits': return `Elle se pose sur le tout premier mot, ${at} (${p}). C'est sa position de repos quand elle n'a rien de précis à chercher : on parle de « puits d'attention ».`;
    case 'induction': return `On dirait une tête d'induction : elle a repéré que ${cur} est déjà apparu plus tôt, et regarde le mot qui le suivait alors, ${at} (${p}). C'est le mécanisme qui pousse un modèle à reproduire ce qu'il a déjà vu.`;
    case 'doublon': return `Elle retrouve l'apparition précédente du même mot, ${at} (${p}).`;
    case 'precedent': return `Elle regarde le mot juste avant : ${at} (${p}).`;
    case 'soi': return `Elle regarde surtout le mot lui-même, ${cur} (${p}).`;
    case 'diffuse': return `Son attention est répartie sur toute la phrase : aucun mot ne dépasse 25 %.`;
    default: return `Elle regarde surtout ${at} (${p}).`;
  }
}

// ------------------------------------------------------------------
// Analyse complète : échelles globales, puis chaque étape et chaque couche
// ------------------------------------------------------------------
const quantile = (arr, qq) => {
  if (!arr.length) return 1;
  const s = Float64Array.from(arr).sort();
  return s[Math.min(s.length - 1, Math.floor(s.length * qq))] || 1;
};

export function analyze(manifest, steps) {
  const L = manifest.n_layers, H = manifest.n_heads, M = manifest.d_mlp, N = steps.length;

  // Une échelle par couche pour les neurones : la valeur qui dépasse 99,5 % des activations positives,
  // sur toutes les étapes. Une même luminosité veut ainsi dire la même chose d'une étape à l'autre.
  const actScale = Array.from({ length: L }, (_, l) => {
    const pos = [];
    for (const s of steps) for (const v of s.mlp[l]) if (v > 0) pos.push(v);
    return quantile(pos, 0.995);
  });
  const normMax = Math.max(...steps.flatMap(s => s.res_norm));

  // Flux d'attention reçu de chaque mot (toutes têtes confondues), et son échelle globale
  const flows = steps.map(s => s.flow.map(layer => {
    const n = s.tokens.length;
    const f = new Float64Array(n);
    for (const row of layer) for (let j = 0; j < n; j++) f[j] += row[j];
    return f;
  }));
  const ctxFlows = [];
  flows.forEach((byLayer, k) => byLayer.forEach(f => { for (let j = 0; j < f.length - 1; j++) ctxFlows.push(f[j]); }));
  const flowScale = quantile(ctxFlows, 0.97);
  const headMax = Math.max(...steps.flatMap(s => s.head_norm.flat()));

  const lastTok = steps.map(s => s.tokens[s.tokens.length - 1]);

  // Spécificité d'un neurone : à quelles autres étapes s'allume-t-il aussi fort ?
  function neuronNote(l, i, k) {
    const v = steps[k].mlp[l][i];
    if (v <= 0) return '';
    const hits = [];
    steps.forEach((s, kk) => { if (s.mlp[l][i] >= 0.5 * v) hits.push(kk); });
    if (hits.length === 1) return `ne s'allume aussi fort qu'à cette étape`;
    const words = hits.map(kk => lastTok[kk]);
    if (hits.length <= 4 && words.every(w => w === words[0])) return `s'allume quand le mot en cours est ${q(words[0])}`;
    if (hits.length <= 4) return `rare : ${hits.length} étapes sur ${N}`;
    if (hits.length >= 0.8 * N) return `allumé presque à chaque étape`;
    return `allumé à ${hits.length} étapes sur ${N}`;
  }

  const cache = new Map();
  function step(k) {
    if (cache.has(k)) return cache.get(k);
    const s = steps[k], n = s.tokens.length - 1, chosen = s.chosen;
    const layers = [];
    for (let l = 0; l < L; l++) {
      const heads = s.attention[l].map((row, h) => ({ h, ...classifyHead(row, s.tokens), norm: s.head_norm[l][h] }));
      const byNorm = [...heads].sort((a, b) => b.norm - a.norm);
      byNorm.forEach((hd, r) => hd.rank = r + 1);
      const notable = byNorm.find(hd => NOTABLE.has(hd.type) && hd.rank <= 4) || null;
      const best = (heads.find(hd => hd.type === 'induction' && hd.rank <= 4) || notable || byNorm[0]).h;

      const f = flows[k][l];
      let total = 0;
      for (let j = 0; j < n; j++) total += f[j];
      const sources = Array.from({ length: n }, (_, j) => ({ j, v: f[j], share: total ? f[j] / total : 0 }))
        .sort((a, b) => b.v - a.v);

      const acts = s.mlp[l];
      const idx = Array.from(acts.keys()).sort((a, b) => acts[b] - acts[a]).slice(0, 10);
      let active = 0;
      for (const v of acts) if (v > 1) active++;

      const lensBefore = s.lens[l], lensAfter = s.lens[l + 1];
      const fav = lensAfter.top[0], favBefore = lensBefore.top[0];
      const a = s.attn_norm[l], m = s.mlp_norm[l];

      layers.push({
        heads, best, notable, sources,
        flow: Array.from({ length: n }, (_, j) => Math.min(1, Math.sqrt(f[j] / flowScale))),
        top: idx.map(i => ({ i, v: acts[i], note: neuronNote(l, i, k) })),
        active,
        lens: lensAfter, fav: { t: fav[0], p: fav[1], isChosen: fav[0] === chosen, isNew: fav[0] !== favBefore[0] },
        delta: s.delta[l], attnShare: a / (a + m || 1),
      });
    }
    // Couche à partir de laquelle le token choisi est en tête et n'en bouge plus
    let decide = null;
    for (let p = L; p >= 0 && s.lens[p].top[0][0] === chosen; p--) decide = p;
    const inductionHeads = layers.flatMap((ly, l) => ly.heads.filter(h => h.type === 'induction' && h.rank <= 4).map(h => headName(l, h.h)));
    const res = { n, tokens: s.tokens, chosen, p: s.next[0].p, next: s.next, lens: s.lens, layers, decide, inductionHeads };
    cache.set(k, res);
    return res;
  }

  return { L, H, M, N, actScale, normMax, flowScale, headMax, step, lastTok };
}

// ------------------------------------------------------------------
// Phrases pour l'interface
// ------------------------------------------------------------------
export function certainty(p) {
  if (p >= 0.6) return 'très sûr de lui';
  if (p >= 0.3) return 'assez sûr de lui';
  if (p >= 0.15) return 'hésitant';
  return 'très hésitant';
}

export function decideText(a, L) {
  if (a.decide === null) return '';
  if (a.decide === 0) return `${q(a.chosen)} est en tête dès l'entrée dans le réseau.`;
  if (a.decide === L) return `${q(a.chosen)} ne passe en tête qu'à la toute dernière couche.`;
  return `${q(a.chosen)} passe en tête après la couche ${a.decide} et ne la quitte plus.`;
}

export function attentionText(a, l) {
  const ly = a.layers[l], cur = q(a.tokens[a.n]);
  const [s1, s2] = ly.sources;
  let t = `${cur} consulte les ${a.n} mots précédents.`;
  if (s1) t += ` Ce qu'il en rapporte vient surtout de ${q(a.tokens[s1.j])} (${pct(s1.share, 0)})` + (s2 && s2.share > 0.1 ? ` et de ${q(a.tokens[s2.j])} (${pct(s2.share, 0)}).` : '.');
  if (ly.notable) {
    const hd = ly.notable;
    if (hd.type === 'induction') t += ` La tête ${hd.h + 1} a repéré une répétition : elle pointe vers ${q(a.tokens[hd.j])}, le mot qui suivait ${cur} plus tôt.`;
    else if (hd.type === 'doublon') t += ` La tête ${hd.h + 1} retrouve l'apparition précédente de ${cur}.`;
    else if (hd.type === 'precedent') t += ` La tête ${hd.h + 1} regarde le mot juste avant.`;
  }
  return t;
}

export function mlpText(a, l, M) {
  const ly = a.layers[l];
  let t = `${M.toLocaleString('fr-FR')} neurones transforment l'information : ${ly.active} sont nettement allumés.`;
  t += ` Après cette couche, le favori est ${q(ly.fav.t)} (${pct(ly.fav.p, 0)})`;
  t += ly.fav.isNew ? ' : il vient de changer.' : '.';
  return t;
}

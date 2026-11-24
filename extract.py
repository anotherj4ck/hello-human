"""
extract.py — v2 : enregistrer ce qui se passe à l'intérieur de GPT-2 pendant qu'il écrit.

À chaque token généré, on fait passer tout le texte dans le réseau et on capture,
pour le dernier token (celui dont le modèle prépare la suite) :

  attention   où regarde chacune des 12 têtes de chaque couche (scores d'attention)
  flow        ce que chaque tête rapporte réellement de chaque mot : attention × taille
              de l'information lue (norme de la « valeur » projetée). Un score élevé
              sur un mot qui n'apporte rien ne compte presque pas.
  head_norm   l'influence de chaque tête : taille de ce qu'elle écrit dans le flux
  mlp         l'activation des 3 072 neurones de chaque couche
  res_norm    la taille du vecteur du mot à l'entrée et après chaque couche
  delta       de combien chaque couche modifie ce vecteur (en %)
  attn_norm,  ce que la couche ajoute via l'attention et via les neurones
  mlp_norm
  lens        la « lentille logit » : ce que le modèle prédirait s'il s'arrêtait
              après chaque couche (on applique directement sa sortie à ce stade)
  next        les 10 tokens suivants les plus probables (sortie réelle)
  ids         le numéro de chaque token dans le vocabulaire du modèle

Le script vérifie lui-même la cohérence de ce qu'il capture (assertions) : si une
future version de transformers change la structure du modèle, il s'arrête au lieu
d'écrire des données fausses.

Usage :
    python extract.py --tokens 17 --prompt "I am an AI and I"
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FORMAT = 2


def r(t, nd=4):
    """Tenseur -> liste Python arrondie (JSON plus léger).
    Conversion en double AVANT l'arrondi : arrondi en float32, 0.001 s'écrirait 0.0010000000474974513."""
    return torch.round(t.double(), decimals=nd).tolist()


def as_in_out(layer):
    """Matrice de poids au format [entrée, sortie], que la couche soit un Conv1D (GPT-2) ou un Linear."""
    w = layer.weight
    return w.T if isinstance(layer, torch.nn.Linear) else w


def first(x):
    """Certains modules renvoient un tuple (sortie, extras) : on garde la sortie."""
    return x[0] if isinstance(x, (tuple, list)) else x


def record_pass(model, ids, tok, top_k=10, lens_k=5):
    cfg = model.config
    blocks = model.transformer.h
    L, H, d = cfg.n_layer, cfg.n_head, cfg.n_embd
    dh = d // H
    cap = {k: {} for k in ("in", "out", "attn", "weights", "v", "z", "act", "mlp")}
    hooks = []

    def keep(name, i, fn):
        def hook(*a):
            cap[name][i] = fn(*a).detach()
        return hook

    for i, b in enumerate(blocks):
        # Entrée et sortie du bloc entier (flux résiduel), pour le dernier token
        hooks.append(b.register_forward_hook(keep("in", i, lambda m, args, kw, out: (args[0] if args else kw["hidden_states"])[0, -1]), with_kwargs=True))
        hooks.append(b.register_forward_hook(keep("out", i, lambda m, inp, out: first(out)[0, -1])))
        # Sortie de l'attention (ce qu'elle ajoute au flux) et scores d'attention de toutes les têtes
        hooks.append(b.attn.register_forward_hook(keep("attn", i, lambda m, inp, out: first(out)[0, -1])))
        hooks.append(b.attn.register_forward_hook(keep("weights", i, lambda m, inp, out: out[1][0, :, -1])))
        # Valeurs V de toutes les positions : 3e tiers de la projection q/k/v
        hooks.append(b.attn.c_attn.register_forward_hook(keep("v", i, lambda m, inp, out: out[0, :, 2 * d:])))
        # Sorties des 12 têtes, concaténées, juste avant leur projection commune
        hooks.append(b.attn.c_proj.register_forward_pre_hook(keep("z", i, lambda m, inp: inp[0][0, -1])))
        # Neurones du MLP après activation, puis ce que le MLP ajoute au flux
        hooks.append(b.mlp.c_proj.register_forward_pre_hook(keep("act", i, lambda m, inp: inp[0][0, -1])))
        hooks.append(b.mlp.register_forward_hook(keep("mlp", i, lambda m, inp, out: first(out)[0, -1])))

    try:
        with torch.no_grad():
            out = model(ids)
    finally:
        for h in hooks:  # toujours retirer les hooks, même en cas d'erreur
            h.remove()

    with torch.no_grad():
        flow, head_norm = [], []
        for i, b in enumerate(blocks):
            w_o = as_in_out(b.attn.c_proj).view(H, dh, d)                       # [têtes, 64, 768]
            # Ce que chaque tête écrit dans le flux du dernier token
            per_head = torch.einsum("hk,hkd->hd", cap["z"][i].view(H, dh), w_o)   # [têtes, 768]
            assert torch.allclose(per_head.sum(0) + b.attn.c_proj.bias, cap["attn"][i], atol=1e-3), "décomposition des têtes incohérente"
            head_norm.append(per_head.norm(dim=-1))
            # Ce que chaque tête rapporte de chaque position : attention × norme de (valeur × sortie)
            v = cap["v"][i].view(-1, H, dh)                                        # [positions, têtes, 64]
            vo_norm = torch.einsum("shk,hkd->shd", v, w_o).norm(dim=-1)           # [positions, têtes]
            flow.append(cap["weights"][i] * vo_norm.T)                             # [têtes, positions]

        res = torch.stack([cap["in"][0]] + [cap["out"][i] for i in range(L)])      # [L+1, 768]
        for i in range(L):
            assert torch.allclose(cap["out"][i] - cap["in"][i], cap["attn"][i] + cap["mlp"][i], atol=1e-3), "flux résiduel incohérent"
            if i:
                assert torch.allclose(cap["in"][i], cap["out"][i - 1], atol=1e-4), "sortie et entrée de blocs incohérentes"

        # Lentille logit : appliquer la normalisation finale et la tête de sortie à chaque étape du flux
        lens_logits = model.lm_head(model.transformer.ln_f(res))                  # [L+1, vocab]
        assert torch.allclose(lens_logits[-1], out.logits[0, -1], atol=1e-2), "lentille logit incohérente avec la sortie réelle"

    probs = torch.softmax(out.logits[0, -1], dim=-1)
    top_p, top_i = torch.topk(probs, top_k)
    chosen = top_i[0].item()

    lens = []
    lens_probs = torch.softmax(lens_logits, dim=-1)
    for row_logits, row_p in zip(lens_logits, lens_probs):
        p, i = torch.topk(row_p, lens_k)
        lens.append({
            "top": [[tok.decode([t]), round(q, 4)] for q, t in zip(p.tolist(), i.tolist())],
            "p": round(row_p[chosen].item(), 5),                                   # proba du token finalement choisi
            "rank": int((row_logits > row_logits[chosen]).sum().item()) + 1,      # son rang à ce stade
        })

    delta = [((cap["out"][i] - cap["in"][i]).norm() / cap["in"][i].norm()).item() for i in range(L)]

    return {
        "tokens": [tok.decode([t]) for t in ids[0].tolist()],
        "ids": ids[0].tolist(),                                                    # numéro de chaque token dans le vocabulaire
        "attention": r(torch.stack([cap["weights"][i] for i in range(L)]), 4),   # [couche][tête][position]
        "flow": r(torch.stack(flow), 3),                                           # [couche][tête][position]
        "head_norm": r(torch.stack(head_norm), 3),                                 # [couche][tête]
        "mlp": r(torch.stack([cap["act"][i] for i in range(L)]), 3),               # [couche][neurone]
        "res_norm": r(res.norm(dim=-1), 3),                                        # [L+1]
        "delta": [round(x, 4) for x in delta],                                     # [couche]
        "attn_norm": r(torch.stack([cap["attn"][i].norm() for i in range(L)]), 3),
        "mlp_norm": r(torch.stack([cap["mlp"][i].norm() for i in range(L)]), 3),
        "lens": lens,                                                              # [L+1]
        "next": [{"token": tok.decode([i]), "p": round(p, 4)} for p, i in zip(top_p.tolist(), top_i.tolist())],
        "chosen": tok.decode([chosen]),
        "chosen_id": chosen,
    }


def run(model, tok, prompt, n_tokens, out_dir, model_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("step_*.json"):      # pas de mélange avec les fichiers d'un ancien lancement
        old.unlink()
    cfg = model.config
    ids = tok(prompt, return_tensors="pt").input_ids
    prompt_len = ids.shape[1]
    steps = []

    # Génération gloutonne (toujours le plus probable) : résultat reproductible.
    # Passage complet à chaque token, sans cache : chaque fichier contient l'état entier du réseau.
    for step in range(n_tokens):
        data = record_pass(model, ids, tok)
        name = f"step_{step:02d}.json"
        (out_dir / name).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        steps.append({"file": name, "input": "".join(data["tokens"]), "chosen": data["chosen"], "p": data["next"][0]["p"]})
        print(f"[{step + 1}/{n_tokens}] {''.join(data['tokens'])!r} -> {data['chosen']!r} ({data['next'][0]['p']:.1%})")
        if data["chosen_id"] == tok.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[data["chosen_id"]]])], dim=1)

    manifest = {
        "format": FORMAT,
        "model": model_name,
        "prompt": prompt,
        "prompt_len": prompt_len,
        "output": tok.decode(ids[0, prompt_len:]),
        "params": sum(p.numel() for p in model.parameters()),
        "n_layers": cfg.n_layer,
        "n_heads": cfg.n_head,
        "d_model": cfg.n_embd,
        "d_mlp": cfg.n_inner or 4 * cfg.n_embd,
        "vocab": cfg.vocab_size,
        "steps": steps,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTerminé : {len(steps)} étapes dans {out_dir.resolve()}")
    print(f"Réponse générée : {manifest['output']!r}")


def main():
    ap = argparse.ArgumentParser(description="Enregistre l'intérieur d'un LLM pendant qu'il génère.")
    ap.add_argument("--model", default="gpt2", help="modèle Hugging Face de la famille GPT-2 (défaut : gpt2, 124 M paramètres)")
    ap.add_argument("--prompt", default="I am an AI and I")
    ap.add_argument("--tokens", type=int, default=17, help="nombre de tokens à générer")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    # "eager" : l'implémentation classique de l'attention, la seule qui renvoie les scores
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")
    model.eval()
    run(model, tok, args.prompt, args.tokens, Path(args.out), args.model)


if __name__ == "__main__":
    main()

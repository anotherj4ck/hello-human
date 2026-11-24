"""
extract.py — Étape 1 : enregistrer ce qui se passe à l'intérieur d'un vrai LLM.

On donne un texte à GPT-2, on le laisse générer quelques tokens, et à chaque
passage dans le réseau on capture :
  - les tokens (entrée + tokens générés)
  - le flux résiduel après chaque couche (hidden states)
  - les scores d'attention de chaque tête de chaque couche
  - les activations du MLP (3 072 neurones par couche) pour le token en cours
  - les 10 tokens suivants les plus probables, avec leur probabilité

Le tout est écrit en JSON dans un dossier : c'est ce que le site rejouera.

Usage :
    python extract.py
    python extract.py --prompt "Hello world" --tokens 6 --out data
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def r(t, nd=4):
    """Tenseur -> liste Python arrondie (JSON plus léger)."""
    return torch.round(t.float(), decimals=nd).tolist()


def record_pass(model, input_ids, tokenizer, top_k=10):
    """Un passage complet dans le réseau, avec capture de l'intérieur."""
    mlp_acts = {}

    # Hook : juste avant la projection de sortie du MLP, on récupère
    # les 3 072 activations après GELU. On ne garde que le dernier token,
    # celui pour lequel le modèle est en train de prédire la suite.
    def make_hook(layer):
        def hook(_module, inputs):
            mlp_acts[layer] = inputs[0][0, -1].detach()
        return hook

    handles = [
        block.mlp.c_proj.register_forward_pre_hook(make_hook(i))
        for i, block in enumerate(model.transformer.h)
    ]
    try:
        with torch.no_grad():
            out = model(input_ids, output_attentions=True, output_hidden_states=True)
    finally:
        for h in handles:  # toujours retirer les hooks, même en cas d'erreur
            h.remove()

    probs = torch.softmax(out.logits[0, -1], dim=-1)
    top_p, top_i = torch.topk(probs, top_k)

    # hidden_states : (n_couches + 1) tenseurs [1, seq, 768] ; l'indice 0 = embeddings
    hidden = torch.stack(out.hidden_states)[:, 0]          # [13, seq, 768]
    attn = torch.stack(out.attentions)[:, 0]               # [12, têtes, seq, seq]
    mlp = torch.stack([mlp_acts[i] for i in range(len(mlp_acts))])  # [12, 3072]

    return {
        "tokens": [tokenizer.decode([t]) for t in input_ids[0].tolist()],
        "residual_norm": r(hidden.norm(dim=-1), 3),        # intensité du flux, par couche et par token
        "residual_last": r(hidden[:, -1], 3),              # vecteur complet du dernier token, par couche
        "attention": r(attn, 4),                           # [couche][tête][depuis][vers]
        "mlp_last": r(mlp, 3),                             # [couche][neurone]
        "next": [
            {"token": tokenizer.decode([i]), "p": round(p, 4)}
            for p, i in zip(top_p.tolist(), top_i.tolist())
        ],
        "chosen": tokenizer.decode([top_i[0].item()]),
        "chosen_id": top_i[0].item(),
    }


def main():
    ap = argparse.ArgumentParser(description="Enregistre l'intérieur d'un LLM pendant qu'il génère.")
    ap.add_argument("--model", default="gpt2", help="modèle Hugging Face (défaut : gpt2, 124 M paramètres)")
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--tokens", type=int, default=6, help="nombre de tokens à générer")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    # "eager" : l'implémentation classique de l'attention, la seule qui renvoie les scores
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")
    model.eval()
    run(model, tok, args.prompt, args.tokens, Path(args.out), args.model)


def run(model, tok, prompt, n_tokens, out_dir, model_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    ids = tok(prompt, return_tensors="pt").input_ids
    steps = []

    # Génération gloutonne (on prend toujours le plus probable) : résultat reproductible.
    # On refait le passage complet à chaque token, sans cache : plus lent,
    # mais chaque étape contient l'état entier du réseau.
    for step in range(n_tokens):
        data = record_pass(model, ids, tok)
        name = f"step_{step:02d}.json"
        (out_dir / name).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        steps.append({"file": name, "input": "".join(data["tokens"]), "chosen": data["chosen"]})
        print(f"[{step + 1}/{n_tokens}] {''.join(data['tokens'])!r} -> {data['chosen']!r} ({data['next'][0]['p']:.1%})")
        if data["chosen_id"] == tok.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[data["chosen_id"]]])], dim=1)

    manifest = {
        "model": model_name,
        "prompt": prompt,
        "output": tok.decode(ids[0, tok(prompt, return_tensors="pt").input_ids.shape[1]:]),
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


if __name__ == "__main__":
    main()

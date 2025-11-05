import os
import torch
from huggingface_hub import login as hf_login
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
from typing import Dict, List, Tuple
import yaml

DEFAULT_MODEL = ("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")

def _maybe_login_hf() -> None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        try:
            hf_login(token=tok, new_session=True)
        except Exception:
            pass

def load_conch(model_id: Tuple[str, str] = DEFAULT_MODEL):
    _maybe_login_hf()
    name, source = model_id
    model, preprocess = create_model_from_pretrained(name, source)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model_dtype = next(model.parameters()).dtype
    tokenizer = get_tokenizer()
    return device, model, preprocess, tokenizer, model_dtype

@torch.no_grad()
def build_text_features(prompts_dict: Dict[str, List[str]],
                        class_order: List[str],
                        tokenizer,
                        model,
                        device) -> torch.Tensor:
    feats = []
    for cls in class_order:
        texts = prompts_dict[cls]
        toks = tokenizer(texts)
        toks = toks.to(device)
        txt = model.encode_text(toks)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        mean_txt = txt.mean(dim=0)
        feats.append(mean_txt / mean_txt.norm())
    return torch.stack(feats, dim=0)

def build_text_features_from_yaml(yaml_path: str, tokenizer, model, device):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["classes"]
    prompts = cfg["prompts"]
    text_feats = build_text_features(prompts, classes, tokenizer, model, device)
    return classes, text_feats

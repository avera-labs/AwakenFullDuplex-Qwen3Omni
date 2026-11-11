"""Load a talker checkpoint for inference."""
from __future__ import annotations

import torch


def load_talker(ckpt, config, *, thinker_emb_path: str = "data/thinker_text_emb.pt", dtype=torch.bfloat16):
    """Build the talker LM from its model config, install the Qwen text path, then load the checkpoint. The order
    matters: the checkpoint's text-embedding weights load into the installed ThinkerTextEmb modules.
    A relative `thinker_emb_path` is read from the current directory; the realtime engine passes the absolute path
    under the model folder (paths.MODEL_ROOT)."""
    from safetensors.torch import load_file

    from duplexaudio.common.utils.config import load_yaml
    from duplexaudio.models.talker.modeling_talker import build_talker_lm, install_qwen_text

    cfg = load_yaml(str(config))
    lm = build_talker_lm(cfg["moshi_lm"], dtype)
    install_qwen_text(lm, thinker_emb_path, dtype)
    sd = {k: v.to("cuda", dtype) for k, v in load_file(str(ckpt)).items()}
    _, unexpected = lm.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0, unexpected[:5]
    return lm.eval()

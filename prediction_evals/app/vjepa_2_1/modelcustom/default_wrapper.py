"""
V-JEPA 2.1 wrapper for the IntPhys2 surprise evaluation.

The model code in app/vjepa_2_1/models is vendored from
facebookresearch/vjepa2 (commit 45d025f, "V-JEPA 2.1 (#130)").

V-JEPA 2.1 (ViT-g / ViT-G) uses deep self-supervision: the context encoder
outputs the concatenation of 4 normalized intermediate layers, and the
predictor regresses the concatenation of the same 4 layers of the target
encoder. For the surprise metric we only keep the LAST layer slice of both
the predictions and the targets. Targets are layer-normed (no affine) per
level, exactly as in app/vjepa_2_1/train.py::forward_target.

forward(x) returns (preds, targets) of shape [B, N_pred_tokens, embed_dim].
"""

import logging
import os
from urllib.parse import urlparse

import torch
import torch.nn.functional as F

import app.vjepa_2_1.models.predictor as vit_pred
import app.vjepa_2_1.models.vision_transformer as vit
from src.masks.utils import apply_masks

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _resolve_checkpoint_path(checkpoint: str) -> str:
    """Local path, or URL downloaded once into $TORCH_HOME/hub/checkpoints."""
    if not checkpoint.startswith(("http://", "https://")):
        return checkpoint
    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, os.path.basename(urlparse(checkpoint).path))
    if not os.path.exists(path):
        logger.info(f"Downloading {checkpoint} -> {path}")
        torch.hub.download_url_to_file(checkpoint, path, progress=True)
    return path


def _load_weights(module, state_dict, name):
    state_dict = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in state_dict.items()
    }
    msg = module.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded {name} with msg: {msg}")


def init_module(
    frames_per_clip: int,
    nb_context_frames: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    path = _resolve_checkpoint_path(checkpoint)
    # mmap avoids loading the full (~17 GB, incl. optimizer state) file in RAM
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=True)

    resolution = model_kwargs["resolution"]

    # ----------------------------------------------------------------------- #
    # Encoders (context + target share the same architecture)
    # ----------------------------------------------------------------------- #
    enc_kwargs = dict(model_kwargs["encoder"])
    enc_ckp_key = enc_kwargs.pop("checkpoint_key")
    enc_model_name = enc_kwargs.pop("model_name")
    target_ckp_key = model_kwargs["target_encoder"]["checkpoint_key"]

    def build_encoder():
        return vit.__dict__[enc_model_name](
            img_size=(resolution, resolution),
            num_frames=frames_per_clip,
            **enc_kwargs,
        )

    encoder = build_encoder()
    _load_weights(encoder, ckpt[enc_ckp_key], "encoder")
    target_encoder = build_encoder()
    _load_weights(target_encoder, ckpt[target_ckp_key], "target_encoder")

    # ----------------------------------------------------------------------- #
    # Predictor
    # ----------------------------------------------------------------------- #
    pred_kwargs = dict(model_kwargs["predictor"])
    pred_ckp_key = pred_kwargs.pop("checkpoint_key")
    pred_model_name = pred_kwargs.pop("model_name", "vit_predictor")
    predictor = vit_pred.__dict__[pred_model_name](
        img_size=(resolution, resolution),
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        num_frames=frames_per_clip,
        embed_dim=encoder.embed_dim,
        **pred_kwargs,
    )
    _load_weights(predictor, ckpt[pred_ckp_key], "predictor")
    del ckpt

    model = VJEPA21SurpriseWrapper(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        frames_per_clip=frames_per_clip,
        nb_context_frames=nb_context_frames,
        **wrapper_kwargs,
    )
    model.embed_dim = encoder.embed_dim
    return model


class VJEPA21SurpriseWrapper(torch.nn.Module):
    """Predict the future (non-context) tokens from the first nb_context_frames."""

    def __init__(
        self,
        encoder,
        target_encoder,
        predictor,
        frames_per_clip=64,
        nb_context_frames=16,
        mask_index=0,
    ):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        # frames_per_clip / nb_context_frames / grid_depth are overwritten by
        # evals/intphys2/eval.py before each forward pass.
        self.frames_per_clip = frames_per_clip
        self.nb_context_frames = nb_context_frames
        self.grid_depth = frames_per_clip // encoder.tubelet_size
        self.mask_index = mask_index

    def forward(self, x):
        """
        :param x: (Tensor) video of shape [B, C, T, H, W]
        """
        B, _, T, H, W = x.shape
        tubelet = self.encoder.tubelet_size
        patch = self.encoder.patch_size
        D = self.encoder.embed_dim
        assert self.nb_context_frames % tubelet == 0, "context must be a multiple of tubelet_size"

        tokens_per_step = (H // patch) * (W // patch)
        n_total = (T // tubelet) * tokens_per_step
        n_ctxt = (self.nb_context_frames // tubelet) * tokens_per_step

        ids = torch.arange(n_total, device=x.device)
        mask_ctxt = ids[:n_ctxt].unsqueeze(0).repeat(B, 1)
        mask_pred = ids[n_ctxt:].unsqueeze(0).repeat(B, 1)

        # -- targets: last hierarchical level of the target encoder, layer-normed
        h = self.target_encoder(x, training=True)  # [B, N, 4*D]
        h = F.layer_norm(h[..., -D:], (D,))
        targets = apply_masks(h, [mask_pred])

        # -- predictions: all 4 levels are needed as predictor input
        z = self.encoder(x, masks=[mask_ctxt], training=True)  # [B, N_ctxt, 4*D]
        preds, _ = self.predictor(
            z, [mask_ctxt], [mask_pred], mod="video", mask_index=self.mask_index
        )
        preds = preds[..., -D:]

        return preds, targets

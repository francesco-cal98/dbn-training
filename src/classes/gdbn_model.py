import math
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import wandb
from tqdm import tqdm
from sklearn.decomposition import PCA

# === Utility imports ===
from src.utils.wandb_utils import (
    plot_2d_embedding_and_correlations,
    plot_3d_embedding_and_correlations,
)
from src.utils.probe_utils import (
    log_linear_probe,
    compute_val_embeddings_and_features,
)


# -------------------------
# Helpers
# -------------------------
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


# -------------------------
# RBM (Bernoulli vis/hidden + gruppi softmax opzionali)
# -------------------------
class RBM(nn.Module):
    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        dynamic_lr: bool = False,
        final_momentum: float = 0.97,
        sparsity: bool = False,
        sparsity_factor: float = 0.05,
        softmax_groups: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.dynamic_lr = bool(dynamic_lr)
        self.final_momentum = float(final_momentum)
        self.sparsity = bool(sparsity)
        self.sparsity_factor = float(sparsity_factor)

        # compat vecchi pickle
        self.softmax_groups = softmax_groups or []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(
            torch.randn(self.num_visible, self.num_hidden, device=device) / math.sqrt(max(1, self.num_visible))
        )
        self.hid_bias = nn.Parameter(torch.zeros(self.num_hidden, device=device))
        self.vis_bias = nn.Parameter(torch.zeros(self.num_visible, device=device))

        # momentum buffers
        self.W_m = torch.zeros_like(self.W)
        self.hb_m = torch.zeros_like(self.hid_bias)
        self.vb_m = torch.zeros_like(self.vis_bias)
        # persistent chains for PCD
        self._pcd_v: Optional[torch.Tensor] = None

    # RBM.forward
    def forward(self, v: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        return sigmoid((v @ self.W + self.hid_bias) / max(1e-6, T))

    # RBM._visible_logits
    def _visible_logits(self, h: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        return (h @ self.W.T + self.vis_bias) / max(1e-6, T)

    # RBM.visible_probs
    def visible_probs(self, h: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        logits = self._visible_logits(h, T=T)
        v_prob = torch.sigmoid(logits)
        for s, e in getattr(self, "softmax_groups", []):
            v_prob[:, s:e] = torch.softmax(logits[:, s:e], dim=1)
        return v_prob


    # ---- sample v ~ p(v|h)
    def sample_visible(self, v_prob: torch.Tensor) -> torch.Tensor:
        v = (v_prob > torch.rand_like(v_prob)).float()
        groups = getattr(self, "softmax_groups", [])
        for s, e in groups:
            probs = v_prob[:, s:e].clamp(1e-8, 1)
            idx = torch.distributions.Categorical(probs=probs).sample()
            v[:, s:e] = 0.0
            v[torch.arange(v.size(0), device=v.device), s + idx] = 1.0
        return v

    # ---- decoder compatibile
    def backward(self, h: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        logits = self._visible_logits(h)
        if return_logits:
            return logits
        return self.visible_probs(h)

    @torch.no_grad()
    def backward_sample(self, h: torch.Tensor) -> torch.Tensor:
        return self.sample_visible(self.visible_probs(h))

    # ---- single Gibbs
    @torch.no_grad()
    def gibbs_step(self, v: torch.Tensor, sample_h: bool = True, sample_v: bool = True):
        h_prob = self.forward(v)
        h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob
        v_prob = self.visible_probs(h)
        v_next = self.sample_visible(v_prob) if sample_v else v_prob
        return v_next, v_prob, h, h_prob

    # ---- CD-k (supports PCD + y-block scaling for joint RBM)
    @torch.no_grad()
    def train_epoch(
        self,
        data: torch.Tensor,
        epoch: int,
        max_epochs: int,
        CD: int = 1,
        use_pcd: bool = True,
        clip_w: Optional[float] = None,
        clip_b: Optional[float] = None,
    ):
        # If requested, use the legacy unimodal CD-k update exactly as specified.
        if getattr(self, "_use_legacy_unimodal", False) and not getattr(self, "softmax_groups", []):
            lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
            mom = self.momentum if epoch <= 5 else self.final_momentum
            B = data.size(0)
            with torch.no_grad():
                # Positive phase
                pos_hid_probs = self.forward(data)
                pos_hid_states = (pos_hid_probs > torch.rand_like(pos_hid_probs)).float()
                pos_assoc = data.T @ pos_hid_probs

                # Negative phase (CD-k)
                neg_data = data
                for _ in range(int(CD)):
                    neg_vis_probs = self.backward(pos_hid_states)
                    neg_data = (neg_vis_probs > torch.rand_like(neg_vis_probs)).float()
                    neg_hid_probs = self.forward(neg_data)
                    pos_hid_states = (neg_hid_probs > torch.rand_like(neg_hid_probs)).float()
                neg_assoc = neg_data.T @ neg_hid_probs

                # Update weights
                self.W_m.mul_(mom).add_(
                    lr * ((pos_assoc - neg_assoc) / B - self.weight_decay * self.W)
                )
                self.W.add_(self.W_m)

                # Update hidden biases
                self.hb_m.mul_(mom).add_(lr * (pos_hid_probs.sum(0) - neg_hid_probs.sum(0)) / B)
                if self.sparsity:
                    Q = pos_hid_probs.sum(0) / B
                    avg_Q = Q.mean()
                    if avg_Q > self.sparsity_factor:
                        self.hb_m.add_(-lr * (Q - self.sparsity_factor))
                self.hid_bias.add_(self.hb_m)

                # Update visible biases
                self.vb_m.mul_(mom).add_(lr * (data.sum(0) - neg_data.sum(0)) / B)
                self.vis_bias.add_(self.vb_m)

                # Reconstruction error
                batch_loss = torch.sum((data - neg_vis_probs) ** 2) / B
            return batch_loss

        # Default: enhanced CD/PCD update used for joint/multimodal or non-legacy usage
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        mom = self.momentum if epoch <= 5 else self.final_momentum
        B = data.size(0)

        # positive
        pos_h = self.forward(data)
        pos_assoc = data.T @ pos_h

        # negative: PCD chains if available (robust to varying batch sizes)
        V = int(self.num_visible)
        if use_pcd and (self._pcd_v is not None) and (self._pcd_v.size(1) == V):
            if self._pcd_v.size(0) >= B:
                v = self._pcd_v[:B].clone()
            else:
                extra = torch.rand(B - self._pcd_v.size(0), V, device=data.device)
                v = torch.cat([self._pcd_v, extra], dim=0).clone()
        else:
            v = torch.rand(B, V, device=data.device)
        for _ in range(int(CD)):
            h_prob = self.forward(v)
            h = (h_prob > torch.rand_like(h_prob)).float()
            v_prob = self.visible_probs(h)
            v = self.sample_visible(v_prob)
        if use_pcd:
            if self._pcd_v is None or self._pcd_v.size(1) != V:
                self._pcd_v = v.detach()
            else:
                if self._pcd_v.size(0) >= B:
                    self._pcd_v[:B] = v.detach()
                else:
                    self._pcd_v = v.detach()
        neg_assoc = v.T @ h_prob

        # raw grads
        W_posneg = (pos_assoc - neg_assoc) / B
        dhb = (pos_h.sum(0) - h_prob.sum(0)) / B
        dvb = (data.sum(0) - v.sum(0)) / B

        # Standard update (no y-block scaling for unimodal)
        dW = W_posneg - self.weight_decay * self.W
        if clip_w is not None:
            dW = dW.clamp(-float(clip_w), float(clip_w))
        self.W_m.mul_(mom).add_(lr * dW)
        self.W.add_(self.W_m)

        upd_hb = lr * dhb
        upd_vb = lr * dvb
        if clip_b is not None:
            upd_hb = upd_hb.clamp(-float(clip_b), float(clip_b))
            upd_vb = upd_vb.clamp(-float(clip_b), float(clip_b))
        if self.sparsity:
            Q = pos_h.mean(0)
            upd_hb = upd_hb + (-lr * (Q - self.sparsity_factor))
        self.hb_m.mul_(mom).add_(upd_hb); self.hid_bias.add_(self.hb_m)
        self.vb_m.mul_(mom).add_(upd_vb); self.vis_bias.add_(self.vb_m)

        loss = torch.mean((data - v_prob) ** 2)
        return loss


# -------------------------
# iDBN (immagini) — con PCA/Probes/AutoRecon
# -------------------------
class iDBN:
    def __init__(self, layer_sizes, params, dataloader, val_loader, device, wandb_run=None, logging_config_path: Optional[str] = None):
        self.layers: List[RBM] = []
        self.params = params
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.device = device
        self.wandb_run = wandb_run
        # load logging config if present
        self.logging_cfg = {}
        try:
            import yaml
            from pathlib import Path
            cfg_path = Path(logging_config_path) if logging_config_path else Path("src/configs/logging_config.yaml")
            if cfg_path.exists():
                with cfg_path.open("r") as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    self.logging_cfg = cfg
        except Exception:
            pass

        # campi attesi dalle tue utils
        self.text_flag = False
        self.arch_str = "-".join(map(str, layer_sizes))
        self.arch_dir = os.path.join("logs-idbn", f"architecture_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

        self.cd_k = int(self.params.get("CD", 1))
        self.sparsity_last = bool(self.params.get("SPARSITY", False))
        self.sparsity_factor = float(self.params.get("SPARSITY_FACTOR", 0.1))

        # cache val
        try:
            self.val_batch, self.val_labels = next(iter(val_loader))
        except Exception:
            self.val_batch, self.val_labels = None, None

        # features complete (no shuffle sul val_loader!)
        self.features = None
        try:
            indices = val_loader.dataset.indices
            base = val_loader.dataset.dataset
            numeric_labels = torch.tensor([base.labels[i] for i in indices], dtype=torch.float32)
            cumArea_vals = [base.cumArea_list[i] for i in indices]
            convex_hull = [base.CH_list[i] for i in indices]
            density_src = getattr(base, "density_list", None)
            density_vals = [density_src[i] for i in indices] if density_src is not None else None
            self.features = {
                "Cumulative Area": torch.tensor(cumArea_vals, dtype=torch.float32),
                "Convex Hull": torch.tensor(convex_hull, dtype=torch.float32),
                "Labels": numeric_labels,
            }
            if density_vals is not None:
                self.features["Density"] = torch.tensor(density_vals, dtype=torch.float32)
        except Exception:
            pass

        # costruzione RBM
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=self.params["LEARNING_RATE"],
                weight_decay=self.params["WEIGHT_PENALTY"],
                momentum=self.params["INIT_MOMENTUM"],
                dynamic_lr=self.params["LEARNING_RATE_DYNAMIC"],
                final_momentum=self.params["FINAL_MOMENTUM"],
                sparsity=(self.sparsity_last and i == len(layer_sizes) - 2),
                sparsity_factor=self.sparsity_factor,
            ).to(self.device)
            # Force legacy unimodal CD-k update for plain iDBN training
            rbm._use_legacy_unimodal = True
            self.layers.append(rbm)

    # quali layer monitorare (come nei tuoi log)
    def _layers_to_monitor(self) -> List[int]:
        layers = {len(self.layers)}
        if len(self.layers) > 1:
            layers.add(1)
        return sorted(layers)

    def _layer_tag(self, idx: int) -> str:
        return f"layer{idx}"

    # TRAIN con autorecon + PCA + probes
    def train(self, epochs: int, log_every_pca: int = 25, log_every_probe: int = 10):
        for epoch in tqdm(range(int(epochs)), desc="iDBN"):
            losses = []
            for img, _ in self.dataloader:
                v = img.to(self.device).view(img.size(0), -1).float()
                for rbm in self.layers:
                    loss = rbm.train_epoch(v, epoch, epochs, CD=self.cd_k)
                    v = rbm.forward(v)
                    losses.append(float(loss))
            if self.wandb_run and losses:
                self.wandb_run.log({"idbn/loss": float(np.mean(losses)), "epoch": epoch})

            # Auto-recon snapshot
            if self.wandb_run and self.val_batch is not None and epoch % 5 == 0:
                with torch.no_grad():
                    rec = self.reconstruct(self.val_batch[:8].to(self.device))
                img0 = self.val_batch[:8]
                try:
                    B, C, H, W = img0.shape
                    recv = rec.view(B, C, H, W).clamp(0, 1)
                except Exception:
                    side = int(rec.size(1) ** 0.5)
                    C = 1
                    H = W = side
                    recv = rec.view(-1, C, H, W).clamp(0, 1)
                    img0 = img0.view(-1, C, H, W)
                grid = vutils.make_grid(torch.cat([img0.cpu(), recv.cpu()], dim=0), nrow=img0.size(0))
                self.wandb_run.log({"idbn/auto_recon_grid": wandb.Image(grid.permute(1, 2, 0).numpy()), "epoch": epoch})
                try:
                    mse = F.mse_loss(img0.to(self.device).view(img0.size(0), -1), recv.view(img0.size(0), -1))
                    self.wandb_run.log({"idbn/auto_recon_mse": mse.item(), "epoch": epoch})
                except Exception:
                    pass

            # PCA + PROBES per-layer
            if self.wandb_run and self.val_loader is not None and self.features is not None:
                if epoch % log_every_pca == 0:
                    for layer_idx in self._layers_to_monitor():
                        tag = self._layer_tag(layer_idx)
                        try:
                            E, feats = compute_val_embeddings_and_features(self, upto_layer=layer_idx)
                            if E.numel() == 0:
                                continue
                            emb_np = E.numpy()
                            feat_map = {
                                "Cumulative Area": feats["cum_area"].numpy(),
                                "Convex Hull": feats["convex_hull"].numpy(),
                                "Labels": feats["labels"].numpy(),
                            }
                            if "density" in feats:
                                feat_map["Density"] = feats["density"].numpy()
                            if emb_np.shape[0] > 2 and emb_np.shape[1] > 2:
                                p2 = PCA(n_components=2).fit_transform(emb_np)
                                plot_2d_embedding_and_correlations(
                                    emb_2d=p2,
                                    features=feat_map,
                                    arch_name=f"iDBN_{tag}",
                                    dist_name="val",
                                    method_name="pca",
                                    wandb_run=self.wandb_run,
                                )
                                if emb_np.shape[1] >= 3:
                                    p3 = PCA(n_components=3).fit_transform(emb_np)
                                    plot_3d_embedding_and_correlations(
                                        emb_3d=p3,
                                        features=feat_map,
                                        arch_name=f"iDBN_{tag}",
                                        dist_name="val",
                                        method_name="pca",
                                        wandb_run=self.wandb_run,
                                    )
                        except Exception as e:
                            self.wandb_run.log({f"warn/idbn_pca_error_{tag}": str(e)})

                if epoch % log_every_probe == 0:
                    for layer_idx in self._layers_to_monitor():
                        tag = self._layer_tag(layer_idx)
                        try:
                            log_linear_probe(
                                self,
                                epoch=epoch,
                                n_bins=5,
                                test_size=0.2,
                                steps=1000,
                                lr=1e-2,
                                patience=20,
                                min_delta=0.0,
                                upto_layer=layer_idx,
                                layer_tag=tag,
                            )
                        except Exception as e:
                            self.wandb_run.log({f"warn/idbn_probe_error_{tag}": str(e)})

    @torch.no_grad()
    def represent(self, x: torch.Tensor, upto_layer: Optional[int] = None) -> torch.Tensor:
        v = x.view(x.size(0), -1).float().to(self.device)
        L = len(self.layers) if (upto_layer is None) else max(0, min(len(self.layers), int(upto_layer)))
        for i in range(L):
            v = self.layers[i].forward(v)
        return v

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        v = x.view(x.size(0), -1).float().to(self.device)
        cur = v
        for rbm in self.layers:
            cur = rbm.forward(cur)
        for rbm in reversed(self.layers):
            cur = rbm.backward(cur)
        return cur

    def decode(self, top: torch.Tensor) -> torch.Tensor:
        cur = top.to(self.device)
        for rbm in reversed(self.layers):
            cur = rbm.backward(cur)
        return cur

    def save_model(self, path: str):
        model_copy = {"layers": self.layers, "params": self.params}
        with open(path, "wb") as f:
            pickle.dump(model_copy, f)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MedicalConceptModel(nn.Module):
    def __init__(
        self,
        model_name="aysangh/medsiglip-448-vindr-bin",  # backbone
        num_concepts=12,
        num_classes=1,
        projection_dim=128,
        prototypes_per_concept=4,
        freeze_backbone=True,
        contrastive_lambda=10.0,  # λ sharpening for loss
        contrastive_gamma=10.0,   # γ for assignment softmax
        contrastive_delta=0.1,    # margin δ
    ):
        super().__init__()

        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        self.M = prototypes_per_concept

        # backbone
        print(f"[Model] Loading backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)

        
        try:
            self.feature_dim = self.backbone.vision_model.config.hidden_size
        except:
            self.feature_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---- Concept head: 1x1 conv to produce K CAMs (no sigmoid) ----
        # Output shape: (B, K, H, W)
        self.concept_head = nn.Conv2d(self.feature_dim, self.num_concepts, kernel_size=1)

        # ---- Projector P: projects patch features to projection_dim ----
        # We apply projector per patch (i.e. Linear on channel dim)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        )

        # ---- Prototypes: (num_concepts, M, projection_dim) learnable ----
        # Initialize and then L2-normalize in forward / loss
        prot = torch.randn(self.num_concepts, self.M, self.projection_dim)
        prot = F.normalize(prot, dim=-1)
        self.prototypes = nn.Parameter(prot)  # will be learned

        # ---- Classifier: takes similarity vector s (K*M or aggregated K) ----
        # We'll let the classifier accept aggregated sim per concept (K) by default.
        # If you prefer classifier over K*M raw scores, change input dim accordingly.
        self.classifier = nn.Linear(self.num_concepts, self.num_classes)

        # contrastive hyperparams
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_gamma = contrastive_gamma
        self.contrastive_delta = contrastive_delta

    def _get_feature_map(self, x):
        """
        Backbones differ. Many HF vision models expose vision_model and return last_hidden_state.
        We'll try common access patterns.
        Returns feature_map of shape (B, C, H, W).
        """
        # Try vision_model API (e.g., SigLIP)
        try:
            outputs = self.backbone.vision_model(pixel_values=x)
            last_hidden = outputs.last_hidden_state  # (B, L, D)
        except Exception:
            # fallback: assume model returns last_hidden_state directly
            outputs = self.backbone(pixel_values=x)
            last_hidden = outputs.last_hidden_state  # (B, L, D)

        B, L, D = last_hidden.shape
        # if CLS token exists, remove it
        # detect if L is perfect square
        H_grid = int(L ** 0.5)
        if H_grid * H_grid == L:
            patch_tokens = last_hidden  # (B, L, D)
            W_grid = H_grid
        else:
            # assume first token is CLS
            patch_tokens = last_hidden[:, 1:, :]
            L_new = patch_tokens.shape[1]
            H_grid = int(L_new ** 0.5)
            W_grid = H_grid

        feature_map = patch_tokens.permute(0, 2, 1).contiguous().view(B, D, H_grid, W_grid)
        return feature_map  # (B, C, H, W)

    def _spatial_softmax(self, cams):
        """
        cams: (B, K, H, W) - raw scores (no activation)
        return weights: (B, K, H, W) normalized over spatial dims (h,w)
        """
        B, K, H, W = cams.shape
        cams_flat = cams.view(B, K, -1)  # (B, K, H*W)
        weights = F.softmax(cams_flat, dim=-1)  # (B, K, H*W)
        weights = weights.view(B, K, H, W)
        return weights

    def compute_local_concept_vectors(self, feature_map, cams):
        """
        feature_map: (B, C, H, W)
        cams: (B, K, H, W) raw
        return v_local: (B, K, C) -- each concept per image
        """
        B, C, H, W = feature_map.shape
        B2, K, H2, W2 = cams.shape
        assert B == B2 and H == H2 and W == W2

        weights = self._spatial_softmax(cams)  # (B, K, H, W)
        # expand feature_map to (B, 1, C, H, W) and weights (B, K, 1, H, W)
        feat = feature_map.unsqueeze(1)  # (B,1,C,H,W)
        w = weights.unsqueeze(2)        # (B,K,1,H,W)
        weighted = feat * w             # (B,K,C,H,W)
        v_local = weighted.sum(dim=[3, 4])  # (B, K, C)
        return v_local  # not projected yet

    def _project_patches(self, feature_map):
        """
        feature_map: (B, C, H, W)
        returns:
            patches_proj: (B, H*W, P) normalized
            patches_raw: (B, H*W, C)
            H, W for reconstruction if needed
        """
        B, C, H, W = feature_map.shape
        patches = feature_map.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, H*W, C)
        # project
        proj = self.projector(patches)  # (B, H*W, P)
        proj_norm = F.normalize(proj, p=2, dim=-1)
        return proj_norm, patches, H, W

    def compute_similarity_maps_and_scores(self, patches_proj, H, W):
        """
        patches_proj: (B, S, P) where S = H*W
        prototypes: (K, M, P)
        returns:
            sim_maps: (B, K, M, H, W) = cosine between p_km and each patch
            skm_max: (B, K, M) = max over spatial
        """
        B, S, P = patches_proj.shape
        K, M, Pp = self.prototypes.shape
        assert P == Pp

        # normalize prototypes
        prototypes = F.normalize(self.prototypes, dim=-1)  # (K, M, P)
        # reshape prototypes to (K*M, P)
        prot_flat = prototypes.view(K * M, P)  # (K*M, P)

        # compute dot product between patches and prot: (B, S, K*M)
        # patches_proj: (B, S, P)  -> (B, S, 1, P)
        # prot_flat: (K*M, P) -> (1, 1, K*M, P)
        # efficient matmul: (B, S, P) @ (P, K*M) -> (B, S, K*M)
        prot_t = prot_flat.t()  # (P, K*M)
        dots = patches_proj @ prot_t  # (B, S, K*M)

        # reshape to (B, K, M, S)
        dots = dots.view(B, S, K, M).permute(0, 2, 3, 1).contiguous()  # (B, K, M, S)
        # reshape S->H,W
        sim_maps = dots.view(B, K, M, H, W)  # (B,K,M,H,W)
        # max over spatial -> skm_max
        skm_max = sim_maps.view(B, K, M, -1).max(dim=-1).values  # (B,K,M)
        return sim_maps, skm_max

    def aggregate_sim_per_concept_from_local(self, v_local_proj):
        """
        v_local_proj: (B, K, P) -> projected local concept vectors (normalized)
        prototypes: (K, M, P)
        returns:
            sim_k_agg: (B, K) aggregated sim per concept defined as sum_m q_m * <p_km, v'>
            q_assign: (B, K, M) assignment probabilities for each v'
        """
        B, K, P = v_local_proj.shape
        prot = F.normalize(self.prototypes, dim=-1)  # (K, M, P)

        # compute dot product <p_km, v'_k> -> (B, K, M)
        # expand dims for broadcasting:
        v_expand = v_local_proj.unsqueeze(2)  # (B,K,1,P)
        dots = (v_expand * prot.unsqueeze(0)).sum(dim=-1)  # (B,K,M)

        # assignment q_m(v') = softmax_m( gamma * <p_km, v'> )
        q_logits = self.contrastive_gamma * dots  # (B,K,M)
        q = F.softmax(q_logits, dim=-1)  # (B,K,M)

        # sim_km contribution = q * dot
        sim_km = q * dots  # (B,K,M)
        sim_k = sim_km.sum(dim=-1)  # (B,K)
        return sim_k, q, dots

    def contrastive_loss(self, v_local):
        """
        v_local: (B, K, C) raw local vectors (before projector)
        Implement SoftTriple-like loss as Eqn (9) in paper.
        We'll:
          - project v_local -> v'_ (B,K,P)
          - compute sim_k(v') = sum_m q_m * <p_km, v'>
          - compute loss per positive concept: -log( exp(λ (sim_k + δ)) / sum_{k'} exp(λ sim_{k'}) )
        Note: v_local may contain cases where concept absent -> in training you should call this per *positive* concept instances.
        For simplicity here we compute loss across all (B,K) entries optionally using a mask provided by caller.
        """
        # In general caller should provide mask_pos: (B, K) bool where concept present.
        # We'll accept an optional keyword mask in arguments in future; for now assume all are positive.
        B, K, C = v_local.shape
        # project and normalize
        v_local_flat = v_local.view(B * K, C)  # (B*K, C)
        v_proj_flat = self.projector(v_local_flat)  # (B*K, P)
        v_proj_flat = F.normalize(v_proj_flat, dim=-1)
        v_proj = v_proj_flat.view(B, K, -1)  # (B, K, P)

        # compute sim_k for each (B,K)
        sim_k, q, dots = self.aggregate_sim_per_concept_from_local(v_proj)  # (B,K), (B,K,M), (B,K,M)

        # Now compute logits for denominator: for each (B,K) we need sim for ALL concepts k'
        # We already computed sim_k for the "positive" concept per slot. But we need sim_{k'}(v'_{k}) for k' != k.
        # To compute sim_{k'}(v'_{k}), we need v'_{k} (which is associated with concept k) vs prototypes of all concepts.
        # Let's compute dot_all: (B, K, K', M) where K' = num_concepts
        # Efficient way: v_proj: (B, K, P), prototypes: (K', M, P)
        B, K, P = v_proj.shape
        prot = F.normalize(self.prototypes, dim=-1)  # (K, M, P)

        # compute dot_all[k_query,k_proto] = <v_proj[:,k_query,:], p[k_proto,m,:]> -> result (B, K_query, K_proto, M)
        # We can use einsum
        # v_proj: (B, Kq, P), prot: (Kp, M, P) -> dots_all: (B, Kq, Kp, M)
        # Use different letters for query and proto: bip,jmp->bijm
        dots_all = torch.einsum("bip,jmp->bijm", v_proj, prot)  # (B, K, K, M)

        # compute q' assignment over M for each (B, Kq, Kp)
        # q'_m = softmax_m( gamma * dots_all )
        q_logits_all = self.contrastive_gamma * dots_all  # (B, Kq, Kp, M)
        q_all = F.softmax(q_logits_all, dim=-1)  # (B, Kq, Kp, M)

        # sim_all over prototypes per concept: sum_m q_all * dots_all -> (B, Kq, Kp)
        sim_all = (q_all * dots_all).sum(dim=-1)  # (B, Kq, Kp)

        # For each v' (slot kq) the positive concept is where kq == k_proto (diagonal)
        # positive_sim = sim_all[..., kq == kp] -> that's sim_k (we already had sim_k)
        # Now compute loss per v' (B,K): -log( exp(λ(sim_pos + δ)) / sum_k' exp(λ sim_all[...,k']) )
        lambda_ = self.contrastive_lambda
        delta = self.contrastive_delta

        # numerator:
        pos_sim = sim_all[..., torch.arange(self.num_concepts)]  # (B, K) diagonal extraction
        # pos_sim equals sim_k most of the time
        numer = torch.exp(lambda_ * (pos_sim + delta))  # (B, K)
        denom = torch.exp(lambda_ * sim_all).sum(dim=-1)  # (B, K)
        loss_matrix = -torch.log((numer + 1e-12) / (denom + 1e-12))  # (B, K)
        loss = loss_matrix.mean()
        return loss

    def forward(self, x, return_all=True):
        """
        x: (B, 3, H, W)
        returns dict with:
            - feature_map (B,C,H,W)
            - cams (B,K,H,W) raw
            - cam_weights (B,K,H,W) spatial softmax weights
            - v_local (B,K,C) local concept vectors (pre-projection)
            - v_local_proj (B,K,P) projected local (normalized)
            - patches_proj (B, S, P) normalized projected patches
            - sim_maps (B,K,M,H,W)
            - skm_max (B,K,M)
            - sim_k_agg (B,K)  aggregated sim per concept from v_local
            - class_logits (B, num_classes)
        """
        feature_map = self._get_feature_map(x)  # (B,C,H,W)
        cams = self.concept_head(feature_map)   # raw CAMs (B,K,H,W)

        # local concept vectors
        v_local = self.compute_local_concept_vectors(feature_map, cams)  # (B,K,C)

        # project local vectors
        B, K, C = v_local.shape
        v_local_flat = v_local.view(B * K, C)
        v_local_proj_flat = self.projector(v_local_flat)  # (B*K, P)
        v_local_proj_flat = F.normalize(v_local_proj_flat, dim=-1)
        v_local_proj = v_local_proj_flat.view(B, K, -1)  # (B,K,P)

        # project patches and compute similarity maps / skm_max
        patches_proj, patches_raw, H, W = self._project_patches(feature_map)  # patches_proj: (B, S, P)
        sim_maps, skm_max = self.compute_similarity_maps_and_scores(patches_proj, H, W)  # (B,K,M,H,W),(B,K,M)

        # aggregated sim per concept from v_local (contrastive aggregation)
        sim_k_agg, q_assign, dots = self.aggregate_sim_per_concept_from_local(v_local_proj)  # (B,K),(B,K,M),(B,K,M)

        # classifier uses aggregated sim per concept (K)
        class_logits = self.classifier(sim_k_agg)  # (B, num_classes)

        out = {
            "feature_map": feature_map,
            "cams": cams,
            "cam_weights": self._spatial_softmax(cams),
            "v_local": v_local,
            "v_local_proj": v_local_proj,
            "patches_proj": patches_proj,
            "sim_maps": sim_maps,
            "skm_max": skm_max,
            "sim_k_agg": sim_k_agg,
            "q_assign": q_assign,
            "class_logits": class_logits,
            "prototypes": self.prototypes,
        }
        if return_all:
            return out
        else:
            return {"class_logits": class_logits, "sim_k_agg": sim_k_agg}

# --- Quick unit test (shapes) ---
if __name__ == "__main__":
    dummy_img = torch.randn(2, 3, 448, 448)
    model = MedicalConceptModel(
        model_name="aysangh/medsiglip-448-vindr-bin",
        num_concepts=12,
        num_classes=4,
        projection_dim=128,
        prototypes_per_concept=4,
        freeze_backbone=True,
    )
    with torch.no_grad():
        out = model(dummy_img)
    print("cams:", out["cams"].shape)             # expect (B, K, H, W)
    print("v_local:", out["v_local"].shape)       # (B, K, C)
    print("v_local_proj:", out["v_local_proj"].shape)  # (B, K, P)
    print("patches_proj:", out["patches_proj"].shape)  # (B, H*W, P)
    print("sim_maps:", out["sim_maps"].shape)     # (B, K, M, H, W)
    print("skm_max:", out["skm_max"].shape)       # (B, K, M)
    print("sim_k_agg:", out["sim_k_agg"].shape)   # (B, K)
    print("class_logits:", out["class_logits"].shape)  # (B, num_classes)
    print("prototypes:", out["prototypes"].shape) # (K, M, P)
    print("Unit test shapes OK.")

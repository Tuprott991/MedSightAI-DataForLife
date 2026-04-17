import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F


class SimCAM(nn.Module):
    """
    SimCAM with forward-hook feature extraction (robust for any model).
    """

    def __init__(self, model, target_layer, fc=None):
        """
        model: full model
        target_layer: nn.Module (the exact layer you want feature map from)
        fc: optional (ONLY if it maps channels -> embedding_dim per spatial position)
        """
        super(SimCAM, self).__init__()
        self.model = model
        self.target_layer = target_layer
        self.fc = fc

    def Point_Specific(self, decom, point=[0, 0], size=(224, 224)):
        decom_padding = nn.functional.pad(
            decom.permute(2, 3, 0, 1), (1, 1, 1, 1), mode="replicate"
        ).permute(2, 3, 0, 1)

        x = (point[0] + 0.5) / size[0] * (decom_padding.shape[0] - 2)
        y = (point[1] + 0.5) / size[1] * (decom_padding.shape[1] - 2)

        x = x + 0.5
        y = y + 0.5

        x_min = int(np.floor(x))
        y_min = int(np.floor(y))
        x_max = x_min + 1
        y_max = y_min + 1

        dx = x - x_min
        dy = y - y_min

        interpolation = (
            decom_padding[x_min, y_min] * (1 - dx) * (1 - dy)
            + decom_padding[x_max, y_min] * dx * (1 - dy)
            + decom_padding[x_min, y_max] * (1 - dx) * dy
            + decom_padding[x_max, y_max] * dx * dy
        )

        return interpolation.clamp(min=0)

    def forward(self, x_q, x, point=None):
        _, _, H, W = x_q.size()

        feats = []

        def hook_fn(module, inp, out):
            feats.append(out)

        handle = self.target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            # x_q: [1,3,H,W]
            # x:   [N,3,H,W]
            x_all = torch.cat((x_q, x), dim=0)  # [N+1,3,H,W]

            _ = self.model(x_all)
            handle.remove()

            if len(feats) == 0:
                raise RuntimeError("SimCAM hook failed: no features captured.")

            fmap = feats[0]  # [B, C, h, w]
            B, C, h, w = fmap.shape
            assert B >= 2, "Need at least 1 query + 1 retrieval image"

            # Convert to [B, h*w, C]
            # each pixel location becomes a token
            tokens = fmap.permute(0, 2, 3, 1).reshape(B, h * w, C)

            # Split query vs retrievals
            q = tokens[0:1]  # [1, hw, C]
            r = tokens[1:]  # [B-1, hw, C]

            # Optional: apply fc per spatial location (rarely correct for your metric setup)
            if self.fc is not None:
                # tokens @ W^T + bias/(hw)
                W_fc = self.fc.weight.data.t()  # [inC, outC]
                b_fc = self.fc.bias.data  # [outC]
                q = q @ W_fc + b_fc / (h * w)
                r = r @ W_fc + b_fc / (h * w)

            # ---------------------------------------------------------
            # Vectorized decomposition
            #
            # q: [1, hw, C]
            # r: [B-1, hw, C]
            #
            # We want D: [B-1, hw_query, hw_retrieval]
            # D[n] = q[0] @ r[n].T
            # ---------------------------------------------------------

            # [B-1, hw, hw]
            D = torch.matmul(q.expand(r.shape[0], -1, -1), r.transpose(1, 2))

            # Normalize each retrieval pair independently (safer)
            D = D / (D.amax(dim=(1, 2), keepdim=True) + 1e-8)

            # ReLU
            D = D.clamp(min=0)

            # Reshape into spatial decomposition:
            # [B-1, h, w, h, w]
            D = D.view(r.shape[0], h, w, h, w)

            # ---------------------------------------------------------
            # Build final maps
            # decom_1: sum over retrieval positions -> query heatmap
            # decom_2: sum over query positions -> retrieval heatmap
            # ---------------------------------------------------------

            # [B-1, h, w]
            decom_1 = D.sum(dim=(3, 4))

            if point is not None:
                # point-specific decom_2
                # We need D for ONE retrieval at a time
                # because Point_Specific expects [h,w,h,w]
                decom_2_list = []
                for n in range(D.shape[0]):
                    decom_2_list.append(self.Point_Specific(D[n], point, size=(H, W)))
                decom_2 = torch.stack(decom_2_list, dim=0)  # [B-1, h, w]
            else:
                # [B-1, h, w]
                decom_2 = D.sum(dim=(1, 2))

            # ---------------------------------------------------------
            # Upsample to input size
            # Return shape: [B-1, 2, H, W]
            # ---------------------------------------------------------

            maps = torch.stack((decom_1, decom_2), dim=1)  # [B-1, 2, h, w]

            maps = nn.functional.interpolate(
                maps, size=(H, W), mode="bilinear", align_corners=False
            )

        return maps

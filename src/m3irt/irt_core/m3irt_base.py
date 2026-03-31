from typing import Dict

import numpy as np
import torch

from m3irt.irt_core._multimodal_irt_base import BaseMultimodalIRT


class M3IRT_base(BaseMultimodalIRT):
    include_difficulty_components = True

    def _combine_linear(
        self,
        theta_base: torch.Tensor,
        theta_text: torch.Tensor,
        theta_image: torch.Tensor,
        theta_synergy: torch.Tensor,
        a_base: torch.Tensor,
        a_text: torch.Tensor,
        a_image: torch.Tensor,
        a_synergy: torch.Tensor,
        b_full: torch.Tensor,
        r_text: torch.Tensor,
        r_image: torch.Tensor,
    ) -> torch.Tensor:
        return (
            a_base * theta_base
            + r_image * a_image * theta_image
            + r_text * a_text * theta_text
            + r_image * r_text * a_synergy * theta_synergy
            - b_full
        )

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            positive_params = [
                (self.theta, self.theta_max),
                (self.theta[:, 0], self.theta_base_max),
                (self.a_base_raw, self.a_scale),
                (self.a_text_raw, self.a_scale),
                (self.a_image_raw, self.a_scale),
                (self.a_synergy_raw, self.a_scale),
            ]
            for param, max_val in positive_params:
                if self.enable_abs_clamp:
                    param.data.clamp_(1e-4, max_val)
                else:
                    param.data.clamp_(min=1e-4)

            if self.enable_abs_clamp:
                self.b_base_raw.data.clamp_(self.difficulty_base_min, self.difficulty_base_max)
            else:
                self.b_base_raw.data.clamp_(min=0)

            for param in [self.b_text_raw, self.b_image_raw, self.b_synergy_raw]:
                if self.enable_abs_clamp:
                    param.data.clamp_(-self.difficulty_other_max, -self.difficulty_other_min)
                else:
                    param.data.clamp_(max=0)

    def _after_theta_update(self) -> None:
        self.clamp_parameters()

    def compute_item_fisher_information(self, model_name: str) -> Dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            fisher_info_dict = {}
            try:
                model_idx = self.student_names.index(model_name)
            except ValueError as exc:
                raise ValueError(f"Model '{model_name}' is not found in the model.") from exc

            theta_model = self.theta.detach()[model_idx]

            for item_idx, item_name in enumerate(self.test_names):
                r_text = self.mask[item_idx, 0].item()
                r_image = self.mask[item_idx, 1].item()
                param_idx = self.param_indices[item_idx].item()

                a_base = self.a_base_raw[param_idx].item()
                a_text = self.a_text_raw[param_idx].item()
                a_image = self.a_image_raw[param_idx].item()
                a_synergy = self.a_synergy_raw[param_idx].item()
                b_base = self.b_base_raw[param_idx].item()
                b_text = self.b_text_raw[param_idx].item()
                b_image = self.b_image_raw[param_idx].item()
                b_synergy = self.b_synergy_raw[param_idx].item()
                b_full = b_base + r_text * b_text + r_image * b_image + r_text * r_image * b_synergy

                g = torch.tensor(
                    [a_base, r_text * a_text, r_image * a_image, r_text * r_image * a_synergy],
                    device=self.device,
                )
                linear = (
                    a_base * theta_model[0]
                    + r_image * a_image * theta_model[2]
                    + r_text * a_text * theta_model[1]
                    + r_text * r_image * a_synergy * theta_model[3]
                    - b_full
                )
                probs = self._sigmoid(linear)
                coef = probs * (1 - probs)
                fisher_info_dict[item_name] = (coef * torch.outer(g, g)).cpu().numpy()
        self.train()
        return fisher_info_dict

from typing import Dict, List

import numpy as np
import torch

from m3irt.irt_core._multimodal_irt_base import BaseMultimodalIRT


class M2IRT_base(BaseMultimodalIRT):
    def __init__(
        self,
        response_data: np.ndarray,
        train_mask: np.ndarray,
        student_names: List[str],
        test_names: List[str],
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 1000,
        device: str = None,
        eps: float = 1e-3,
        theta_init: float = 0.5,
        theta_base_max: float = 0.1,
        theta_max: float = 1.0,
        difficulty_base_init: float = 0.0,
        difficulty_base_max: float = 3.0,
        difficulty_base_min: float = 0.0,
        difficulty_other_init: float = 0.0,
        difficulty_other_max: float = 1.5,
        difficulty_other_min: float = 0.0,
        a_init: float = 0.2,
        a_scale: float = 3.0,
        enable_abs_clamp: bool = True,
    ):
        super().__init__(
            response_data=response_data,
            train_mask=train_mask,
            student_names=student_names,
            test_names=test_names,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            device=device,
            eps=eps,
            theta_init=theta_init,
            theta_base_max=theta_base_max,
            theta_max=theta_max,
            difficulty_base_init=difficulty_base_init,
            difficulty_base_max=difficulty_base_max,
            difficulty_base_min=difficulty_base_min,
            difficulty_other_init=difficulty_other_init,
            difficulty_other_max=difficulty_other_max,
            difficulty_other_min=difficulty_other_min,
            a_init=a_init,
            a_scale=a_scale,
            enable_abs_clamp=enable_abs_clamp,
        )

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
        a_full = a_base + r_text * a_text + r_image * a_image + r_text * r_image * a_synergy
        theta_full = theta_base + r_text * theta_text + r_image * theta_image + r_text * r_image * theta_synergy
        return a_full * theta_full - b_full

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            self.theta.data.clamp_(0, self.theta_max)
            self.theta.data[:, 0].clamp_(0, self.theta_base_max)
            for param in [self.a_base_raw, self.a_text_raw, self.a_image_raw, self.a_synergy_raw]:
                param.data.clamp_(1e-4, self.a_scale)
            self.b_base_raw.data.clamp_(self.difficulty_base_min, self.difficulty_base_max)
            for param in [self.b_text_raw, self.b_image_raw, self.b_synergy_raw]:
                param.data.clamp_(-self.difficulty_other_max, 0.0)

    def _after_theta_update(self) -> None:
        self.clamp_parameters()

    def compute_item_fisher_information(self, model_name: str) -> Dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            fisher = {}
            try:
                model_idx = self.student_names.index(model_name)
            except ValueError as exc:
                raise ValueError(f"Model '{model_name}' is not found in the model.") from exc

            theta_model = self.theta.detach()[model_idx]
            for item_idx, name in enumerate(self.test_names):
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

                a_full = a_base + r_text * a_text + r_image * a_image + r_text * r_image * a_synergy
                theta_full = (
                    theta_model[0]
                    + r_text * theta_model[1]
                    + r_image * theta_model[2]
                    + r_text * r_image * theta_model[3]
                )
                b_full = b_base + r_text * b_text + r_image * b_image + r_text * r_image * b_synergy

                probs = self._sigmoid(a_full * theta_full - b_full)
                fisher[name] = ((a_full**2) * probs * (1 - probs)).cpu().numpy()
        self.train()
        return fisher

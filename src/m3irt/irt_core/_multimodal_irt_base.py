import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class ResponseDataset(Dataset):
    def __init__(self, response_data: np.ndarray, train_mask: np.ndarray):
        self.data = torch.tensor(response_data, dtype=torch.float32)
        self.valid_data = self.data != -1
        self.set_train_mask(train_mask)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.train_indices[idx], idx

    def set_train_mask(self, train_mask: np.ndarray) -> None:
        self.train_mask = torch.tensor(train_mask, dtype=torch.float32)
        self.train_indices = self.train_mask == 1
        self.test_indices = self.train_mask == 0
        self.val_indices = self.train_mask == 2


class BaseMultimodalIRT(nn.Module, ABC):
    include_difficulty_components = False

    def __init__(
        self,
        response_data: np.ndarray,
        train_mask: np.ndarray,
        student_names: List[str],
        test_names: List[str],
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 1000,
        device: Optional[str] = None,
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
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.enable_abs_clamp = enable_abs_clamp
        self.eps = eps

        self.data = response_data
        self.train_mask = train_mask
        self.num_students = response_data.shape[0]
        self.num_items = response_data.shape[1]
        self.student_names = student_names
        self.test_names = test_names
        self.a_scale = a_scale
        self.theta_max = theta_max
        self.theta_base_max = theta_base_max
        self.difficulty_base_max = difficulty_base_max
        self.difficulty_base_min = difficulty_base_min
        self.difficulty_other_max = difficulty_other_max
        self.difficulty_other_min = difficulty_other_min

        self.problem_base_names = []
        self.problem_to_base = {}
        for name in test_names:
            base_name = re.sub(r"^(?:<no_question>|<no_image>|<no_info>)", "", name)
            if base_name not in self.problem_base_names:
                self.problem_base_names.append(base_name)
            self.problem_to_base[name] = base_name
        self.num_unique_problems = len(self.problem_base_names)

        self.a_base_raw = nn.Parameter(torch.full((self.num_unique_problems,), a_init, device=self.device))
        self.a_text_raw = nn.Parameter(torch.full((self.num_unique_problems,), a_init, device=self.device))
        self.a_image_raw = nn.Parameter(torch.full((self.num_unique_problems,), a_init, device=self.device))
        self.a_synergy_raw = nn.Parameter(torch.full((self.num_unique_problems,), a_init, device=self.device))

        self.b_base_raw = nn.Parameter(
            torch.full((self.num_unique_problems,), difficulty_base_init, device=self.device)
        )
        self.b_text_raw = nn.Parameter(
            torch.full((self.num_unique_problems,), difficulty_other_init, device=self.device)
        )
        self.b_image_raw = nn.Parameter(
            torch.full((self.num_unique_problems,), difficulty_other_init, device=self.device)
        )
        self.b_synergy_raw = nn.Parameter(
            torch.full((self.num_unique_problems,), difficulty_other_init, device=self.device)
        )

        self.param_indices = torch.tensor(
            [self.problem_base_names.index(self.problem_to_base[name]) for name in test_names],
            device=self.device,
            dtype=torch.long,
        )
        self.theta = nn.Parameter(torch.full((self.num_students, 4), theta_init, device=self.device))

        tags = []
        for name in test_names:
            if "<no_info>" in name:
                tags.append([0.0, 0.0])
            elif "<no_image>" in name:
                tags.append([1.0, 0.0])
            elif "<no_question>" in name:
                tags.append([0.0, 1.0])
            else:
                tags.append([1.0, 1.0])
        self.register_buffer("mask", torch.tensor(tags, dtype=torch.float32))

        self.to(self.device)
        self.dataset = ResponseDataset(response_data, train_mask)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.max_epochs = max_epochs
        self.loss_history = []

    def set_train_mask(self, train_mask: np.ndarray) -> None:
        self.train_mask = train_mask
        self.dataset.set_train_mask(train_mask)

    def get_parameters(self):
        return (
            self.a_base_raw[self.param_indices],
            self.a_text_raw[self.param_indices],
            self.a_image_raw[self.param_indices],
            self.a_synergy_raw[self.param_indices],
            self.b_base_raw[self.param_indices],
            self.b_text_raw[self.param_indices],
            self.b_image_raw[self.param_indices],
            self.b_synergy_raw[self.param_indices],
        )

    def compute_b_full(self, b_base, b_text, b_image, b_synergy):
        r_text = self.mask[:, 0]
        r_image = self.mask[:, 1]
        return self._combine_b_full(b_base, b_text, b_image, b_synergy, r_text, r_image)

    def _combine_b_full(self, b_base, b_text, b_image, b_synergy, r_text, r_image):
        return b_base + r_text * b_text + r_image * b_image + r_text * r_image * b_synergy

    def _select_parameters(self, item_indices: Optional[torch.Tensor]):
        params = self.get_parameters()
        if item_indices is None:
            return params
        return tuple(param[item_indices] for param in params)

    def _matrix_masks(self, item_indices: Optional[torch.Tensor]):
        masks = self.mask if item_indices is None else self.mask[item_indices]
        return masks[:, 0].unsqueeze(0), masks[:, 1].unsqueeze(0)

    def _pair_masks(self, item_indices: torch.Tensor):
        masks = self.mask[item_indices]
        return masks[:, 0], masks[:, 1]

    def _sigmoid(self, linear: torch.Tensor) -> torch.Tensor:
        linear = torch.clamp(linear, -15, 15)
        probs = 1 / (1 + torch.exp(-linear))
        return torch.clamp(probs, self.eps, 1 - self.eps)

    def _masked_nll(
        self,
        responses: torch.Tensor,
        probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = response_mask.bool()
        rr = responses[valid]
        rp = probs[valid]
        if rr.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        return -(rr * torch.log(rp) + (1 - rr) * torch.log(1 - rp)).mean()

    @abstractmethod
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
        raise NotImplementedError

    def _matrix_probabilities(
        self,
        student_idx: torch.Tensor,
        item_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        theta_components = self.theta[student_idx]
        theta_base = theta_components[:, 0].unsqueeze(1)
        theta_text = theta_components[:, 1].unsqueeze(1)
        theta_image = theta_components[:, 2].unsqueeze(1)
        theta_synergy = theta_components[:, 3].unsqueeze(1)

        a_base, a_text, a_image, a_synergy, b_base, b_text, b_image, b_synergy = self._select_parameters(item_indices)
        r_text, r_image = self._matrix_masks(item_indices)
        b_full = self._combine_b_full(
            b_base,
            b_text,
            b_image,
            b_synergy,
            r_text.squeeze(0),
            r_image.squeeze(0),
        ).unsqueeze(0)
        linear = self._combine_linear(
            theta_base,
            theta_text,
            theta_image,
            theta_synergy,
            a_base.unsqueeze(0),
            a_text.unsqueeze(0),
            a_image.unsqueeze(0),
            a_synergy.unsqueeze(0),
            b_full,
            r_text,
            r_image,
        )
        return self._sigmoid(linear)

    def _pair_probabilities(
        self,
        student_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        theta_components = self.theta[student_indices]
        theta_base = theta_components[:, 0]
        theta_text = theta_components[:, 1]
        theta_image = theta_components[:, 2]
        theta_synergy = theta_components[:, 3]

        a_base, a_text, a_image, a_synergy, b_base, b_text, b_image, b_synergy = self._select_parameters(item_indices)
        r_text, r_image = self._pair_masks(item_indices)
        b_full = self._combine_b_full(b_base, b_text, b_image, b_synergy, r_text, r_image)
        linear = self._combine_linear(
            theta_base,
            theta_text,
            theta_image,
            theta_synergy,
            a_base,
            a_text,
            a_image,
            a_synergy,
            b_full,
            r_text,
            r_image,
        )
        return self._sigmoid(linear)

    def forward(self, responses, response_mask, student_idx):
        probs = self._matrix_probabilities(student_idx)
        return self._masked_nll(responses, probs, response_mask)

    @abstractmethod
    def clamp_parameters(self) -> None:
        raise NotImplementedError

    def _after_theta_update(self) -> None:
        pass

    def fit(self):
        best_loss = float("inf")
        best_epoch = 0
        patience = 100
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            batch_count = 0
            for responses, response_mask, student_idx in self.dataloader:
                responses = responses.to(self.device)
                response_mask = response_mask.to(self.device)
                student_idx = student_idx.to(self.device)

                self.optimizer.zero_grad()
                loss = self.forward(responses, response_mask, student_idx)
                if loss.item() <= 0:
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                self.clamp_parameters()

                epoch_loss += loss.item()
                batch_count += 1

            if batch_count == 0:
                continue

            avg_loss = epoch_loss / batch_count
            self.loss_history.append(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
            if epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch + 1}, best loss {best_loss:.4f}")
                break
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs}  Loss={avg_loss:.4f}")
        return self.get_estimates()

    def get_estimates(self) -> Dict:
        theta_components = self.theta.detach().cpu().numpy()
        theta_dict = {
            self.student_names[i]: {
                "theta_base": theta_components[i, 0],
                "theta_text": theta_components[i, 1],
                "theta_image": theta_components[i, 2],
                "theta_synergy": theta_components[i, 3],
            }
            for i in range(self.num_students)
        }

        params = self.get_parameters()
        a_base, a_text, a_image, a_synergy, b_base, b_text, b_image, b_synergy = [
            param.detach().cpu().numpy() for param in params
        ]
        b_full = self.compute_b_full(
            torch.tensor(b_base),
            torch.tensor(b_text),
            torch.tensor(b_image),
            torch.tensor(b_synergy),
        ).numpy()

        result = {
            "theta": theta_dict,
            "discrimination_base": dict(zip(self.test_names, a_base)),
            "discrimination_text": dict(zip(self.test_names, a_text)),
            "discrimination_image": dict(zip(self.test_names, a_image)),
            "discrimination_synergy": dict(zip(self.test_names, a_synergy)),
            "difficulty_full": dict(zip(self.test_names, b_full)),
        }
        if self.include_difficulty_components:
            result.update(
                {
                    "difficulty_base": dict(zip(self.test_names, b_base)),
                    "difficulty_text": dict(zip(self.test_names, b_text)),
                    "difficulty_image": dict(zip(self.test_names, b_image)),
                    "difficulty_synergy": dict(zip(self.test_names, b_synergy)),
                }
            )
        return result

    def _split_pair_indices(self, split: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        split_indices = getattr(self.dataset, f"{split}_indices")
        valid_pairs = (split_indices & self.dataset.valid_data).nonzero(as_tuple=False)
        if valid_pairs.numel() == 0:
            return None, None
        return valid_pairs[:, 0].to(self.device), valid_pairs[:, 1].to(self.device)

    def _evaluate_split(self, split: str) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            student_indices, item_indices = self._split_pair_indices(split)
            if student_indices is None:
                return {}

            probs = self._pair_probabilities(student_indices, item_indices)
            true_responses = torch.tensor(
                self.data[student_indices.cpu(), item_indices.cpu()],
                device=self.device,
            )
            pred_binary = (probs > 0.5).float()

            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            probs_np = probs.cpu().numpy()
            true_np = true_responses.cpu().numpy()
            pred_np = pred_binary.cpu().numpy()

            metrics = {
                "roc_auc": roc_auc_score(true_np, probs_np),
                "accuracy": accuracy_score(true_np, pred_np),
                "precision": precision_score(true_np, pred_np),
                "recall": recall_score(true_np, pred_np),
                "f1": f1_score(true_np, pred_np),
            }
            nll = -(true_responses * torch.log(probs) + (1 - true_responses) * torch.log(1 - probs)).mean()
            metrics["perplexity"] = torch.exp(nll).item()

            item_names = [self.test_names[idx] for idx in item_indices.cpu().tolist()]
            shuffle_mask = np.array([("shuffle" in name) or ("from" in name) for name in item_names])
            normal_mask = ~shuffle_mask

            if split == "test":
                print(f"Number of shuffle items: {int(shuffle_mask.sum())}")
                print(f"Number of normal items: {int(normal_mask.sum())}")

            metrics["roc_auc_shuffle"] = (
                roc_auc_score(true_np[shuffle_mask], probs_np[shuffle_mask]) if shuffle_mask.any() else 0.0
            )
            metrics["roc_auc_normal"] = (
                roc_auc_score(true_np[normal_mask], probs_np[normal_mask]) if normal_mask.any() else 0.0
            )

        self.train()
        return metrics

    def evaluate_predictions(self) -> Dict[str, float]:
        return self._evaluate_split("test")

    def evaluate_validation(self) -> Dict[str, float]:
        return self._evaluate_split("val")

    def predict(self) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            student_idx = torch.arange(self.num_students, device=self.device)
            probs = self._matrix_probabilities(student_idx)
        self.train()
        return probs.cpu().numpy()

    def update_single_theta(
        self,
        model_name: str,
        problem_names: List[str],
        lr: float = 1e-3,
        max_epochs: int = 1000,
        patience: int = 50,
    ) -> Dict:
        try:
            target_student_idx = self.student_names.index(model_name)
        except ValueError:
            print(f"Model '{model_name}' is not found in the model.")
            return self.get_estimates()

        selected_indices = [
            idx for idx, name in enumerate(self.test_names) if self.problem_to_base[name] in problem_names
        ]
        if not selected_indices:
            print("Input problem names are not found in the model.")
            return self.get_estimates()

        selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=self.device)
        optimizer = optim.Adam([self.theta], lr=lr)

        best_loss = float("inf")
        epochs_no_improve = 0

        self.train()
        for _epoch in range(max_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for responses, response_mask, student_idx in self.dataloader:
                responses = responses.to(self.device)
                response_mask = response_mask.to(self.device)
                student_idx = student_idx.to(self.device)

                target_rows = student_idx == target_student_idx
                if not torch.any(target_rows):
                    continue

                target_student_idx_batch = student_idx[target_rows]
                probs = self._matrix_probabilities(target_student_idx_batch, selected_tensor)
                loss = self._masked_nll(
                    responses[target_rows][:, selected_tensor],
                    probs,
                    response_mask[target_rows][:, selected_tensor],
                )
                if not torch.isfinite(loss) or loss.item() <= 0:
                    continue

                optimizer.zero_grad()
                loss.backward()
                if self.theta.grad is not None:
                    grad_mask = torch.zeros_like(self.theta.grad)
                    grad_mask[target_student_idx] = 1
                    self.theta.grad.mul_(grad_mask)
                torch.nn.utils.clip_grad_norm_([self.theta], 1.0)
                optimizer.step()
                self._after_theta_update()

                epoch_loss += loss.item()
                batch_count += 1

            if batch_count == 0:
                continue

            avg_loss = epoch_loss / batch_count
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        self.train()
        return self.get_estimates()

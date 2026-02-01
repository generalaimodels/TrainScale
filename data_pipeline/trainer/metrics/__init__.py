# ════════════════════════════════════════════════════════════════════════════════
# SOTA Trainer - Expanded Metrics Framework
# ════════════════════════════════════════════════════════════════════════════════
# Above-SOTA metrics with full classification, NLP, and regression metrics.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import abc
import math
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor


class Metric(abc.ABC):
    """Abstract base class for metrics with update/compute/reset pattern."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    @abc.abstractmethod
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        pass
    
    @abc.abstractmethod
    def compute(self) -> float:
        pass
    
    @abc.abstractmethod
    def reset(self) -> None:
        pass
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> float:
        self.reset()
        self.update(predictions, targets)
        return self.compute()


# ═════════════════════════════════════════════════════════════════════════════════
# Classification Metrics
# ═════════════════════════════════════════════════════════════════════════════════

class Accuracy(Metric):
    """Classification accuracy."""
    
    def __init__(self, name: str = "accuracy", top_k: int = 1):
        self.top_k = top_k
        super().__init__(name)
    
    def reset(self) -> None:
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        if predictions.dim() > 1:
            if self.top_k == 1:
                predictions = predictions.argmax(dim=-1)
            else:
                _, top_k_preds = predictions.topk(self.top_k, dim=-1)
                correct = top_k_preds.eq(targets.unsqueeze(-1)).any(dim=-1)
                self.correct += correct.sum().item()
                self.total += targets.numel()
                return
        self.correct += (predictions == targets).sum().item()
        self.total += targets.numel()
    
    def compute(self) -> float:
        return self.correct / max(1, self.total)


class Precision(Metric):
    """Precision metric (TP / (TP + FP))."""
    
    def __init__(self, name: str = "precision", average: str = "macro", num_classes: Optional[int] = None):
        self.average = average
        self.num_classes = num_classes
        super().__init__(name)
    
    def reset(self) -> None:
        self.tp: Dict[int, int] = defaultdict(int)
        self.fp: Dict[int, int] = defaultdict(int)
        self.support: Dict[int, int] = defaultdict(int)
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)
        preds = predictions.cpu().numpy().flatten()
        targs = targets.cpu().numpy().flatten()
        for p, t in zip(preds, targs):
            self.support[int(t)] += 1
            if p == t:
                self.tp[int(p)] += 1
            else:
                self.fp[int(p)] += 1
    
    def compute(self) -> float:
        classes = set(self.tp) | set(self.fp) | set(self.support)
        if self.average == "micro":
            total_tp = sum(self.tp.values())
            total_fp = sum(self.fp.values())
            return total_tp / max(1, total_tp + total_fp)
        precisions = []
        weights = []
        for c in classes:
            tp, fp = self.tp[c], self.fp[c]
            if tp + fp > 0:
                precisions.append(tp / (tp + fp))
                weights.append(self.support[c])
        if not precisions:
            return 0.0
        if self.average == "weighted":
            total = sum(weights)
            return sum(p * w for p, w in zip(precisions, weights)) / max(1, total)
        return sum(precisions) / len(precisions)


class Recall(Metric):
    """Recall metric (TP / (TP + FN))."""
    
    def __init__(self, name: str = "recall", average: str = "macro"):
        self.average = average
        super().__init__(name)
    
    def reset(self) -> None:
        self.tp: Dict[int, int] = defaultdict(int)
        self.fn: Dict[int, int] = defaultdict(int)
        self.support: Dict[int, int] = defaultdict(int)
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)
        preds = predictions.cpu().numpy().flatten()
        targs = targets.cpu().numpy().flatten()
        for p, t in zip(preds, targs):
            self.support[int(t)] += 1
            if p == t:
                self.tp[int(t)] += 1
            else:
                self.fn[int(t)] += 1
    
    def compute(self) -> float:
        classes = set(self.tp) | set(self.fn) | set(self.support)
        if self.average == "micro":
            total_tp = sum(self.tp.values())
            total_fn = sum(self.fn.values())
            return total_tp / max(1, total_tp + total_fn)
        recalls = []
        weights = []
        for c in classes:
            tp, fn = self.tp[c], self.fn[c]
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))
                weights.append(self.support[c])
        if not recalls:
            return 0.0
        if self.average == "weighted":
            total = sum(weights)
            return sum(r * w for r, w in zip(recalls, weights)) / max(1, total)
        return sum(recalls) / len(recalls)


class F1Score(Metric):
    """F1 score (harmonic mean of precision and recall)."""
    
    def __init__(self, name: str = "f1", average: str = "macro"):
        self.average = average
        self.precision = Precision(name="p", average=average)
        self.recall = Recall(name="r", average=average)
        super().__init__(name)
    
    def reset(self) -> None:
        self.precision.reset()
        self.recall.reset()
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
    
    def compute(self) -> float:
        p, r = self.precision.compute(), self.recall.compute()
        return 2 * p * r / max(1e-8, p + r) if (p + r) > 0 else 0.0


class AUROC(Metric):
    """Area Under ROC Curve (binary classification)."""
    
    def __init__(self, name: str = "auroc"):
        super().__init__(name)
    
    def reset(self) -> None:
        self.scores: List[float] = []
        self.labels: List[int] = []
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        if predictions.dim() > 1:
            predictions = predictions[:, 1]  # Positive class probability
        self.scores.extend(predictions.cpu().tolist())
        self.labels.extend(targets.cpu().tolist())
    
    def compute(self) -> float:
        if not self.scores:
            return 0.0
        # Sort by score descending
        pairs = sorted(zip(self.scores, self.labels), reverse=True)
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.0
        # Compute AUC via trapezoidal rule
        tp, fp, auc = 0, 0, 0.0
        prev_score = float("inf")
        for score, label in pairs:
            if score != prev_score:
                auc += tp * fp
                prev_score = score
            if label == 1:
                tp += 1
            else:
                fp += 1
        auc += tp * fp
        return auc / (n_pos * n_neg)


# ═════════════════════════════════════════════════════════════════════════════════
# Language Modeling Metrics
# ═════════════════════════════════════════════════════════════════════════════════

class Perplexity(Metric):
    """Perplexity for language models."""
    
    def __init__(self, name: str = "perplexity"):
        super().__init__(name)
    
    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        if predictions.dim() == 3:
            predictions = predictions.view(-1, predictions.size(-1))
            targets = targets.view(-1)
        loss = torch.nn.functional.cross_entropy(
            predictions, targets, reduction="sum", ignore_index=-100
        )
        self.total_loss += loss.item()
        self.total_tokens += (targets != -100).sum().item()
    
    def compute(self) -> float:
        if self.total_tokens == 0:
            return float("inf")
        return math.exp(min(self.total_loss / self.total_tokens, 100))


class BLEUScore(Metric):
    """BLEU score for machine translation."""
    
    def __init__(self, name: str = "bleu", max_n: int = 4, smooth: bool = True):
        self.max_n = max_n
        self.smooth = smooth
        super().__init__(name)
    
    def reset(self) -> None:
        self.matches = [0] * self.max_n
        self.totals = [0] * self.max_n
        self.ref_len = 0
        self.hyp_len = 0
    
    def update(self, predictions: Union[Tensor, List], targets: Union[Tensor, List]) -> None:
        if isinstance(predictions, Tensor):
            predictions = predictions.tolist()
        if isinstance(targets, Tensor):
            targets = targets.tolist()
        for hyp, ref in zip(predictions, targets):
            self._update_single(hyp, ref)
    
    def _update_single(self, hyp: List, ref: List) -> None:
        self.hyp_len += len(hyp)
        self.ref_len += len(ref)
        for n in range(1, self.max_n + 1):
            hyp_ngrams = self._get_ngrams(hyp, n)
            ref_ngrams = self._get_ngrams(ref, n)
            matches = sum(min(hyp_ngrams.get(ng, 0), ref_ngrams.get(ng, 0)) for ng in hyp_ngrams)
            self.matches[n - 1] += matches
            self.totals[n - 1] += max(1, len(hyp) - n + 1)
    
    def _get_ngrams(self, tokens: List, n: int) -> Dict:
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i:i + n])] += 1
        return ngrams
    
    def compute(self) -> float:
        precisions = []
        for n in range(self.max_n):
            if self.totals[n] == 0:
                precisions.append(0.0)
            elif self.matches[n] == 0:
                precisions.append(1.0 / (self.totals[n] + 1) if self.smooth else 0.0)
            else:
                precisions.append(self.matches[n] / self.totals[n])
        if any(p == 0 for p in precisions):
            return 0.0
        log_prec = sum(math.log(p) for p in precisions)
        bp = 1.0 if self.hyp_len >= self.ref_len else math.exp(1 - self.ref_len / max(1, self.hyp_len))
        return bp * math.exp(log_prec / self.max_n)


class ROUGEScore(Metric):
    """ROUGE-L score for summarization."""
    
    def __init__(self, name: str = "rouge_l"):
        super().__init__(name)
    
    def reset(self) -> None:
        self.scores: List[float] = []
    
    def update(self, predictions: List[List], targets: List[List]) -> None:
        for hyp, ref in zip(predictions, targets):
            lcs_len = self._lcs_length(hyp, ref)
            p = lcs_len / max(1, len(hyp))
            r = lcs_len / max(1, len(ref))
            f1 = 2 * p * r / max(1e-8, p + r) if (p + r) > 0 else 0.0
            self.scores.append(f1)
    
    def _lcs_length(self, x: List, y: List) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
    
    def compute(self) -> float:
        return sum(self.scores) / max(1, len(self.scores))


# ═════════════════════════════════════════════════════════════════════════════════
# Regression Metrics
# ═════════════════════════════════════════════════════════════════════════════════

class MeanSquaredError(Metric):
    """Mean Squared Error."""
    
    def __init__(self, name: str = "mse"):
        super().__init__(name)
    
    def reset(self) -> None:
        self.sum_squared_error = 0.0
        self.count = 0
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        self.sum_squared_error += ((predictions - targets) ** 2).sum().item()
        self.count += targets.numel()
    
    def compute(self) -> float:
        return self.sum_squared_error / max(1, self.count)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error."""
    
    def __init__(self, name: str = "mae"):
        super().__init__(name)
    
    def reset(self) -> None:
        self.sum_abs_error = 0.0
        self.count = 0
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        self.sum_abs_error += (predictions - targets).abs().sum().item()
        self.count += targets.numel()
    
    def compute(self) -> float:
        return self.sum_abs_error / max(1, self.count)


class R2Score(Metric):
    """R² coefficient of determination."""
    
    def __init__(self, name: str = "r2"):
        super().__init__(name)
    
    def reset(self) -> None:
        self.ss_res = 0.0
        self.ss_tot = 0.0
        self.y_sum = 0.0
        self.y_sq_sum = 0.0
        self.count = 0
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        self.ss_res += ((targets - predictions) ** 2).sum().item()
        self.y_sum += targets.sum().item()
        self.y_sq_sum += (targets ** 2).sum().item()
        self.count += targets.numel()
    
    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        y_mean = self.y_sum / self.count
        self.ss_tot = self.y_sq_sum - self.count * y_mean ** 2
        if self.ss_tot == 0:
            return 0.0
        return 1 - self.ss_res / self.ss_tot


# ═════════════════════════════════════════════════════════════════════════════════
# Metric Collection
# ═════════════════════════════════════════════════════════════════════════════════

class MetricCollection:
    """Collection of metrics to compute together."""
    
    def __init__(self, metrics: Sequence[Metric]):
        self.metrics = {m.name: m for m in metrics}
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        for m in self.metrics.values():
            m.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        return {n: m.compute() for n, m in self.metrics.items()}
    
    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()
    
    def add(self, metric: Metric) -> None:
        self.metrics[metric.name] = metric


# Functional API
def compute_accuracy(preds: Tensor, targets: Tensor) -> float:
    return Accuracy()(preds, targets)

def compute_f1(preds: Tensor, targets: Tensor, average: str = "macro") -> float:
    return F1Score(average=average)(preds, targets)

def compute_perplexity(preds: Tensor, targets: Tensor) -> float:
    return Perplexity()(preds, targets)


__all__ = [
    "Metric", "Accuracy", "Precision", "Recall", "F1Score", "AUROC",
    "Perplexity", "BLEUScore", "ROUGEScore",
    "MeanSquaredError", "MeanAbsoluteError", "R2Score",
    "MetricCollection", "compute_accuracy", "compute_f1", "compute_perplexity",
]

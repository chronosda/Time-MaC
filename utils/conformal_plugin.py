import numpy as np


def _bisection(f, lo, hi, tol=1e-6, max_iter=64):
    flo, fhi = f(lo), f(hi)
    if flo < 0 and fhi < 0:
        return hi
    if flo > 0 and fhi > 0:
        return lo
    a, fa = lo, flo
    b, fb = hi, fhi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        if (fa > 0 and fm > 0) or (fa < 0 and fm < 0):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5 * (a + b)


def _hoeffding_ucb(mean_loss, n, delta):
    return mean_loss + np.sqrt(np.log(1.0 / delta) / (2.0 * n))


def _resolve_threshold_upper(scores: np.ndarray, lam_hi: float | None = None) -> float:
    if lam_hi is not None and lam_hi > 0:
        return float(lam_hi)
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 1.0
    hi = float(np.quantile(scores, 0.999) * 10.0)
    return max(hi, 1.0)


def solve_bq_threshold(
    scores: np.ndarray,
    alpha: float,
    method: str = "hpd",
    max_risk: float = 1.0,
    lam_hi: float | None = None,
    hpd_level: float = 0.95,
    num_dir: int = 1000,
    delta: float = 0.05,
    rng: np.random.Generator | None = None,
) -> float:
    score_vec = np.asarray(scores, dtype=np.float64).reshape(-1)
    score_vec = score_vec[np.isfinite(score_vec)]
    if score_vec.size == 0:
        return 1.0

    n = score_vec.shape[0]
    lam_hi = _resolve_threshold_upper(score_vec, lam_hi)

    if method == "crc":

        def f(lam):
            loss = (score_vec > lam).astype(np.float64)
            risk = float(np.mean(loss))
            return (n / (n + 1.0)) * risk + max_risk / (n + 1.0) - alpha

    elif method == "hpd":
        rng = np.random.default_rng(0) if rng is None else rng
        weights = rng.dirichlet(np.ones(n + 1), size=int(num_dir))

        def f(lam):
            loss = (score_vec > lam).astype(np.float64)
            l_plus = weights[:, 1:] @ loss + weights[:, 0] * max_risk
            return float(np.quantile(l_plus, hpd_level)) - alpha

    elif method == "rcps":

        def f(lam):
            loss = (score_vec > lam).astype(np.float64)
            ucb = _hoeffding_ucb(float(np.mean(loss)), n, delta)
            return ucb - alpha

    else:
        raise ValueError(f"unknown threshold method {method}")

    return float(_bisection(f, 0.0, lam_hi))


def estimate_scale_from_residuals(residuals: np.ndarray, method: str = "mad") -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    if method == "mad":
        med = np.median(residuals, axis=0, keepdims=True)
        scale = np.median(np.abs(residuals - med), axis=0) / 0.6745
    elif method == "std":
        scale = residuals.std(axis=0)
    elif method == "global_mad":
        med = np.median(residuals)
        scale = np.median(np.abs(residuals - med)) / 0.6745
        scale = np.full_like(residuals[0:1], float(scale))
    else:
        raise ValueError(f"unknown scale method {method}")
    return np.maximum(np.asarray(scale, dtype=np.float64), 1e-6)


def infer_horizon_axis(shape: tuple[int, ...], pred_len: int | None = None) -> int:
    if len(shape) < 2:
        raise ValueError("Residual tensors must include a batch axis and at least one forecast axis.")
    if pred_len is not None:
        candidates = [axis for axis in range(1, len(shape)) if shape[axis] == pred_len]
        if len(candidates) == 1:
            return candidates[0]
    return len(shape) - 1


class ConformalCalibrator:
    """Static post-hoc conformal calibration for point forecasts."""

    def __init__(
        self,
        method: str = "crc",
        alpha: float = 0.1,
        max_risk: float = 1.0,
        max_threshold: float | None = None,
        hpd_level: float = 0.95,
        num_dir: int = 1000,
        delta: float = 0.05,
        rng_seed: int = 0,
    ) -> None:
        self.method = method
        self.alpha = float(alpha)
        self.max_risk = float(max_risk)
        self.hpd_level = float(hpd_level)
        self.num_dir = int(num_dir)
        self.delta = float(delta)
        self.max_threshold = max_threshold
        self.rng = np.random.default_rng(rng_seed)

        self.lam_hat_: float | None = None
        self.scale_shape_: tuple | None = None

    @staticmethod
    def _flatten_with_broadcast(residuals: np.ndarray, scale: np.ndarray):
        r = np.asarray(residuals, dtype=np.float64)
        s = np.asarray(scale, dtype=np.float64)
        s = np.maximum(s, 1e-12)
        if s.shape != r.shape:
            s = np.broadcast_to(s, r.shape)
        return r.reshape(-1), s.reshape(-1)

    def _empirical_miscoverage(self, lam: float, r_flat: np.ndarray, s_flat: np.ndarray) -> float:
        return float(np.mean((r_flat > lam * s_flat).astype(np.float64)))

    def fit(self, residuals: np.ndarray, scale: np.ndarray, lam_hi: float | None = None):
        r_flat, s_flat = self._flatten_with_broadcast(residuals, scale)
        n = r_flat.shape[0]
        if lam_hi is None:
            ratio = r_flat / s_flat
            ratio = ratio[np.isfinite(ratio)]
            if ratio.size == 0:
                lam_hi = 1.0
            else:
                lam_hi = float(np.quantile(ratio, 0.999) * 10.0)
                lam_hi = max(lam_hi, 1.0)
        self.max_threshold = lam_hi

        if self.method == "crc":

            def f(lam):
                risk = self._empirical_miscoverage(lam, r_flat, s_flat)
                return (n / (n + 1.0)) * risk + self.max_risk / (n + 1.0) - self.alpha

        elif self.method == "hpd":
            w = self.rng.dirichlet(np.ones(n + 1), size=self.num_dir)

            def f(lam):
                l_vec = (r_flat > lam * s_flat).astype(np.float64)
                l_plus = w[:, 1:] @ l_vec + w[:, 0] * self.max_risk
                return np.quantile(l_plus, self.hpd_level) - self.alpha

        elif self.method == "rcps":

            def f(lam):
                l_vec = (r_flat > lam * s_flat).astype(np.float64)
                ucb = _hoeffding_ucb(float(np.mean(l_vec)), n, self.delta)
                return ucb - self.alpha

        else:
            raise ValueError(f"unknown method {self.method}")

        lam_lo = 0.0
        lam_hi = float(self.max_threshold)
        self.lam_hat_ = float(_bisection(f, lam_lo, lam_hi))
        self.scale_shape_ = np.shape(scale)
        return self

    def apply(self, preds: np.ndarray, scale: np.ndarray):
        if self.lam_hat_ is None:
            raise RuntimeError("call fit() before apply().")
        lam = float(self.lam_hat_)
        p = np.asarray(preds, dtype=np.float64)
        s = np.asarray(scale, dtype=np.float64)
        if s.shape != p.shape:
            s = np.broadcast_to(s, p.shape)
        lower = p - lam * s
        upper = p + lam * s
        return lower, upper


class AdaptiveOnlineConformalCalibrator:
    """ACI-inspired online conformal calibration with horizon-wise local thresholds.

    The calibrator initializes from a chronological calibration split, then applies
    predict -> observe -> update on the test stream. It maintains a recent score
    buffer and a horizon-wise adaptive miscoverage target alpha_t. Thresholds are
    recomputed from the recent buffer either by empirical quantiles or by
    BQ-style threshold solvers (crc/hpd/rcps). To reduce overlap-induced noise,
    updates can be executed block-wise instead of every batch.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        update_lr: float = 0.005,
        buffer_size: int = 4096,
        alpha_min: float = 0.005,
        alpha_max: float = 0.30,
        pred_len: int | None = None,
        horizon_axis: int | None = None,
        threshold_method: str = "hpd",
        recompute_method: str = "bq",
        hpd_level: float = 0.95,
        num_dir: int = 1000,
        delta: float = 0.05,
        max_risk: float = 1.0,
        max_threshold: float | None = None,
        update_block_size: int = 1,
        rng_seed: int = 0,
    ) -> None:
        self.alpha = float(alpha)
        self.update_lr = float(update_lr)
        self.buffer_size = int(buffer_size)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.pred_len = pred_len
        self.horizon_axis = horizon_axis
        self.threshold_method = threshold_method
        self.recompute_method = recompute_method
        self.hpd_level = float(hpd_level)
        self.num_dir = int(num_dir)
        self.delta = float(delta)
        self.max_risk = float(max_risk)
        self.max_threshold = max_threshold
        self.update_block_size = max(1, int(update_block_size))
        self.rng_seed = int(rng_seed)
        self.rng = np.random.default_rng(rng_seed)

        self.horizon_axis_: int | None = None
        self.num_horizons_: int | None = None
        self.alpha_t_: np.ndarray | None = None
        self.lambda_t_: np.ndarray | None = None
        self.score_buffer_: np.ndarray | None = None
        self.pending_score_rows_: list[np.ndarray] = []
        self.pending_miss_rows_: list[np.ndarray] = []

    def _resolve_horizon_axis(self, arr: np.ndarray) -> int:
        if self.horizon_axis is not None:
            axis = self.horizon_axis
            if axis < 0:
                axis += arr.ndim
            return axis
        return infer_horizon_axis(arr.shape, pred_len=self.pred_len)

    def _broadcast_scale(self, arr: np.ndarray, scale: np.ndarray) -> np.ndarray:
        s = np.asarray(scale, dtype=np.float64)
        s = np.maximum(s, 1e-12)
        if s.shape != arr.shape:
            s = np.broadcast_to(s, arr.shape)
        return s

    def _scores_to_horizon_matrix(self, scores: np.ndarray, horizon_axis: int) -> np.ndarray:
        moved = np.moveaxis(scores, horizon_axis, -1)
        return moved.reshape(-1, moved.shape[-1])

    def _compute_lambda_from_buffer(self) -> np.ndarray:
        if self.score_buffer_ is None or self.alpha_t_ is None:
            raise RuntimeError("online conformal state is not initialized")
        lam = np.empty(self.score_buffer_.shape[1], dtype=np.float64)
        for h in range(self.score_buffer_.shape[1]):
            scores_h = self.score_buffer_[:, h]
            if self.recompute_method == "quantile":
                q = float(np.clip(1.0 - self.alpha_t_[h], 0.0, 1.0))
                lam[h] = float(np.quantile(scores_h, q))
            else:
                lam[h] = solve_bq_threshold(
                    scores=scores_h,
                    alpha=float(self.alpha_t_[h]),
                    method=self.threshold_method,
                    max_risk=self.max_risk,
                    lam_hi=self.max_threshold,
                    hpd_level=self.hpd_level,
                    num_dir=self.num_dir,
                    delta=self.delta,
                    rng=self.rng,
                )
        return np.maximum(lam, 1e-8)

    def fit(self, residuals: np.ndarray, scale: np.ndarray):
        residuals = np.asarray(residuals, dtype=np.float64)
        horizon_axis = self._resolve_horizon_axis(residuals)
        scale = self._broadcast_scale(residuals, scale)
        scores = np.abs(residuals) / scale
        score_matrix = self._scores_to_horizon_matrix(scores, horizon_axis)
        if score_matrix.shape[0] == 0:
            raise ValueError("no calibration scores available for online conformal initialization")

        self.horizon_axis_ = int(horizon_axis)
        self.num_horizons_ = int(score_matrix.shape[1])
        if self.buffer_size <= 0:
            self.buffer_size = score_matrix.shape[0]
        self.score_buffer_ = score_matrix[-self.buffer_size :].copy()
        self.alpha_t_ = np.full(self.num_horizons_, self.alpha, dtype=np.float64)
        self.lambda_t_ = self._compute_lambda_from_buffer()
        self.pending_score_rows_ = []
        self.pending_miss_rows_ = []
        return self

    def apply(self, preds: np.ndarray, scale: np.ndarray):
        if self.lambda_t_ is None or self.horizon_axis_ is None:
            raise RuntimeError("call fit() before apply().")
        preds = np.asarray(preds, dtype=np.float64)
        scale = self._broadcast_scale(preds, scale)
        lam_shape = [1] * preds.ndim
        lam_shape[self.horizon_axis_] = self.lambda_t_.shape[0]
        lam = self.lambda_t_.reshape(lam_shape)
        lower = preds - lam * scale
        upper = preds + lam * scale
        return lower, upper

    def update(self, preds: np.ndarray, trues: np.ndarray, scale: np.ndarray):
        if self.lambda_t_ is None or self.alpha_t_ is None or self.horizon_axis_ is None:
            raise RuntimeError("call fit() before update().")

        preds = np.asarray(preds, dtype=np.float64)
        trues = np.asarray(trues, dtype=np.float64)
        scale = self._broadcast_scale(preds, scale)
        lower, upper = self.apply(preds, scale)

        miss = ((trues < lower) | (trues > upper)).astype(np.float64)
        miss_matrix = self._scores_to_horizon_matrix(miss, self.horizon_axis_)
        scores = np.abs(trues - preds) / scale
        score_matrix = self._scores_to_horizon_matrix(scores, self.horizon_axis_)
        self.pending_score_rows_.append(score_matrix)
        self.pending_miss_rows_.append(miss_matrix)

        update_applied = False
        if len(self.pending_score_rows_) >= self.update_block_size:
            block_scores = np.concatenate(self.pending_score_rows_, axis=0)
            block_miss = np.concatenate(self.pending_miss_rows_, axis=0)
            block_miscoverage = block_miss.mean(axis=0)

            self.score_buffer_ = np.concatenate([self.score_buffer_, block_scores], axis=0)
            if self.score_buffer_.shape[0] > self.buffer_size:
                self.score_buffer_ = self.score_buffer_[-self.buffer_size :]

            self.alpha_t_ = np.clip(
                self.alpha_t_ + self.update_lr * (self.alpha - block_miscoverage),
                self.alpha_min,
                self.alpha_max,
            )
            self.lambda_t_ = self._compute_lambda_from_buffer()
            self.pending_score_rows_.clear()
            self.pending_miss_rows_.clear()
            batch_miscoverage = block_miscoverage
            update_applied = True
        else:
            batch_miscoverage = miss_matrix.mean(axis=0)

        return {
            "batch_miscoverage": batch_miscoverage,
            "mean_batch_miscoverage": float(np.mean(batch_miscoverage)),
            "mean_lambda": float(np.mean(self.lambda_t_)),
            "mean_alpha_t": float(np.mean(self.alpha_t_)),
            "update_applied": update_applied,
            "pending_blocks": len(self.pending_score_rows_),
        }

    def state_dict(self, scale: np.ndarray) -> dict:
        if self.lambda_t_ is None or self.alpha_t_ is None or self.score_buffer_ is None:
            raise RuntimeError("online conformal state is not initialized")
        return {
            "mode": np.array("online"),
            "scale": np.asarray(scale, dtype=np.float64),
            "score_buffer": self.score_buffer_.astype(np.float64),
            "alpha_t": self.alpha_t_.astype(np.float64),
            "lambda_t": self.lambda_t_.astype(np.float64),
            "update_lr": np.array(self.update_lr, dtype=np.float64),
            "buffer_size": np.array(self.buffer_size, dtype=np.int64),
            "alpha_min": np.array(self.alpha_min, dtype=np.float64),
            "alpha_max": np.array(self.alpha_max, dtype=np.float64),
            "alpha": np.array(self.alpha, dtype=np.float64),
            "pred_len": np.array(-1 if self.pred_len is None else self.pred_len, dtype=np.int64),
            "horizon_axis": np.array(self.horizon_axis_, dtype=np.int64),
            "threshold_method": np.array(self.threshold_method),
            "recompute_method": np.array(self.recompute_method),
            "hpd_level": np.array(self.hpd_level, dtype=np.float64),
            "num_dir": np.array(self.num_dir, dtype=np.int64),
            "delta": np.array(self.delta, dtype=np.float64),
            "max_risk": np.array(self.max_risk, dtype=np.float64),
            "max_threshold": np.array(-1.0 if self.max_threshold is None else self.max_threshold, dtype=np.float64),
            "update_block_size": np.array(self.update_block_size, dtype=np.int64),
            "rng_seed": np.array(self.rng_seed, dtype=np.int64),
        }

    @classmethod
    def from_npz(cls, payload):
        pred_len = int(payload["pred_len"])
        if pred_len <= 0:
            pred_len = None
        calibrator = cls(
            alpha=float(payload["alpha"]),
            update_lr=float(payload["update_lr"]),
            buffer_size=int(payload["buffer_size"]),
            alpha_min=float(payload["alpha_min"]),
            alpha_max=float(payload["alpha_max"]),
            pred_len=pred_len,
            horizon_axis=int(payload["horizon_axis"]),
            threshold_method=str(payload["threshold_method"]) if "threshold_method" in payload else "hpd",
            recompute_method=str(payload["recompute_method"]) if "recompute_method" in payload else "bq",
            hpd_level=float(payload["hpd_level"]) if "hpd_level" in payload else 0.95,
            num_dir=int(payload["num_dir"]) if "num_dir" in payload else 1000,
            delta=float(payload["delta"]) if "delta" in payload else 0.05,
            max_risk=float(payload["max_risk"]) if "max_risk" in payload else 1.0,
            max_threshold=(
                None
                if ("max_threshold" not in payload or float(payload["max_threshold"]) <= 0)
                else float(payload["max_threshold"])
            ),
            update_block_size=int(payload["update_block_size"]) if "update_block_size" in payload else 1,
            rng_seed=int(payload["rng_seed"]) if "rng_seed" in payload else 0,
        )
        calibrator.horizon_axis_ = int(payload["horizon_axis"])
        calibrator.score_buffer_ = np.asarray(payload["score_buffer"], dtype=np.float64)
        calibrator.alpha_t_ = np.asarray(payload["alpha_t"], dtype=np.float64)
        calibrator.lambda_t_ = np.asarray(payload["lambda_t"], dtype=np.float64)
        calibrator.num_horizons_ = int(calibrator.lambda_t_.shape[0])
        calibrator.pending_score_rows_ = []
        calibrator.pending_miss_rows_ = []
        return calibrator

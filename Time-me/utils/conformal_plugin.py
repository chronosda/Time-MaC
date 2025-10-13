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


class ConformalCalibrator:
    """Post-hoc conformal calibration for point forecasts.

    Calibrates a global threshold lambda such that miscoverage P[r > lambda * s] <= alpha,
    where r is residual magnitude and s is a scale proxy (e.g., MAD/STD).
    """

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
            W = self.rng.dirichlet(np.ones(n + 1), size=self.num_dir)
            def f(lam):
                l_vec = (r_flat > lam * s_flat).astype(np.float64)
                l_plus = W[:, 1:] @ l_vec + W[:, 0] * self.max_risk
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


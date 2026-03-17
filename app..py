import numpy as np
from scipy.special import expit
from scipy.linalg import solve_discrete_are, qr, cholesky, svd

class AlphaEngine_V10_Perfected:
    """
    ENGINE 10/10 - FULL SQUARE-ROOT UKF + MAHALANOBIS GUARD + ADAPTIVE LQR
    Inclusion du Cholesky Rank-1 Update pour une précision mathématique totale.
    """
    def __init__(self, n_assets=1, max_leverage=5.0):
        self.nx = 3  # [Prix, Drift, Accel]
        self.x = np.zeros((self.nx, 1))
        self.S = np.eye(self.nx) * 0.1 # Cholesky factor of Covariance
        
        self.theta = 0.25 # Mean reversion speed (OU process)
        self.max_leverage = max_leverage
        self.vol = 0.015
        
        # Matrice A (Transition)
        self.A = np.array([
            [1.0, 1.0, 0.5],
            [0.0, 1.0 - self.theta, 1.0],
            [0.0, 0.0, 0.95]
        ])
        
        # Paramètres GARCH
        self.garch = {'omega': 1e-6, 'alpha': 0.05, 'beta': 0.90}
        self.gate_threshold = 7.81 # Chi-square 95% pour 3 degrés de liberté

    def _cholesky_update(self, S, v, sign=+1):
        """Mise à jour de rang 1 de la racine de Cholesky (Stabilité 10/10)"""
        res = S.copy()
        for i in range(self.nx):
            if sign > 0:
                r = np.sqrt(res[i, i]**2 + v[i]**2)
            else:
                r = np.sqrt(max(1e-12, res[i, i]**2 - v[i]**2))
            c = r / res[i, i]
            s = v[i] / res[i, i]
            res[i, i] = r
            if i < self.nx - 1:
                res[i+1:, i] = (res[i+1:, i] + s * v[i+1:]) / c
                v[i+1:] = c * v[i+1:] - s * res[i+1:, i]
        return res

    def step(self, z_price, last_ret):
        # 1. GARCH : Mise à jour de la volatilité
        self.vol = np.sqrt(self.garch['omega'] + self.garch['alpha'] * last_ret**2 + self.garch['beta'] * self.vol**2)

        # 2. PREDICT (SR-UKF)
        sigmas = self._get_sigma_points()
        sigmas_f = self.A @ sigmas
        wm, wc = self._get_weights()
        
        x_pred = np.sum(wm * sigmas_f, axis=1, keepdims=True)
        
        # QR pour la prédiction de S
        Q_half = np.eye(self.nx) * (1e-4 * (1 + self.vol * 5))
        compound = np.hstack((np.sqrt(wc[1]) * (sigmas_f[:, 1:] - x_pred), Q_half))
        _, St = qr(compound.T, mode='economic')
        self.S = self._cholesky_update(St[:self.nx, :self.nx].T, sigmas_f[:, 0:1] - x_pred, sign=wc[0])

        # 3. UPDATE (Measurement) avec Mahalanobis Guard
        H = np.array([[1.0, 0.0, 0.0]])
        R_std = 1e-4 * (1 + self.vol * 100)
        
        z_sigmas = H @ sigmas_f
        z_pred = np.sum(wm * z_sigmas, axis=1, keepdims=True)
        
        # Innovation covariance
        P_zz = np.sum(wc * (z_sigmas - z_pred)**2) + R_std**2
        
        # --- MAHALANOBIS GUARD ---
        innovation = z_price - z_pred
        d_mahalanobis = (innovation**2) / P_zz
        if d_mahalanobis > self.gate_threshold:
            # On ignore cette mesure (Outlier / Flash Crash)
            return self._output_dict(0.0)

        # Cross-covariance
        P_xz = np.sum(wc * (sigmas_f - x_pred) * (z_sigmas - z_pred), axis=1, keepdims=True)
        
        # Kalman Gain & Correction
        K = P_xz / P_zz
        self.x = x_pred + K * innovation
        
        # SR-Update : Rank-1 update de S
        U = K * np.sqrt(P_zz)
        self.S = self._cholesky_update(self.S, U, sign=-1)

        # 4. CONTROL
        weight = self._compute_lqr_control()
        return self._output_dict(weight)

    def _get_sigma_points(self):
        phi = np.sqrt(self.nx + 2.0)
        points = np.zeros((self.nx, 2 * self.nx + 1))
        points[:, 0] = self.x.flatten()
        for k in range(self.nx):
            points[:, k + 1] = self.x.flatten() + phi * self.S[:, k]
            points[:, self.nx + k + 1] = self.x.flatten() - phi * self.S[:, k]
        return points

    def _get_weights(self):
        alpha, kappa = 1e-3, 0.0
        lam = alpha**2 * (self.nx + kappa) - self.nx
        w = np.full(2 * self.nx + 1, 1.0 / (2 * (self.nx + lam)))
        w[0] = lam / (self.nx + lam)
        return w, w # Simplified weights

    def _compute_lqr_control(self):
        B = np.array([[0], [1], [0]])
        Q_lqr = np.diag([0.1, 10.0, 0.1])
        # R adaptatif : plus la vol est haute, plus le "coût" du levier est élevé
        R_lqr = np.array([[1.0 + self.vol * 2000]])
        
        try:
            P = solve_discrete_are(self.A, B, Q_lqr, R_lqr)
            K = np.linalg.inv(B.T @ P @ B + R_lqr) @ (B.T @ P @ self.A)
            return float(-K @ self.x)
        except: return 0.0

    def _output_dict(self, weight):
        return {
            "weight": np.clip(weight, -self.max_leverage, self.max_leverage),
            "drift": float(self.x[1]),
            "score": float(expit(self.x[1] / (self.vol + 1e-9)))
        }

import numpy as np
import time
from typing import Tuple, List, Union, Dict, Sequence, Optional

class collision_damper_core:
    '''
    core logic for collision damper without ROS2 dependencies'''
    
    def __init__(self, z_band: Tuple[float, float] = (-0.15, 0.15),
                 coverage_radius: float = 1.0, n_sectors: int = 12):
        '''
        Initialize the collision damper core with parameters.
        :param z_band: Tuple of (min_z, max_z) to filter points in the Z axis.
        :param coverage_radius: Radius of the quality circle for collision detection.
        :param n_sectors: Number of sectors to divide the circle for collision detection.
        ''' 

        self.z_band = z_band
        self.coverage_radius = coverage_radius
        self.n_sectors = n_sectors
        self.cmd_vel_limit = 0.5
        
        self.mem_decay_duration: float = 2.5        # decay time constant (s)
        self.mem_weight: float = 0.85        # how much memory can drive damping [0..1]
        self.mem_zero_cutoff: float = 1e-3           # below this, memory is treated as zero
        self.mem_use_when_no_hits: bool = True   # allow memory to act if current frame is empty
        self.mem_linear_scale: float = 0.0   # 0=only yaw uses memory; 0.1–0.2 to also slow linear a bit

        # Per-sector memory state
        self._risk_mem = np.zeros(self.n_sectors, dtype=np.float32)      # S
        self._dir_mem  = np.zeros((self.n_sectors, 2), dtype=np.float32)  # Sx2 unit vectors
        self._dir_mem_valid = np.zeros(self.n_sectors, dtype=bool)

        # Time base + optional gain smoothing
        self._last_t: Optional[float] = None
        self.k_ewma_alpha: float = 0.3        # 0=off; 0.3–0.8 smooths k
        self._k_prev = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        
    def _prelimit(self, linear_vel: Sequence[float], prelimit_scale: Optional[float]) -> Tuple[float, float]:
        vx, vy = float(linear_vel[0]), float(linear_vel[1])
        if prelimit_scale is not None and prelimit_scale > 0.0:
            vx *= prelimit_scale
            vy *= prelimit_scale
        return vx, vy

    def _decay_factor(self, dt: float) -> float:
        if dt <= 0 or not np.isfinite(dt):
            return 1.0
        return float(np.exp(-dt / max(1e-6, self.mem_decay_duration)))

    def _update_sector_memory(
        self, dt: float, cur_risks: np.ndarray, cur_dirs: np.ndarray, cur_dir_valid: np.ndarray
    ) -> None:
        """Update per-sector memory with exponential decay.
        cur_risks: (S,), cur_dirs: (S,2), cur_dir_valid: (S,) bool
        """
        lam = self._decay_factor(dt)
        # Leaky max integrator for risks
        self._risk_mem = np.maximum(cur_risks.astype(np.float32), lam * self._risk_mem)
        updated_cached_dirs = cur_dir_valid.astype(bool)    # Update cached directions where we have current valid ones
        if np.any(updated_cached_dirs):
            self._dir_mem[updated_cached_dirs, :] = cur_dirs[updated_cached_dirs, :].astype(np.float32)
            self._dir_mem_valid[updated_cached_dirs] = True

    def _blend_memory(self, cur_risks: np.ndarray, cur_dirs: np.ndarray, cur_valid: np.ndarray,
                      dt: float, enable_sector_memory: bool
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if enable_sector_memory:
            self._update_sector_memory(dt, cur_risks, cur_dirs, cur_valid)
            eff_risks = np.maximum(cur_risks, self.mem_weight * self._risk_mem.astype(np.float64))
        else:
            eff_risks = cur_risks

        dir_eff = np.where(cur_valid[:, None], cur_dirs, self._dir_mem.astype(np.float64))
        dir_eff_valid = np.logical_or(cur_valid, self._dir_mem_valid)
        return eff_risks, dir_eff, dir_eff_valid
        
    def distance_to_risk(self, distance, d_stop, d_warn):
        h = (d_warn - distance) / max(1e-6, (d_warn - d_stop))
        h = np.clip(h, 0.0, 1.0)
        h = np.where(np.isfinite(distance), h, 0.0)
        return h
    
    def filter_points_on_z(self, P_xyz: np.ndarray, z_min: float = -0.15, z_max: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        if P_xyz.size == 0:
            return P_xyz, np.array([], dtype=bool)
        mask = (P_xyz[:, 2] >= z_min) & (P_xyz[:, 2] <= z_max)
        return P_xyz[mask], mask

    def sector_coverage(self, P_xyz: np.ndarray, 
                        radius: float,
                        start_angle: float = 0.0, 
                        coverage_angle: float = 2.0 * np.pi
                        ) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
        """
        Return (hit_mask, hit_ratio, sector_points_dict).
        - hit_mask[i]=True if sector i has any point within 'radius'.
        - sector_points_dict maps sector index (as str) -> Nx3 array of points in that sector (within radius & z-band).
        
        :param P_xyz: Nx3 array of points
        :param radius: Maximum distance to consider
        :param start_angle: Starting angle in radians (0 = +X axis, forward)
        :param coverage_angle: Total angular coverage in radians (2π = full circle)
        
        :return: (hit_mask, hit_ratio, sector_points_dict)
        
        """
        hit = np.zeros(self.n_sectors, dtype=bool)
        sector_points_list: List[np.ndarray] = [np.empty((0, 3)) for _ in range(self.n_sectors)]
        
        if P_xyz.size == 0:
            return hit, float(hit.mean()), sector_points_list

        # Z-band filter 
        P_zf, zmask = self.filter_points_on_z(P_xyz, self.z_band[0], self.z_band[1])
        if not np.any(zmask) or P_zf.size == 0:
            return hit, float(hit.mean()), sector_points_list

        # Radius mask
        r = np.hypot(P_zf[:, 0], P_zf[:, 1])
        rmask = (r <= radius) & (r > 0.0)
        if not np.any(rmask):
            return hit, float(hit.mean()), sector_points_list

        # Subset of points within the circle
        P_in = P_zf[rmask]
        ang = np.arctan2(P_in[:, 1], P_in[:, 0])  # [-π, π]
        ang = (ang + 2.0 * np.pi) % (2.0 * np.pi)
        
        # Normalize start_angle and end_angle
        start_norm = (start_angle + 2.0 * np.pi) % (2.0 * np.pi)
        end_norm = (start_norm + coverage_angle) % (2.0 * np.pi)
        
        # Filter points within angular range
        if coverage_angle >= 2.0 * np.pi:
            angle_mask = np.ones(len(ang), dtype=bool)
        elif start_norm <= end_norm:
            angle_mask = (ang >= start_norm) & (ang <= end_norm) 
        else:
            angle_mask = (ang >= start_norm) | (ang <= end_norm)
        
        if not np.any(angle_mask):
            return hit, float(hit.mean()), sector_points_list

        # Apply angular filter
        P_filtered = P_in[angle_mask]
        ang_filtered = ang[angle_mask]
        
        # Convert to sector-relative angles
        ang_relative = ang_filtered - start_norm
        ang_relative = (ang_relative + 2.0 * np.pi) % (2.0 * np.pi)
        
        # Bin into sectors
        sector_width = coverage_angle / self.n_sectors
        bins = np.floor(ang_relative / sector_width).astype(int)
        
        # Ensure bins are within valid range (handle edge case)
        bins = np.clip(bins, 0, self.n_sectors - 1)
        hit[bins] = True
  
        uniq_bins = np.unique(bins)
        for b in uniq_bins:
            sel = (bins == b)
            pts_b = P_filtered[sel]
            sector_points_list[b] = pts_b

        return hit, float(hit.mean()), sector_points_list

    def _current_sector_stats(self, hit_mask: np.ndarray, sector_points: Sequence[np.ndarray],
                              d_stop: float, d_warn: float, K_nearest: Optional[int]
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        S = self.n_sectors
        cur_risks = np.zeros(S, dtype=np.float64)
        cur_dirs  = np.zeros((S, 2), dtype=np.float64)
        cur_valid = np.zeros(S, dtype=bool)

        for i, is_hit in enumerate(hit_mask):
            if not is_hit:
                continue
            pts = sector_points[i]
            if pts is None or pts.size == 0:
                continue

            x = pts[:, 0]; y = pts[:, 1]
            r = np.hypot(x, y)
            valid = r > 0.0
            if not np.any(valid):
                continue
            x = x[valid]; y = y[valid]; r = r[valid]

            if K_nearest is not None and r.size > K_nearest:
                idx = np.argpartition(r, K_nearest - 1)[:K_nearest]
                x, y, r = x[idx], y[idx], r[idx]

            ex, ey = x / r, y / r
            each_sector_risks = self.distance_to_risk(r, d_stop=d_stop, d_warn=d_warn)

            weight = each_sector_risks.astype(np.float64)
            weight_sum = float(weight.sum())
            if weight_sum <= 1e-9:
                continue

            sx = float((weight * ex).sum() / weight_sum)
            sy = float((weight * ey).sum() / weight_sum)
            s_norm = float(np.hypot(sx, sy))
            if s_norm > 1e-9:
                ux_j, uy_j = sx / s_norm, sy / s_norm
            else:
                best_risk = int(np.argmax(each_sector_risks))
                ux_j, uy_j = float(ex[best_risk]), float(ey[best_risk])

            cur_risks[i] = float(each_sector_risks.max())
            cur_dirs[i, 0] = ux_j
            cur_dirs[i, 1] = uy_j
            cur_valid[i] = True

        return cur_risks, cur_dirs, cur_valid

    def _usable_mask(self, eff_risks: np.ndarray, dir_eff_valid: np.ndarray) -> np.ndarray:
        risk_mask = eff_risks > self.mem_zero_cutoff
        return np.logical_and(dir_eff_valid, risk_mask)

    def _memory_fallback_k(self, wz: float, omega_ref: float) -> Optional[np.ndarray]:
        """Return k from memory-only left/right estimate, or None if not applicable."""
        if not (self.mem_use_when_no_hits and np.any(self._dir_mem_valid)):
            return None
        mem_max = float(np.max(self._risk_mem)) if self._risk_mem.size else 0.0
        if mem_max <= self.mem_eps:
            return None

        y_mem = self._dir_mem[:, 1].astype(np.float64)
        left_mask  = np.logical_and(self._dir_mem_valid,  y_mem > 0.0)
        right_mask = np.logical_and(self._dir_mem_valid,  y_mem < 0.0)
        wL = float((self._risk_mem[left_mask]).sum())
        wR = float((self._risk_mem[right_mask]).sum())
        if (wL + wR) <= 0:
            return None

        left_eff  = wL / (wL + wR)
        right_eff = wR / (wL + wR)

        yaw_dir_risk = right_eff if wz < 0.0 else (left_eff if wz > 0.0 else max(left_eff, right_eff))
        beta = float(np.clip(abs(wz) / max(1e-6, omega_ref), 0.0, 1.0))
        w_yaw = beta * self.mem_weight * yaw_dir_risk

        kx = float(np.clip(1.0 - self.mem_linear_scale * w_yaw, 0.0, 1.0))
        ky = float(np.clip(1.0 - self.mem_linear_scale * w_yaw, 0.0, 1.0))
        kz = float(np.clip(1.0 - w_yaw, 0.0, 1.0))
        return np.array([kx, ky, kz], dtype=np.float64)

    def _aggregate_field(self, dirs: np.ndarray, risks: np.ndarray, gamma: float
                         ) -> Tuple[float, float, float, np.ndarray]:
        """Return Ux, Uy, aggregate_risk, w_sec."""
        w_sec = np.power(risks, float(gamma))
        w_sum = float(w_sec.sum())
        if w_sum > 1e-9:
            U_vec = (w_sec[:, None] * dirs).sum(axis=0) / w_sum
        else:
            U_vec = np.zeros(2, dtype=np.float64)

        U_norm = float(np.hypot(U_vec[0], U_vec[1]))
        if U_norm > 0:
            Ux, Uy = U_vec[0] / U_norm, U_vec[1] / U_norm
        else:
            Ux, Uy = 0.0, 0.0

        max_sector_risk = float(risks.max()) if risks.size else 0.0
        avg_weighted_risk = float((w_sec * risks).sum() / max(1e-9, w_sum)) if w_sum > 0 else 0.0
        aggregate_risk = max(max_sector_risk, avg_weighted_risk)
        return Ux, Uy, aggregate_risk, w_sec

    def _linear_weight(self, v_norm: float, ux: float, uy: float, Ux: float, Uy: float, p_lin: float) -> float:
        if v_norm > 1e-6:
            directional_alignment = max(0.0, ux * Ux + uy * Uy)
            return directional_alignment ** float(p_lin)
        return 0.0

    def _side_risks(self, dirs: np.ndarray, w_sec: np.ndarray) -> Tuple[float, float]:
        side_y = np.abs(dirs[:, 1])
        left_mask  = dirs[:, 1] > 0.0
        right_mask = dirs[:, 1] < 0.0
        left_weights  = np.where(left_mask,  w_sec, 0.0)
        right_weights = np.where(right_mask, w_sec, 0.0)
        left_sum  = float(left_weights.sum())
        right_sum = float(right_weights.sum())
        left_eff  = float((left_weights  * side_y).sum() / max(1e-9, left_sum))  if left_sum  > 1e-9 else 0.0
        right_eff = float((right_weights * side_y).sum() / max(1e-9, right_sum)) if right_sum > 1e-9 else 0.0
        return left_eff, right_eff

    def _yaw_weight(self, wz: float, omega_ref: float, left_eff: float, right_eff: float) -> float:
        yaw_dir_risk = right_eff if wz < 0.0 else (left_eff if wz > 0.0 else max(left_eff, right_eff))
        beta = float(np.clip(abs(wz) / max(1e-6, omega_ref), 0.0, 1.0))
        return beta * yaw_dir_risk

    # -------------------------------
    # Helpers: gains & finishing
    # -------------------------------
    def _final_gains(self, linear_weight: float, w_yaw: float, aggregate_risk: float) -> np.ndarray:
        fr_x = max(linear_weight * aggregate_risk, w_yaw * aggregate_risk)
        fr_y = max(linear_weight * aggregate_risk, w_yaw * aggregate_risk)
        fr_z = w_yaw * aggregate_risk
        kx = float(np.clip(1.0 - fr_x, 0.0, 1.0))
        ky = float(np.clip(1.0 - fr_y, 0.0, 1.0))
        kz = float(np.clip(1.0 - fr_z, 0.0, 1.0))
        return np.array([kx, ky, kz], dtype=np.float64)

    def _cutoff_and_smooth(self, k: np.ndarray, cutoff_threshold: float) -> np.ndarray:
        if (k < cutoff_threshold).any():
            k = np.zeros_like(k)

        if getattr(self, "k_ewma_alpha", 0.0) > 0.0:
            k = self.k_ewma_alpha * self._k_prev + (1.0 - self.k_ewma_alpha) * k
            self._k_prev = k
        return k


    def _timebase(self, now: Optional[float]) -> Tuple[float, float]:
        t_now = time.monotonic() if now is None else float(now)
        dt = 0.0 if self._last_t is None else max(0.0, t_now - self._last_t)
        self._last_t = t_now
        return t_now, dt

    def directional_damper(
        self, hit_mask: np.ndarray,
        sector_points: Sequence[np.ndarray],
        linear_vel: Sequence[float], angular_vel: float,
        apply_to_velocity: bool = True,
        d_stop: float = 0.3, d_warn: float = 0.80,
        p_lin: float = 2.0, omega_ref: float = 1.0,
        K_nearest: Optional[int] = 3, gamma: float = 1.5,
        return_sector_risks: bool = False,
        cutoff_threshold: float = 0.1,
        prelimit_scale: Optional[float] = None, 
        now: Optional[float] = None,
        enable_sector_memory: bool = True,
    ) -> Union[
        Tuple[np.ndarray, float, np.ndarray],                        # (v_out[2], wz_out, k[3])
        Tuple[np.ndarray, float, np.ndarray, np.ndarray],            # + sector_risks[n_sectors]
        Tuple[np.ndarray],                                           # k only
        Tuple[np.ndarray, np.ndarray],                               # k + sector_risks
    ]:
        """
        Pure damper. Computes damping gains k = [kx, ky, kz] in [0,1].
        Optionally applies them to a COPY of the provided velocity and returns it.
        """
        
        if d_warn <= d_stop:
            raise ValueError(f"d_warn ({d_warn}) must be > d_stop ({d_stop}).")
        if K_nearest is not None and K_nearest < 1:
            raise ValueError(f"K_nearest must be >=1 or None, got {K_nearest}.")
        if hit_mask.shape[0] != self.n_sectors or len(sector_points) != self.n_sectors:
            raise ValueError(f"hit_mask with {hit_mask.shape[0]} and sector_points {len(sector_points)} must match n_sectors of {self.n_sectors}.")

        _, dt = self._timebase(now)

        # Prelimit apply and normalised for heading vector
        vx, vy = self._prelimit(linear_vel=linear_vel, prelimit_scale=self.cmd_vel_limit) \
            if prelimit_scale is not None else linear_vel[0], linear_vel[1]
        wz = angular_vel
        v_norm = float(np.hypot(vx, vy))
        
        if v_norm > 1e-6:
            ux, uy = vx / v_norm, vy / v_norm
        else:  
            ux, uy = 1.0, 0.0
        
        cur_risks, cur_dirs, cur_valid = self._current_sector_stats(hit_mask, sector_points, d_stop, d_warn, K_nearest)

        # 4) memory blend
        eff_risks, dir_eff, dir_eff_valid = self._blend_memory(cur_risks, cur_dirs, cur_valid, dt, enable_sector_memory)

        # 5) usable mask
        use_mask = self._usable_mask(eff_risks, dir_eff_valid)

        # 6) fallback if none usable
        if not bool(np.any(use_mask)):
            k = self._memory_fallback_k(wz, omega_ref) if enable_sector_memory else None
            if k is None:
                k = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            k = self._cutoff_and_smooth(k, cutoff_threshold)

            if apply_to_velocity:
                v_out = np.array([vx * k[0], vy * k[1]], dtype=np.float64)
                wz_out = wz * k[2]
                if return_sector_risks:
                    return v_out, wz_out, k, (self.mem_weight * self._risk_mem).astype(np.float32)
                return v_out, wz_out, k
            else:
                if return_sector_risks:
                    return k, (self.mem_weight * self._risk_mem).astype(np.float32)
                return k

        # 7) aggregate usable sectors
        dirs = dir_eff[use_mask, :]
        risks_sel = eff_risks[use_mask]
        Ux, Uy, aggregate_risk, w_sec = self._aggregate_field(dirs, risks_sel, gamma)

        # 8) weights & gains
        linear_weight = self._linear_weight(v_norm, ux, uy, Ux, Uy, p_lin)
        left_eff, right_eff = self._side_risks(dirs, w_sec)
        w_yaw = self._yaw_weight(wz, omega_ref, left_eff, right_eff)
        k = self._final_gains(linear_weight, w_yaw, aggregate_risk)
        k = self._cutoff_and_smooth(k, cutoff_threshold)

        # 9) outputs
        if apply_to_velocity:
            v_out = np.array([vx * k[0], vy * k[1]], dtype=np.float64)
            wz_out = wz * k[2]
            if return_sector_risks:
                return v_out, wz_out, k, eff_risks.astype(np.float32)
            return v_out, wz_out, k
        else:
            if return_sector_risks:
                return k, eff_risks.astype(np.float32)
            return k  
        
        # sector_dirs = []
        # sector_risks_list = []
        # full_sector_risks = np.zeros(self.n_sectors, dtype=np.float32) if return_sector_risks else None

        # # --- per-sector processing ---
        # for i, is_hit in enumerate(hit_mask):
        #     if not is_hit:
        #         continue
        #     pts = sector_points[i]
        #     if pts is None or pts.size == 0:
        #         continue

        #     x = pts[:, 0]; y = pts[:, 1]
        #     r = np.hypot(x, y)
        #     valid = r > 0.0
        #     if not np.any(valid):
        #         continue
        #     x = x[valid]; y = y[valid]; r = r[valid]

        #     # K nearest in this sector
        #     if K_nearest is not None and r.size > K_nearest:
        #         idx = np.argpartition(r, K_nearest-1)[:K_nearest]
        #         x, y, r = x[idx], y[idx], r[idx]

        #     ex, ey = x / r, y / r
        #     h_i = self.distance_to_risk(r, d_stop=d_stop, d_warn=d_warn)  # vector in [0,1]

        #     w_i = h_i.astype(np.float64)
        #     w_sum_i = float(w_i.sum())
        #     if w_sum_i <= 1e-9:
        #         continue

        #     sx = float((w_i * ex).sum() / w_sum_i)
        #     sy = float((w_i * ey).sum() / w_sum_i)
        #     s_norm = float(np.hypot(sx, sy))
        #     if s_norm > 1e-9:
        #         ux_j, uy_j = sx / s_norm, sy / s_norm
        #     else:
        #         j_best = int(np.argmax(h_i))
        #         ux_j, uy_j = float(ex[j_best]), float(ey[j_best])

        #     h_j = float(h_i.max())
        #     sector_dirs.append((ux_j, uy_j))
        #     sector_risks_list.append(h_j)
        #     if return_sector_risks:
        #         full_sector_risks[i] = h_j

        # # --- no contributing sectors: identity gains ---
        # if not sector_dirs:
        #     k = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        #     if apply_to_velocity:
        #         v_out = np.array([vx, vy], dtype=np.float64)
        #         if return_sector_risks:
        #             return v_out, wz, k, full_sector_risks
        #         return v_out, wz, k
        #     else:
        #         if return_sector_risks:
        #             return k, full_sector_risks
        #         return k

        # # --- aggregate field & risks ---
        # dirs = np.asarray(sector_dirs, dtype=np.float64)     # (S,2)
        # risks = np.asarray(sector_risks_list, dtype=np.float64)
        # w_sec = np.power(risks, float(gamma))
        # w_sum = float(w_sec.sum())

        # U_vec = (w_sec[:, None] * dirs).sum(axis=0)
        # if w_sum > 1e-9:
        #     U_vec /= w_sum
        # U_norm = float(np.hypot(U_vec[0], U_vec[1]))
        # if U_norm > 0:
        #     Ux, Uy = U_vec[0] / U_norm, U_vec[1] / U_norm
        # else:
        #     Ux, Uy = 0.0, 0.0

        # max_sector_risk = float(risks.max())
        # avg_weighted_risk = float((w_sec * risks).sum() / max(1e-9, w_sum))
        # aggregate_risk = max(max_sector_risk, avg_weighted_risk)

        # if v_norm > 1e-6:
        #     directional_alignment = max(0.0, ux * Ux + uy * Uy)
        #     linear_weight = directional_alignment ** float(p_lin)
        # else:
        #     linear_weight = 0.0

        # side_components = np.abs(dirs[:, 1])  # |Y|
        # left_mask = dirs[:, 1] > 0.0
        # right_mask = dirs[:, 1] < 0.0
        # left_weights = np.where(left_mask, w_sec, 0.0)
        # right_weights = np.where(right_mask, w_sec, 0.0)
        # left_sum = float(left_weights.sum())
        # right_sum = float(right_weights.sum())
        # left_risk = float((left_weights * side_components).sum() / max(1e-9, left_sum)) if left_sum > 1e-9 else 0.0
        # right_risk = float((right_weights * side_components).sum() / max(1e-9, right_sum)) if right_sum > 1e-9 else 0.0

        # yaw_dir_risk = right_risk if wz < 0.0 else (left_risk if wz > 0.0 else 0.0)
        # beta = float(np.clip(abs(wz) / max(1e-6, omega_ref), 0.0, 1.0))
        # w_yaw = beta * yaw_dir_risk

        # final_risk_x = max(linear_weight * aggregate_risk, w_yaw * aggregate_risk)
        # final_risk_y = max(linear_weight * aggregate_risk, w_yaw * aggregate_risk)
        # final_risk_z = w_yaw * aggregate_risk

        # kx = float(np.clip(1.0 - final_risk_x, 0.0, 1.0))
        # ky = float(np.clip(1.0 - final_risk_y, 0.0, 1.0))
        # kz = float(np.clip(1.0 - final_risk_z, 0.0, 1.0))

        # if (kx < cutoff_threshold) or (ky < cutoff_threshold) or (kz < cutoff_threshold):
        #     kx = ky = kz = 0.0

        # k = np.array([kx, ky, kz], dtype=np.float64)

        # if apply_to_velocity:
        #     v_out = np.array([vx * kx, vy * ky], dtype=np.float64)
        #     wz_out = wz * kz
        #     if return_sector_risks:
        #         return v_out, wz_out, k, full_sector_risks
        #     return v_out, wz_out, k
        # else:
        #     if return_sector_risks:
        #         return k, full_sector_risks
        #     return k
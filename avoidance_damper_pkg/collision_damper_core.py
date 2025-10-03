import numpy as np
from typing import Tuple, List, Union, Dict
from geometry_msgs.msg import Twist #  TODO: remove ROS2 dependency

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
        if P_xyz.size == 0:
            return np.zeros(self.n_sectors, dtype=bool), 0.0, {}

        # Z-band filter 
        P_zf, zmask = self.filter_points_on_z(P_xyz, self.z_band[0], self.z_band[1])
        if not np.any(zmask) or P_zf.size == 0:
            return np.zeros(self.n_sectors, dtype=bool), 0.0, {}

        # Radius mask
        r = np.hypot(P_zf[:, 0], P_zf[:, 1])
        rmask = (r <= radius) & (r > 0.0)
        if not np.any(rmask):
            return np.zeros(self.n_sectors, dtype=bool), 0.0, {}

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
            return np.zeros(self.n_sectors, dtype=bool), 0.0, [np.empty((0, 3), dtype=P_xyz.dtype) for _ in range(self.n_sectors)]

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
        hit = np.zeros(self.n_sectors, dtype=bool)
        hit[bins] = True

        sector_points_list: List[np.ndarray] = [np.empty((0, 3), dtype=P_filtered.dtype) for _ in range(self.n_sectors)]
        uniq_bins = np.unique(bins)
        for b in uniq_bins:
            sel = (bins == b)
            pts_b = P_filtered[sel]
            sector_points_list[b] = pts_b

        return hit, float(hit.mean()), sector_points_list
    
    def directional_damper(self, hit_mask: np.ndarray, sector_points: list,               
                           cmd_vel: Twist, using_cmd_vel: bool = True,
                           d_stop=0.35, d_warn=0.80,
                           p_lin=2.0, q_rot=1.0,
                           omega_ref=1.0, K_nearest: int = 3,
                           gamma: float = 1.5,
                           return_sector_risks: bool = False,
                           cutoff_threshold: float = 0.1,
                           ) -> Union[Tuple[Twist, np.ndarray], Tuple[Twist, np.ndarray, np.ndarray], np.ndarray]:
        """
        Compute the directional damping factors (k_x, k_y, k_z) in [0,1] based on the sectors' risks.
        
        :param cmd_vel: geometry_msgs/Twist command velocity.
        :param hit_mask: Boolean array of shape (n_sectors,) indicating which sectors have obstacles.
        :param sector_points: List of length n_sectors, each element is an (Mi,3) numpy array of points in that sector.
        :param using_cmd_vel: If False, only compute and return the damping factors without modifying cmd_vel.
        :param d_stop: Distance at which risk is 1.0 (immediate stop).
        :param d_warn: Distance at which risk starts to rise above 0.0.
        :param p_lin: Exponent shaping the forward cone for linear velocity damping.
        :param q_rot: Exponent shaping the yaw damping based on side risks.
        :param omega_ref: Reference angular speed for scaling yaw damping.
        :param K_nearest: Use up to K nearest points per sector for risk calculation.
        :param gamma: Exponent for sector risk weighting.
        :param return_sector_risks: If True, also return individual sector risks for visualization.
        :param cutoff_threshold: If damping gain drops below this threshold, completely stop the robot.

        :return: If using_cmd_vel is True, returns (modified_cmd_vel, k_vector, [sector_risks]).
                 If using_cmd_vel is False, returns k_vector [, sector_risks].
                 sector_risks is only included if return_sector_risks=True.

        """
        
        cmd_vel.linear.x *= self.cmd_vel_limit
        cmd_vel.linear.y *= self.cmd_vel_limit
        cmd_vel.angular.z *= 1  # no limit on angular speed

        vx, vy = cmd_vel.linear.x, cmd_vel.linear.y
        v_norm = float(np.hypot(vx, vy))

        # Command unit headings (for linear directional part)
        if v_norm > 1e-6:
            ux, uy = vx / v_norm, vy / v_norm          # forward
        else:
            ux, uy = 1.0, 0.0                          # dummy; gated below

        sector_dirs = []       # list of (ux_j, uy_j) unit bearings
        sector_risks = []      # list of h_j in [0,1]
        h_global_max = 0.0
        
        # Initialize full sector risks array for visualization
        full_sector_risks = np.zeros(self.n_sectors, dtype=np.float32) if return_sector_risks else None

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

            # K nearest in this sector
            if K_nearest is not None and r.size > K_nearest:
                idx = np.argpartition(r, K_nearest-1)[:K_nearest]
                x, y, r = x[idx], y[idx], r[idx]

            ex, ey = x / r, y / r
            h_i = self.distance_to_risk(r, d_stop=d_stop, d_warn=d_warn)  # vector in [0,1]

            # Sector direction: risk-weighted average of bearings
            w_i = h_i.astype(np.float64)
            w_sum_i = float(w_i.sum())
            if w_sum_i <= 1e-9:
                continue

            sx = float((w_i * ex).sum() / w_sum_i)
            sy = float((w_i * ey).sum() / w_sum_i)
            s_norm = float(np.hypot(sx, sy))
            if s_norm > 1e-9:
                ux_j, uy_j = sx / s_norm, sy / s_norm
            else:
                j_best = int(np.argmax(h_i))
                ux_j, uy_j = float(ex[j_best]), float(ey[j_best])

            h_j = float(h_i.max())              # conservative per-sector risk
            h_global_max = max(h_global_max, h_j)

            sector_dirs.append((ux_j, uy_j))
            sector_risks.append(h_j)
            
            # Store risk for this sector if requested
            if return_sector_risks:
                full_sector_risks[i] = h_j

        # No contributing sectors -> no damping
        if not sector_dirs:
            k_vector = np.array([1.0, 1.0, 1.0])
            if not using_cmd_vel:
                return (k_vector, full_sector_risks) if return_sector_risks else k_vector
            result = (cmd_vel, k_vector)
            if return_sector_risks:
                result = result + (full_sector_risks,)
            return result
        
        dirs = np.asarray(sector_dirs, dtype=np.float64)  # (S,2)
        risks = np.asarray(sector_risks, dtype=np.float64)
        w_sec = np.power(risks, float(gamma))             # risk^gamma weights
        w_sum = float(w_sec.sum())

        # ---- Aggregate obstacle direction U ----
        U_vec = (w_sec[:, None] * dirs).sum(axis=0)
        if w_sum > 1e-9:
            U_vec /= w_sum
        U_norm = float(np.hypot(U_vec[0], U_vec[1]))
        Ux, Uy = (U_vec / max(U_norm, 1e-12)) if U_norm > 0 else (0.0, 0.0)

        max_sector_risk = float(risks.max()) if len(risks) > 0 else 0.0
        avg_weighted_risk = float((w_sec * risks).sum() / max(1e-9, w_sum))
        aggregate_risk = max(max_sector_risk, avg_weighted_risk)

        if v_norm > 1e-6:
            directional_alignment = max(0.0, ux * Ux + uy * Uy)
            w_lin_combined = directional_alignment ** float(p_lin)
        else:
            w_lin_combined = 0.0

        # Use absolute values to ensure symmetric treatment of left/right
        side_components = np.abs(dirs[:, 1])  # |Y component| of obstacle directions
        
        # Determine which obstacles are on left vs right based on Y sign
        left_mask = dirs[:, 1] > 0.0   # Positive Y = left side
        right_mask = dirs[:, 1] < 0.0  # Negative Y = right side
        
        # Calculate risks only for obstacles actually on each side
        left_weights = np.where(left_mask, w_sec, 0.0)
        right_weights = np.where(right_mask, w_sec, 0.0)
        
        # Normalize by the weights of obstacles actually on each side
        left_weight_sum = max(1e-9, left_weights.sum())
        right_weight_sum = max(1e-9, right_weights.sum())
        
        # Calculate side risks as weighted average of side components
        left_risk = float((left_weights * side_components).sum() / left_weight_sum) if left_weight_sum > 1e-9 else 0.0
        right_risk = float((right_weights * side_components).sum() / right_weight_sum) if right_weight_sum > 1e-9 else 0.0

        # Map yaw sign to the side you sweep into:
        yaw_dir_risk = 0.0
        if cmd_vel.angular.z < 0.0:
            yaw_dir_risk = right_risk
        elif cmd_vel.angular.z > 0.0:
            yaw_dir_risk = left_risk

        # Scale yaw influence with |ω| so tiny spins don't over-damp
        beta = float(np.clip(abs(cmd_vel.angular.z) / max(1e-6, omega_ref), 0.0, 1.0))
        w_yaw = beta * yaw_dir_risk

        if v_norm > 1e-6:
            w_lin_combined = max(0.0, ux * Ux + uy * Uy) ** float(p_lin)
        else:
            w_lin_combined = 0.0

        final_risk_x = max(w_lin_combined * aggregate_risk, w_yaw * aggregate_risk)
        final_risk_y = max(w_lin_combined * aggregate_risk, w_yaw * aggregate_risk)
        final_risk_z = w_yaw * aggregate_risk

        k_x = float(np.clip(1.0 - final_risk_x, 0.0, 1.0))
        k_y = float(np.clip(1.0 - final_risk_y, 0.0, 1.0))
        k_z = float(np.clip(1.0 - final_risk_z, 0.0, 1.0))

        # Apply hard cutoff threshold - if any gain drops below threshold, stop completely
        if k_x < cutoff_threshold or k_y < cutoff_threshold or k_z < cutoff_threshold:
            k_x = k_y = k_z = 0.0

        k_vector = np.array([k_x, k_y, k_z])
        if using_cmd_vel:
            cmd_vel.linear.x *= k_x
            cmd_vel.linear.y *= k_y
            cmd_vel.angular.z *= k_z

        # Return results based on parameters
        if using_cmd_vel:
            result = (cmd_vel, k_vector)
            if return_sector_risks:
                result = result + (full_sector_risks,)
            return result
        else:
            return (k_vector, full_sector_risks) if return_sector_risks else k_vector
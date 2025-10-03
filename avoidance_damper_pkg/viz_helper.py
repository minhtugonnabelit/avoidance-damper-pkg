from typing import Tuple, List

from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

import numpy as np

class VizHelper:
    
    @staticmethod    
    def make_sector_marker(
        frame_id: str = "base_link",
        radius: float = 1.0,
        start_angle: float = 0.0,  # radians
        end_angle: float = np.pi/6,  # radians
        z: float = 0.0,
        color=(1.0, 0.0, 0.0, 0.5),  # RGBA - red with transparency
        ns: str = "sector",
        mid: int = 0,
        lifetime_sec: float = 0.0,
    ) -> Marker:
        """Create a TRIANGLE_LIST sector marker in XY plane."""
        m = Marker()
        m.header.frame_id = frame_id
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.ns = ns
        m.id = mid
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = color
        if lifetime_sec > 0:
            m.lifetime = Duration(sec=int(lifetime_sec))
        m.pose.orientation.w = 1.0

        # Create sector as triangular fan from origin
        from geometry_msgs.msg import Point
        pts = []
        
        # Center point
        center = Point()
        center.x, center.y, center.z = 0.0, 0.0, float(z)
        
        # Arc points (more segments for smoother curve)
        n_segments = 20
        for i in range(n_segments + 1):
            angle = start_angle + (end_angle - start_angle) * i / n_segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            arc_point = Point()
            arc_point.x, arc_point.y, arc_point.z = float(x), float(y), float(z)
            
            # Create triangle: center -> current point -> next point
            if i > 0:
                pts.extend([center, prev_point, arc_point])
            prev_point = arc_point
        
        m.points = pts
        return m
    
    @staticmethod
    def risk_to_color(risk: float, alpha: float = 0.6, is_max_risk: bool = False) -> Tuple[float, float, float, float]:
        """Convert risk value (0.0 to 1.0) to RGBA color (green to red gradient)."""
        risk = np.clip(risk, 0.0, 1.0)
        # Green to red gradient: Green(0,1,0) -> Yellow(1,1,0) -> Red(1,0,0)
        if risk < 0.5:
            # Green to Yellow
            r = 2.0 * risk
            g = 1.0
            b = 0.0
        else:
            # Yellow to Red
            r = 1.0
            g = 2.0 * (1.0 - risk)
            b = 0.0
        
        # Highlight maximum risk sector with higher opacity and slight blue tint
        if is_max_risk and risk > 0.1:
            alpha = min(0.9, alpha + 0.3)  # Increase opacity
            b = 0.3  # Add blue component to make it stand out
            
        return (r, g, b, alpha)
    
    @staticmethod
    def make_all_sectors_marker(
        frame_id: str = "base_link",
        radius: float = 1.0, n_sectors: int = 12,
        hit_mask: np.ndarray = None,
        sector_risks: np.ndarray = None,
        start_angle: float = 0.0,
        coverage_angle: float = 2.0 * np.pi,
        z: float = 0.01,
        clear_color=(0.0, 1.0, 0.0, 0.3),
        ns: str = "sectors", lifetime_sec: float = 0.0,
    ) -> List[Marker]:
        """Create markers for sectors within the specified angular range with risk-based colors."""
        markers = []
        sector_width = coverage_angle / n_sectors
        
        # Find the sector with maximum risk
        max_risk_sector = -1
        if sector_risks is not None and len(sector_risks) > 0:
            max_risk_value = np.max(sector_risks)
            if max_risk_value > 0.0:
                max_risk_sector = int(np.argmax(sector_risks))
        
        for i in range(n_sectors):
            sector_start = start_angle + i * sector_width
            sector_end = sector_start + sector_width
            
            # Check if this is the maximum risk sector
            is_max_risk = (i == max_risk_sector)
            
            # Choose color based on risk level
            if sector_risks is not None and i < len(sector_risks):
                # Use risk-based color gradient with max risk highlighting
                risk = sector_risks[i]
                color = VizHelper.risk_to_color(risk, alpha=0.6, is_max_risk=is_max_risk)
            elif hit_mask is not None and i < len(hit_mask) and hit_mask[i]:
                # Fallback to simple red for hits without risk data
                color = (1.0, 0.0, 0.0, 0.6)
            else:
                # Clear sector
                color = clear_color
                
            marker = VizHelper.make_sector_marker(
                frame_id=frame_id,
                radius=radius,
                start_angle=sector_start,
                end_angle=sector_end,
                z=z,
                color=color,
                ns=ns,
                mid=i,
                lifetime_sec=lifetime_sec
            )
            markers.append(marker)
        
        return markers
        
    @staticmethod
    def make_circle_marker(
        frame_id: str = "base_link",
        radius: float = 1.0,
        z: float = 0.0,
        segments: int = 72,
        color=(0.0, 1.0, 0.0, 0.9),  # RGBA
        thickness: float = 0.01,
        ns: str = "coverage_circle",
        mid: int = 1,
        lifetime_sec: float = 0.0,
    ) -> Marker:
        """Create a LINE_STRIP circle marker in XY plane of `frame_id`."""
        m = Marker()
        m.header.frame_id = frame_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.ns = ns
        m.id = mid
        m.scale.x = thickness  # line width
        m.color.r, m.color.g, m.color.b, m.color.a = color
        if lifetime_sec > 0:
            m.lifetime = Duration(sec=int(lifetime_sec))
        m.pose.orientation.w = 1.0  # identity

        # points
        pts = []
        for i in range(segments + 1):  # close the loop
            th = 2.0 * np.pi * i / segments
            x = radius * np.cos(th)
            y = radius * np.sin(th)
            from geometry_msgs.msg import Point
            p = Point(); p.x = float(x); p.y = float(y); p.z = float(z)
            pts.append(p)
        m.points = pts
        return m
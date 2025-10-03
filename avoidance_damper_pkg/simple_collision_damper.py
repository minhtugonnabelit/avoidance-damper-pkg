#!/usr/bin/env python3

from typing import Tuple, List, Union, Dict

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from rcl_interfaces.msg import SetParametersResult

from avoidance_damper_pkg.collision_damper_core import collision_damper_core

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

class SimpleCollisionDamperNode(Node):
    def __init__(self):
        super().__init__('simple_collision_damper_node')

        self.get_logger().info('Simple Collision Damper Node has been started.')

        self.declare_parameter('pc_topic_name', "/utlidar/cloud_base")
        self.declare_parameter('twist_topic_name', "/admittance_controller/cmd_vel")
        self.declare_parameter('using_cmd_vel', True)
        self.declare_parameter('coverage_radius', 1.0)
        self.declare_parameter('n_sectors', 8)
        self.declare_parameter('z_band', (-0.15, 0.2))
        self.declare_parameter('dist_thresh', (0.6, 0.9))
        self.declare_parameter('timer_period', 0.1)  # seconds
        self.declare_parameter('cmd_vel_limit', 1.0)  # m/s, limit for cmd_vel linear speed
        self.declare_parameter('side_coverage_deg', 250.0) # New parameter

        pc_topic_name = self.get_parameter('pc_topic_name').get_parameter_value().string_value
        twist_topic_name = self.get_parameter('twist_topic_name').get_parameter_value().string_value
        self.using_cmd_vel = self.get_parameter('using_cmd_vel').get_parameter_value().bool_value
        self.coverage_radius = self.get_parameter('coverage_radius').get_parameter_value().double_value
        self.n_sectors = self.get_parameter('n_sectors').get_parameter_value().integer_value
        self.z_band = self.get_parameter('z_band').get_parameter_value().double_array_value
        self.dist_thresh = self.get_parameter('dist_thresh').get_parameter_value().double_array_value
        self.timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.cmd_vel_limit = self.get_parameter('cmd_vel_limit').get_parameter_value().double_value
        side_coverage_deg = self.get_parameter('side_coverage_deg').get_parameter_value().double_value
        
        self.side_coverage = self.side_coverage = side_coverage_deg * np.pi / 180.0

        self._pointcloud_subscriber = self.create_subscription(
            PointCloud2,
            pc_topic_name,
            self.pointcloud_callback,
            10)
        
        self._cmd_vel_subscriber = self.create_subscription(
            Twist,
            twist_topic_name,
            self.cmd_vel_callback,
            10
        )

        self.coverage_pub = self.create_publisher(Marker, "/coverage_markers", 1)
        self.sector_pub = self.create_publisher(MarkerArray, "/sector_markers", 1)
        self.damper_publisher_ = self.create_publisher(Float64MultiArray, '/damper_gains', 10)
        self.damped_cmd_vel_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.coverage_timer = self.create_timer(self.timer_period, self._coverage_viz_tick)

        self._damper_core = collision_damper_core(
            z_band=self.z_band,
            n_sectors=self.n_sectors,
            coverage_radius=self.coverage_radius,
        )
        
        self._pcl_numpy = np.empty((0, 3), dtype=np.float32) 
        self._cmd_vel = Twist()
        self._damped_cmd_vel = Twist()
        self._damping_gain = np.zeros(self.n_sectors, dtype=np.float32)
        self.latest_hit_mask = None
        self.latest_sector_risks = None  # Add this to store sector risks
        self.get_logger().info('Simple Collision Damper Node initialized.')

    def parameter_callback(self, params : List[rclpy.parameter.Parameter]) -> SetParametersResult:
        for param in params:
            if param.name == 'coverage_radius':
                self.coverage_radius = param.value
                self._damper_core.coverage_radius = self.coverage_radius
                self.get_logger().info(f"Updated coverage_radius: {self.coverage_radius}")
            elif param.name == 'side_coverage_deg':
                self.side_coverage = param.value * np.pi / 180.0
                self._damper_core.side_coverage = self.side_coverage
                self.get_logger().info(f"Updated side_coverage: {param.value} deg")
            elif param.name == 'cmd_vel_limit':
                self.cmd_vel_limit = param.value
                self._damper_core.cmd_vel_limit = self.cmd_vel_limit
                self.get_logger().info(f"Updated cmd_vel_limit: {self.cmd_vel_limit}")
            elif param.name == 'z_band':
                self.z_band = param.value
                self._damper_core.z_band = self.z_band
                self.get_logger().info(f"Updated z_band: {self.z_band}")
            elif param.name == 'dist_thresh':
                self.dist_thresh = param.value
                self.get_logger().info(f"Updated distance threshold: {self.dist_thresh}")
            elif param.name == 'n_sectors':
                self.n_sectors = param.value
                self._damper_core.n_sectors = self.n_sectors
                self.get_logger().info(f"Updated n_sectors: {self.n_sectors}")
                
        return SetParametersResult(successful=True)

    def _coverage_viz_tick(self):
        m = VizHelper.make_circle_marker(
            frame_id="base_link",
            radius=self.coverage_radius,
            segments=30,
            color=(0.0, 1.0, 0.0, 0.9),
            thickness=0.01,
            ns="coverage_circle",
            mid=1,
            lifetime_sec=0.0
        )
        self.coverage_pub.publish(m)

        # Sector markers
        if self.latest_hit_mask is not None:
            sector_markers = VizHelper.make_all_sectors_marker(
                frame_id="base_link",
                radius=self.coverage_radius,
                n_sectors=self.n_sectors,
                hit_mask=self.latest_hit_mask,
                sector_risks=self.latest_sector_risks,
                start_angle=-self.side_coverage/2,
                coverage_angle=self.side_coverage,
                clear_color=(0.0, 1.0, 0.0, 0.3),
                ns="sectors",
                lifetime_sec=0.2
            )
            
            # Create MarkerArray
            marker_array = MarkerArray()
            marker_array.markers = sector_markers
            self.sector_pub.publish(marker_array)
        
        
    def pointcloud_callback(self, msg : PointCloud2):
         
        # Convert PointCloud2 message to a unstructured numpy array
        self._pcl_numpy = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True, uvs=None)
        self._pcl_numpy = np.array([tuple(p) for p in self._pcl_numpy], dtype=np.float32)

        hit_mask, _, sector_points = self._damper_core.sector_coverage(self._pcl_numpy, 
                                                                       self.coverage_radius, 
                                                                       start_angle=-self.side_coverage/2, 
                                                                       coverage_angle=self.side_coverage)
        self.latest_hit_mask = hit_mask

        # Apply directional damper to cmd_vel and get sector risks for visualization
        if self.using_cmd_vel:
            self._damped_cmd_vel, self._damping_gain, self.latest_sector_risks = self._damper_core.directional_damper(
                cmd_vel=self._cmd_vel,
                hit_mask=hit_mask,
                sector_points=sector_points,
                using_cmd_vel=self.using_cmd_vel,
                d_stop=self.dist_thresh[0] * self.coverage_radius,
                d_warn=self.dist_thresh[1] * self.coverage_radius,
                return_sector_risks=True,
            )
            if self._damped_cmd_vel is not None:
                 
                if min(self._damping_gain) < 0.1:
                    self.get_logger().error(f"Damping gain too low: {self._damping_gain}")
                else:
                    self.get_logger().info(f"Damping gain: {self._damping_gain}")
        else:
            self._damping_gain, self.latest_sector_risks = self._damper_core.directional_damper(
                cmd_vel=self._cmd_vel,
                hit_mask=hit_mask,
                sector_points=sector_points,
                using_cmd_vel=self.using_cmd_vel,
                d_stop=self.dist_thresh[0] * self.coverage_radius,
                d_warn=self.dist_thresh[1] * self.coverage_radius,
                return_sector_risks=True,
            )
            if not np.all(self._damping_gain == 0):
                self.get_logger().info(f"Calculated damping gain: {np.max(self._damping_gain):.3f}")

        self.damper_publisher_.publish(Float64MultiArray(data=self._damping_gain.tolist()))
        self.damped_cmd_vel_publisher_.publish(self._damped_cmd_vel)

    def cmd_vel_callback(self, msg: Twist):
        self._cmd_vel = msg


def main(args=None):
    rclpy.init(args=args)
    
    node = SimpleCollisionDamperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import numpy as np
from typing import List

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.guard_condition import GuardCondition

from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition

from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray
from rcl_interfaces.msg import SetParametersResult
from visualization_msgs.msg import Marker, MarkerArray

from avoidance_damper_pkg.viz_helper import VizHelper
from avoidance_damper_pkg.collision_damper_core import collision_damper_core

class CollisionDamper(LifecycleNode):
    
    def __init__(self):
        super().__init__(node_name="collision_damper")
        self.declare_parameter('autostart', True)
        self.autostart = self.get_parameter('autostart').get_parameter_value().bool_value
        
        self.declare_parameter('pc_topic_name', "/utlidar/cloud_base")
        self.declare_parameter('twist_topic_name', "/teleop/cmd_vel")
        self.declare_parameter('using_cmd_vel', True)
        self.declare_parameter('coverage_radius', 1.0)
        self.declare_parameter('n_sectors', 8)
        self.declare_parameter('z_band', (-0.15, 0.2))
        self.declare_parameter('dist_thresh', (0.6, 0.9))
        self.declare_parameter('timer_period', 0.1)  # seconds
        self.declare_parameter('cmd_vel_limit', 1.0)  # m/s, limit for cmd_vel linear speed
        self.declare_parameter('side_coverage_deg', 250.0) # New parameter
        
    def on_configure(self, state : State) -> TransitionCallbackReturn:

        self.get_logger().info('on_configure() is called')
        self.using_cmd_vel = self.get_parameter('using_cmd_vel').get_parameter_value().bool_value
        self.coverage_radius = self.get_parameter('coverage_radius').get_parameter_value().double_value
        self.n_sectors = self.get_parameter('n_sectors').get_parameter_value().integer_value
        self.z_band = self.get_parameter('z_band').get_parameter_value().double_array_value
        self.dist_thresh = self.get_parameter('dist_thresh').get_parameter_value().double_array_value
        self.timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        self.cmd_vel_limit = self.get_parameter('cmd_vel_limit').get_parameter_value().double_value
        side_coverage_deg = self.get_parameter('side_coverage_deg').get_parameter_value().double_value
        self.side_coverage = side_coverage_deg * np.pi / 180.0

        self.latest_hit_mask = None
        self._cmd_vel = Twist()
        self._damped_cmd_vel = None
        self._damping_gain = None
        self.latest_sector_risks = None
        self.coverage_pub = self.create_publisher(Marker, "/coverage_markers", 1)
        self.sector_pub = self.create_publisher(MarkerArray, "/sector_markers", 1)
        self.damper_publisher_ = self.create_publisher(Float64MultiArray, '/damper_gains', 10)
        self.damped_cmd_vel_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self._damper_core = collision_damper_core(
            z_band=self.z_band,
            n_sectors=self.n_sectors,
            coverage_radius=self.coverage_radius,
        )
        self.add_on_set_parameters_callback(self._parameter_callback)
        self.get_logger().info("Node configured. Ready to be activated.")
        return super().on_configure(state)
    
    def on_activate(self, state : State) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate() is called.")
        self.coverage_timer = self.create_timer(self.timer_period, self._coverage_viz_tick)
        
        pcl_topic_name = self.get_parameter('pc_topic_name').get_parameter_value().string_value
        twist_topic_name = self.get_parameter('twist_topic_name').get_parameter_value().string_value
        
        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        sensor_qos.history = QoSHistoryPolicy.KEEP_LAST
        
        self._pointcloud_subscriber = self.create_subscription(
            PointCloud2,
            pcl_topic_name,
            self._pointcloud_callback,
            sensor_qos)
        
        self._cmd_vel_subscriber = self.create_subscription(
            Twist,
            twist_topic_name,
            self._cmd_vel_callback,
            10
        )
        self.get_logger().info("Node activated. Subscriptions and timers are now live.")
        return super().on_activate(state)
    
    def on_deactivate(self, state):
        self.get_logger().info("on_deactivate() is called.")
        
        # Clean up subcribers, which take the most of the resources
        if hasattr(self, 'coverage_timer'):
            self.coverage_timer.cancel()
        if hasattr(self, '_pointcloud_subscriber'):
            self.destroy_subscription(self._pointcloud_subscriber)
        if hasattr(self, '_cmd_vel_subscriber'):
            self.destroy_subscription(self._cmd_vel_subscriber)
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state):
        self.get_logger().info("on_cleanup() is called.")
        
        # Clean up publisher 
        if hasattr(self, 'coverage_pub'):
            self.destroy_publisher(self.coverage_pub)
        if hasattr(self, 'sector_pub'):
            self.destroy_publisher(self.sector_pub)
        if hasattr(self, 'damper_publisher_'):
            self.destroy_publisher(self.damper_publisher_)
        if hasattr(self, 'damped_cmd_vel_publisher_'):
            self.destroy_publisher(self.damped_cmd_vel_publisher_)

        self._damper_core = None
        return super().on_cleanup(state)
    
    def on_error(self, state : State) -> TransitionCallbackReturn:
        self.get_logger().info("on_error() is called.")
        return super().on_error(state)
    
    def on_shutdown(self, state : State) -> TransitionCallbackReturn:
        
        self.get_logger().info("on_shutdown() is called.")
            
        # Final clean up on resource-intense object
        if hasattr(self, 'coverage_timer'):
            try: self.coverage_timer.cancel()
            except Exception: pass
        for attr in ('_pointcloud_subscriber', '_cmd_vel_subscriber'):
            if hasattr(self, attr):
                try: self.destroy_subscription(getattr(self, attr))
                except Exception: pass
        for attr in ('coverage_pub', 'sector_pub', 'damper_publisher_', 'damped_cmd_vel_publisher_'):
            if hasattr(self, attr):
                try: self.destroy_publisher(getattr(self, attr))
                except Exception: pass
        self._damper_core = None

        return TransitionCallbackReturn.SUCCESS

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
    
    def _parameter_callback(self, params : List[rclpy.parameter.Parameter]) -> SetParametersResult:
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

    def _pointcloud_callback(self, msg : PointCloud2):
         
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

        if self._damping_gain is not None:
            self.damper_publisher_.publish(Float64MultiArray(data=np.asarray(self._damping_gain, dtype=float).tolist()))
        if self._damped_cmd_vel is not None:
            self.damped_cmd_vel_publisher_.publish(self._damped_cmd_vel)

        # self.damper_publisher_.publish(Float64MultiArray(data=self._damping_gain.tolist()))
        # self.damped_cmd_vel_publisher_.publish(self._damped_cmd_vel)

    def _cmd_vel_callback(self, msg: Twist):
        self._cmd_vel = msg
        
def request_transition(node: CollisionDamper, client: rclpy.client.Client, transition_id: int) -> bool:
    req = ChangeState.Request()
    req.transition.id = transition_id
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut)
    result = fut.result()
    return bool(result and result.success)

def main(args=None):
    rclpy.init(args=args)
    node = CollisionDamper()

    # Ensure the node is spinning so its services are available
    rclpy.spin_once(node, timeout_sec=0.1)

    # Perform transitions in main(), based on the param read in __init__
    if node.autostart:
        client = node.create_client(ChangeState, '~/change_state')
        if not client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("change_state service not available; autostart aborted.")
        else:
            ok = request_transition(node, client, Transition.TRANSITION_CONFIGURE)
            if not ok:
                node.get_logger().error("CONFIGURE failed; staying UNCONFIGURED.")
            else:
                ok = request_transition(node, client, Transition.TRANSITION_ACTIVATE)
                if not ok:
                    node.get_logger().error("ACTIVATE failed; node remains INACTIVE.")
                else:
                    node.get_logger().info("Autostart complete: node is ACTIVE.")

    # Keep the node alive
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
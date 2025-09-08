from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file to start the simple_collision_damper node with custom parameters.
    """
    return LaunchDescription([
        Node(
            package='avoidance_damper_pkg',
            executable='simple_collision_damper',
            name='simple_collision_damper_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'twist_topic_name': '/teleop/cmd_vel'},
                {'coverage_radius': 1.0},
                {'side_coverage_deg': 200.0},
                {'cmd_vel_limit': 0.8},
            ]
        )
    ])
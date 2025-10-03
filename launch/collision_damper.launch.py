from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file to start the simple_collision_damper node with custom parameters.
    """
    
    # include teleop joy launcher with joy_vel got remapped to joy_vel if tested standalone

    joy_vel_remap = {
        '/joy_vel': '/teleop/cmd_vel'
    }
    joy_vel_node = Node(
        package='teleop_twist_joy',
        executable='teleop_twist_joy_node',
        name='joy_vel',
        remappings=joy_vel_remap.items()
    )

    return LaunchDescription([
        Node(
            package='avoidance_damper_pkg',
            executable='collision_damper_lifecycle_node',
            name='collision_damper_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'twist_topic_name': '/teleop/cmd_vel'},
                {'coverage_radius': 1.0},
                {'side_coverage_deg': 200.0},
                {'cmd_vel_limit': 0.8},
            ]
        ),
    ])
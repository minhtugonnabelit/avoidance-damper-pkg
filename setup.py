from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'avoidance_damper_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Only need to list package.xml once
        ('share/' + package_name, ['package.xml']),
        # This is a more robust way to install all launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Anh Minh Tu',
    maintainer_email='tu.anhminh1203@gmail.com',
    description='Collision avoidance damper package with Python and C++ support',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_collision_damper = avoidance_damper_pkg.simple_collision_damper:main',
            'collision_damper_lifecycle_node = avoidance_damper_pkg.collision_damper_lifecycle_node:main',
        ],
    },
)
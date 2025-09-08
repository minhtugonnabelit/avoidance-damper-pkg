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
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/collision_damper.launch.py']),
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
        ],
    },
)
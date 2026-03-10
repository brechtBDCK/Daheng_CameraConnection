from fanuc_rmi import RobotClient
from take_image import take_single_image
import random

robot = RobotClient(host="192.168.1.22", startup_port=16001, main_port=16002)
robot.connect()
robot.initialize(uframe=0, utool=1)

# set speed override (controller-specific range)
robot.speed_override(20)
# cartesian = robot.read_cartesian_coordinates()
center_position = {"X": 442.902, "Y": -563.66, "Z": 563.572, "W": -113.518, "P": -88.871, "R": 15.216} 

relative_displacement_xyz = 50
relative_displacement_wpr = 10
num_random_positions = 10

# Assumes robot starts at center position before this script runs.
previous_target_position = center_position.copy()

for i in range(num_random_positions):
    target_offset = {
        "X": random.uniform(-relative_displacement_xyz, relative_displacement_xyz),
        "Y": random.uniform(-relative_displacement_xyz, relative_displacement_xyz),
        "Z": random.uniform(-relative_displacement_xyz, relative_displacement_xyz),
        "W": random.uniform(-relative_displacement_wpr, relative_displacement_wpr),
        "P": random.uniform(-relative_displacement_wpr, relative_displacement_wpr),
        "R": random.uniform(-relative_displacement_wpr, relative_displacement_wpr),
    }

    target_position = {
        axis: center_position[axis] + target_offset[axis]
        for axis in center_position
    }

    # Move from previous sample point to the new target offset around center.
    step_offset = {
        axis: target_position[axis] - previous_target_position[axis]
        for axis in center_position
    }

    robot.linear_relative(step_offset, speed=250, sequence_id=i + 1, uframe=0, utool=1) #speed is in mm/sec here
    take_single_image(exposure_time_us=60000)
    robot.read_cartesian_coordinates() 
    robot.read_joint_coordinates()

    previous_target_position = target_position

# Return to center at end.
robot.linear_relative(
    {
        axis: center_position[axis] - previous_target_position[axis]
        for axis in center_position
    },
    speed=250,
    sequence_id=num_random_positions + 1,
    uframe=0,
    utool=1,
)

robot.close()

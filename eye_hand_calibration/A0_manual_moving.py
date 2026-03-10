from fanuc_rmi import RobotClient
from take_image import take_single_image
import random

robot = RobotClient(host="192.168.1.22", startup_port=16001, main_port=16002)
robot.connect()
robot.initialize(uframe=0, utool=1)


take_single_image(exposure_time_us=60000)
robot.read_cartesian_coordinates() 
robot.read_joint_coordinates()

robot.close()

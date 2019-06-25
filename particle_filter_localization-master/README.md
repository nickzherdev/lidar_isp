## Task 1
Run ROS and rviz
```
roscore 
```
```
rosrun rviz rviz
```
## Task 2
Run bagfile
```
roslaunch particle_filter_localization pf.launch
```
## Task 3
Convert information from odom topic to tf ump -> dump_odom. Visualize scan
## Task 4
Detect beacons from scan and visualize them
```
beacons_detection
```
Use function for publishing beacons
```
publish_beacons
```
## Task 5
Visualize particles
## Task 6
Write motion sampling function for evaluation of particles
## Task 7
Write founction for measuremnt model with known correspondence
## Task 8
Write particle filter node
## Task 9
Use motion sampling and measurement model write Monte-Carlo localization
## Task 10
Visualize result

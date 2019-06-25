#!/usr/bin/env python
import rospy
import tf2_ros
from nav_msgs.msg import Odometry
import pf_utils


class Odom2tf(object):
    def __init__(self):
        rospy.init_node('odom2tf', anonymous=True)

        # objects for listening frame transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.br = tf2_ros.TransformBroadcaster()

        rospy.Subscriber("/odom", Odometry, self.odom_callback)

    def odom_callback(self, data):
        point = pf_utils.cvt_ros_pose2point(data.pose.pose)
        pf_utils.publish_transform(self.br, point, "ump_odom", "ump")


if __name__ == "__main__":
    try:
        node = Odom2tf()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

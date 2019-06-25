#!/usr/bin/env python
import rospy
import tf2_ros
import pf_utils
import numpy as np
import scipy.optimize
from visualization_msgs.msg import MarkerArray, Marker, InteractiveMarker
from geometry_msgs.msg import TransformStamped


class Triangulation(object):
    @staticmethod
    def find_position_triangulation(landmarks, beacons, init_point):
        landmarks1 = pf_utils.cvt_local2global(landmarks, init_point)[:, np.newaxis, :]
        print(landmarks1.shape)
        dist_from_beacon = np.linalg.norm(beacons[np.newaxis, :, :] -
                                          landmarks1,
                                          axis=2)
        ind_closest_beacon = np.argmin(dist_from_beacon, axis=1)
        closest_beacons = beacons[ind_closest_beacon]

        def fun(X):
            points = pf_utils.cvt_local2global(landmarks, X)
            r = np.sum((closest_beacons - points) ** 2, axis=1) ** 0.5
            return r
        res = scipy.optimize.least_squares(fun, init_point)
        return np.array(res.x)


class Localization(object):
    pass


class GlobalLocalizationNode(object):
    # noinspection PyTypeChecker
    def __init__(self):
        rospy.init_node('global_localization_node', anonymous=True)

        self.local_beacons = None
        self.global_beacons = self.load_beacons(rospy.get_param("~global_beacons_file"))
        self.map2world_point = None

        self.start_point = np.array(map(float, rospy.get_param("~start_point", "2 4 -1.57").split()))
        self.robot_frame = rospy.get_param("~robot_frame", "ump")
        self.odom_frame = rospy.get_param("~odom_frame", "ump_odom")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.rate = float(rospy.get_param("~rate", 20))
        self.beacon_radius = float(rospy.get_param("~beacon_radius", 0.05))

        # objects for listening frame transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.br = tf2_ros.TransformBroadcaster()

        self.global_beacons_publisher = rospy.Publisher("/global_beacons", MarkerArray, queue_size=10)
        rospy.Subscriber("beacons", MarkerArray, self.beacons_callback)
        rospy.Timer(rospy.Duration(1. / self.rate), self.timer_callback)
        self.init_position()

    @staticmethod
    def load_beacons(beacons_file):
        return np.loadtxt(beacons_file)

    def beacons_callback(self, data):
        beacons = []
        for beacon in data.markers:
            beacons.append(pf_utils.cvt_ros_pose2point(beacon.pose))
        self.local_beacons = np.array(beacons)

    def init_position(self):
        while self.local_beacons is None:
            rospy.sleep(0.1)
        while len(self.local_beacons) < 2:
            rospy.sleep(0.1)
        # robot2world_point = Triangulation.find_position_triangulation(self.local_beacons[:, :2].copy(), self.global_beacons,
        #                                                               self.start_point)
        robot2world_point = self.start_point
        robot2map_point = pf_utils.get_transform(self.tf_buffer, self.robot_frame, self.odom_frame,
                                                          rospy.Time(0))
        self.map2world_point = pf_utils.find_src(robot2world_point, robot2map_point)

    # noinspection PyUnusedLocal
    def timer_callback(self, event):
        self.publish_beacons(self.global_beacons)
        if self.map2world_point is None:
            return
        self.publish_transform(self.map2world_point)

    def publish_transform(self, point):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform = pf_utils.cvt_point2ros_transform(point)
        self.br.sendTransform(t)

    def publish_beacons(self, beacons):
        markers = []
        for i, beacon in enumerate(beacons):
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = self.map_frame
            marker.ns = "global_beacons"
            marker.id = i
            marker.type = 3
            marker.pose.position.x = beacon[0]
            marker.pose.position.y = beacon[1]
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1
            marker.scale.x = 3 * self.beacon_radius
            marker.scale.y = 3 * self.beacon_radius
            marker.scale.z = 0.2
            marker.color.a = 1
            marker.color.b = 1
            markers.append(marker)
        self.global_beacons_publisher.publish(markers)


if __name__ == "__main__":
    try:
        node = GlobalLocalizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

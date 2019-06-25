#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
# from frames_cvt import cvt_local2global, cvt_global2local, find_src
from pf_utils import cvt_ros_transform2point, cvt_local2global
from visualization_msgs.msg import MarkerArray, Marker
import scipy.optimize
#from cartographer_ros_msgs.msg import LandmarkEntry, LandmarkList
import tf2_ros


class BeaconsDetectionNode(object):
    def __init__(self):
        # ROS initialization
        rospy.init_node('beacons_detection_node', anonymous=True)

        self.min_range = rospy.get_param("~min_range")
        self.beacon_range = rospy.get_param("~beacon_range")
        self.beacon_radius = rospy.get_param("~beacon_radius")
        self.min_points_per_beacon = rospy.get_param("~min_beacon_points")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.beacons_search_range = rospy.get_param("~beacons_search_range")
        self.translation_weight = float(rospy.get_param("~translation_weight"))
        self.min_intens = float(rospy.get_param("~min_intens", "3000"))

        self.global_beacons = None
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        rospy.Subscriber("scan", LaserScan, self.scan_callback, queue_size=1)
        self.beacons_publisher = rospy.Publisher("beacons", MarkerArray, queue_size=2)
        # self.landmarks_publisher = rospy.Publisher("landmark", LandmarkList, queue_size=2)
        rospy.loginfo("Beacon detection node is init")

    def scan_callback(self, scan):
        points = self.point_cloud_from_scan(scan.ranges, scan.angle_min, scan.angle_increment, scan.angle_max)
        points = self.filter_by_min_range_and_intens(points, self.min_range, np.array(scan.intensities), self.min_intens)
        beacons = self.beacons_detection(points)
        if not beacons.shape[0]:
            return
        # self.publish_landmarks(beacons, scan.header)

        self.publish_beacons(beacons, scan.header)

    def beacons_detection(self, points):
        points_number = points.shape[0]
        marked_points = [False] * points_number
        beacons = []
        for i in range(points_number):
            if not marked_points[i]:
                nearest_points = np.linalg.norm(points - points[i], axis=1) < self.beacon_range
                marked_points = nearest_points | marked_points
                rospy.logdebug("num beacons points %d" % int(np.count_nonzero(nearest_points)))
                if np.count_nonzero(nearest_points) >= self.min_points_per_beacon:
                    beacons.append(self.find_beacon(points[nearest_points], self.beacon_radius))
                    rospy.logdebug("find beacon at point (%.3f, %.3f)" % (beacons[-1][0], beacons[-1][1]))
        return np.array(beacons)

    @staticmethod
    def find_beacon(points, beacon_radius):
        def fun(x):
            scs = np.sum((points - x) * (-x), axis=1) / np.linalg.norm(x)
            return np.abs(np.linalg.norm(points - x, axis=1) - beacon_radius) - np.where(scs < 0, scs, 0)
        res = scipy.optimize.least_squares(fun, points[0])
        return np.array(res.x)

    @staticmethod
    def point_cloud_from_scan(ranges, angle_min, angle_increment, angle_max):
        angles = np.linspace(angle_min, angle_max, int((angle_max - angle_min) / angle_increment) + 1)
        ranges = np.array(ranges)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.array([x, y]).T
        return points

    @staticmethod
    def filter_by_min_range_and_intens(points, min_range, intensities, min_initens):
        return points[(np.linalg.norm(points, axis=1) >= min_range) & (intensities > min_initens)]

    def publish_beacons(self, beacons, header):
        markers = []
        for i, beacon in enumerate(beacons):
            marker = Marker()
            marker.header = header
            marker.ns = "beacons"
            marker.id = i
            marker.type = 3
            marker.pose.position.x = beacon[0]
            marker.pose.position.y = beacon[1]
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1
            marker.scale.x = 3 * self.beacon_radius
            marker.scale.y = 3 * self.beacon_radius
            marker.scale.z = 0.2
            marker.color.a = 0.5
            marker.color.g = 1
            marker.lifetime = rospy.Duration(0.1)
            markers.append(marker)

        self.beacons_publisher.publish(markers)

    @staticmethod
    def get_tf_coords(buffer_, child_frame, parent_frame, stamp):
        try:
            t = buffer_.lookup_transform(parent_frame, child_frame, stamp)
            return cvt_ros_transform2point(t.transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as msg:
            rospy.logwarn(str(msg))
            return None


if __name__ == '__main__':
    try:
        node = BeaconsDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

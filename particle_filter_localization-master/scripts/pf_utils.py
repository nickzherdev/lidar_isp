import numpy as np
import scipy.optimize
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Pose, Transform
import tf_conversions
# import tf2_ros
import rospy


# noinspection PyPep8Naming
def cvt_local2global(local_point, src_point):
    """
    Convert points from local frame to global
    :param local_point: A local point or array of local points that must be converted 1-D np.array or 2-D np.array
    :param src_point: A
    :return:
    """
    size = local_point.shape[-1]
    x, y, a = 0, 0, 0
    if size == 3:
        x, y, a = local_point.T
    elif size == 2:
        x, y = local_point.T
    # noinspection PyPep8Naming
    X, Y, A = src_point.T
    x1 = x * np.cos(A) - y * np.sin(A) + X
    y1 = x * np.sin(A) + y * np.cos(A) + Y
    a1 = (a + A + np.pi) % (2 * np.pi) - np.pi
    if size == 3:
        return np.array([x1, y1, a1]).T
    elif size == 2:
        return np.array([x1, y1]).T
    else:
        return


def cvt_global2local(global_point, src_point):
    size = global_point.shape[-1]
    x1, y1, a1 = 0, 0, 0
    if size == 3:
        x1, y1, a1 = global_point.T
    elif size == 2:
        x1, y1 = global_point.T
    X, Y, A = src_point.T
    x = x1 * np.cos(A) + y1 * np.sin(A) - X * np.cos(A) - Y * np.sin(A)
    y = -x1 * np.sin(A) + y1 * np.cos(A) + X * np.sin(A) - Y * np.cos(A)
    a = (a1 - A + np.pi) % (2 * np.pi) - np.pi
    if size == 3:
        return np.array([x, y, a]).T
    elif size == 2:
        return np.array([x, y]).T
    else:
        return


def find_src(global_point, local_point):
    x, y, a = local_point.T
    x1, y1, a1 = global_point.T
    A = (a1 - a) % (2 * np.pi)
    X = x1 - x * np.cos(A) + y * np.sin(A)
    Y = y1 - x * np.sin(A) - y * np.cos(A)
    return np.array([X, Y, A]).T


# noinspection PyUnresolvedReferences
def cvt_point2ros_pose(point):
    pose = Pose()
    pose.position.x = point[0]
    pose.position.y = point[1]
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, point[2])
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    return pose


# noinspection PyUnresolvedReferences
def cvt_ros_pose2point(pose):
    x = pose.position.x
    y = pose.position.y
    q = [pose.orientation.x,
         pose.orientation.y,
         pose.orientation.z,
         pose.orientation.w]
    _, _, a = tf_conversions.transformations.euler_from_quaternion(q)
    return np.array([x, y, a])


def cvt_point2ros_transform(point):
    transform = Transform()
    transform.translation.x = point[0]
    transform.translation.y = point[1]
    # noinspection PyUnresolvedReferences
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, point[2])
    transform.rotation.x = q[0]
    transform.rotation.y = q[1]
    transform.rotation.z = q[2]
    transform.rotation.w = q[3]
    return transform


# noinspection PyUnresolvedReferences
def cvt_ros_transform2point(transform):
    x = transform.translation.x
    y = transform.translation.y
    q = [transform.rotation.x,
         transform.rotation.y,
         transform.rotation.z,
         transform.rotation.w]
    _, _, a = tf_conversions.transformations.euler_from_quaternion(q)
    return np.array([x, y, a])


def cvt_ros_scan2points(scan):
    ranges = np.array(scan.ranges)
    ranges[ranges != ranges] = 0
    ranges[ranges == np.inf] = 0
    n = ranges.shape[0]
    angles = np.arange(scan.angle_min, scan.angle_min + n * scan.angle_increment, scan.angle_increment)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.array([x, y]).T


def get_transform(buffer_, child_frame, parent_frame, stamp):
    t = buffer_.lookup_transform(parent_frame, child_frame, stamp)
    return cvt_ros_transform2point(t.transform)


def publish_beacons(self, beacons, header, publisher):
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
        marker.scale.x = 10 * self.beacon_radius
        marker.scale.y = 10 * self.beacon_radius
        marker.scale.z = 0.2
        marker.color.a = 1
        marker.color.g = 1
        markers.append(marker)

    publisher.publish(markers)


def point_cloud_from_scan(ranges, angle_min, angle_increment, angle_max):
    angles = np.linspace(angle_min, angle_max, int((angle_max - angle_min) / angle_increment) + 1)
    ranges = np.array(ranges)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points = np.array([x, y]).T
    return points


def beacons_detection(points, beacon_radius, beacon_range, min_points_per_beacon):
    points_number = points.shape[0]
    marked_points = [False] * points_number
    beacons = []
    for i in range(points_number):
        if not marked_points[i]:
            nearest_points = np.linalg.norm(points - points[i], axis=1) < beacon_range
            marked_points = nearest_points | marked_points
            rospy.logdebug("num beacons points %d" % int(np.count_nonzero(nearest_points)))
            if np.count_nonzero(nearest_points) >= min_points_per_beacon:
                beacons.append(find_beacon(points[nearest_points], beacon_radius))
                rospy.logdebug("find beacon at point (%.3f, %.3f)" % (beacons[-1][0], beacons[-1][1]))
    return np.array(beacons)


# noinspection PyUnresolvedReferences
def find_beacon(points, beacon_radius):
    def fun(x):
        scs = np.sum((points - x) * (-x), axis=1) / np.linalg.norm(x)
        return np.abs(np.linalg.norm(points - x, axis=1) - beacon_radius) - np.where(scs < 0, scs, 0)
    res = scipy.optimize.least_squares(fun, points[0])
    return np.array(res.x)


def publish_transform(br, point, parent_frame, child_frame):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.transform = cvt_point2ros_transform(point)
    br.sendTransform(t)


def resample(weights, n=1000):
    indices_buf = []
    weights = np.array(weights)
    c = weights[0]
    j = 0
    m = n
    M = 1. / m
    r = np.random.uniform(0, M)
    for i in range(m):
        u = r + i * M
        while u > c:
            j += 1
            c = c + weights[j]
        indices_buf.append(j)
    return indices_buf

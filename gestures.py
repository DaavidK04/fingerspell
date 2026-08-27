import math

from landmarks import INDEX_FINGER_MCP, PINKY_MCP, THUMB_TIP, WRIST

THUMB_SPREAD_THRESHOLD = 0.73


def is_stretched(points, finger):
    _mcp, pip, _dip, tip = finger
    return math.dist(points[tip], points[WRIST]) > math.dist(points[pip], points[WRIST])


def thumb_ratio(points):
    wrist_to_pinkymcp_scale = math.dist(points[WRIST], points[PINKY_MCP])
    thumb_measure = math.dist(points[THUMB_TIP], points[INDEX_FINGER_MCP])
    ratio = thumb_measure / wrist_to_pinkymcp_scale

    return ratio


def is_thumb_spread(points):
    return thumb_ratio(points) > THUMB_SPREAD_THRESHOLD


def check_thumb_direction(points):
    # hand_vector_y = points[INDEX_FINGER_MCP].y - points[WRIST].y
    # thumb_vector_y = points[THUMB_TIP].y - points[WRIST].y
    is_hand_upright = points[INDEX_FINGER_MCP].y < points[WRIST].y

    if is_hand_upright:
        if points[THUMB_TIP].y < points[INDEX_FINGER_MCP].y:
            return "THUMB UP"
        elif points[THUMB_TIP].y > points[WRIST].y:
            return "THUMB DOWN"
    else:
        if points[THUMB_TIP].y > points[INDEX_FINGER_MCP].y:
            return "THUMB DOWN"
        elif points[THUMB_TIP].y < points[WRIST].y:
            return "THUMB UP"

    return "NEUTRAL OR OTHER"

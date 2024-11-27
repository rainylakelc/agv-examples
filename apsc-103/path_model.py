class LineSegment:
    def __init__(self, start_x, start_y, end_x, end_y) -> None:
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

class ArcSegment:
    def __init__(self, center_x, center_y, radius, start_angle, end_angle) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle

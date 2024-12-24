import math

class LineSegment:
    def __init__(self, start_x, start_y, end_x, end_y) -> None:
        # Set initial conditions:
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.length = math.sqrt((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)
    def position_at_point(self, t, speed):
        # Position of robot at time t: 
        dist = speed * t
        ratio = dist / self.length
        x = self.start_x + ratio * (self.end_x - self.start_x)
        y = self.start_y + ratio * (self.end_y - self.start_y)
        return x, y
    def velocity_at_point(self) -> int: 
        return 1; #[m/s]


class ArcSegment:
    def __init__(self, center_x, center_y, radius, start_angle, end_angle) -> None:
        # Set initial conditions: 
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.length = self.radius * abs(self.end_angle - self.start_angle)
    def position_at_point(self, t, angular_speed):
        # Find robot position at time t: 
        angle = self.start_angle + angular_speed * t
        x = self.center_x + self.radius * math.cos(angle)
        y = self.center_y + self.radius * math.sin(angle)
        return x, y
    def velocity_at_point(self): 
        #angular velocity
        return 0.1 #const 


        
import math


class Segment:
    X: int
    Y: int

    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def goto(self, x, y):
        self.X = x
        self.Y = y


class Snake:
    x: int
    y: int

    def __init__(self, x, y, color):
        self.direction = "stop"
        self.body = list()
        self.color = color
        self.x = x
        self.y = y
        self.reset()

    def move_snake(self):
        if self.direction == 'Up':
            x = self.x
            self.x = x - 1
        if self.direction == 'Right':
            y = self.y
            self.y = y + 1
        if self.direction == 'Down':
            x = self.x
            self.x = x + 1
        if self.direction == 'Left':
            y = self.y
            self.y = y - 1

    def move_body(self, reward_given):
        if len(self.body) > 0 and not reward_given:
            for index in range(len(self.body) - 1, 0, -1):
                x = self.body[index - 1].X
                y = self.body[index - 1].Y
                self.body[index].goto(x, y)

            self.body[0].goto(self.x, self.y)

    def get_distance(self, seg: Segment):
        return math.sqrt((self.x - seg.X) ** 2 + (self.y - seg.Y) ** 2)

    def reset(self):
        body = list()
        body.append(Segment(self.x, self.y))
        self.body = body

    def go_up(self):
        if self.direction != "Down":
            self.direction = "Up"

    def go_down(self):
        if self.direction != "Up":
            self.direction = "Down"

    def go_right(self):
        if self.direction != "Left":
            self.direction = "Right"

    def go_left(self):
        if self.direction != "Right":
            self.direction = "Left"

from snake import Snake, Segment
import gym
import numpy as np
import random
import time


class Field(gym.Env):
    def __init__(self, x, y, human=False):
        super(Field, self).__init__()
        self.size_x = x
        self.size_y = y
        self.action_space = 4
        self.human = human
        self.done = False

        self.base = np.zeros((self.size_x, self.size_y))
        self.snake = Snake(int(self.base.shape[0] / 2), int(self.base.shape[1] / 2), "black")

        self.apple = Segment(*self.random_coordinates())

        self.state = self.get_state()
        self.prev_distance = 1000
        self.distance = self.snake.get_distance(self.apple)

        self.reward = 0
        self.score = 0

    def random_coordinates(self):
        bad_coord = True
        coord_list = list()
        x, y = 0, 0
        for segment in self.snake.body:
            coord_list.append((segment.X, segment.Y))
        while bad_coord:
            x = random.randint(0, self.size_x-1)
            y = random.randint(0, self.size_y-1)
            if not ((x, y) in coord_list):
                bad_coord = False
        return x, y

    def get_state(self):
        base = np.zeros((self.size_x, self.size_y))
        if not self.done:
            base[self.snake.x, self.snake.y] = 3
            for i in self.snake.body[1:]:
                base[i.X, i.Y] = 1
            base[self.apple.X, self.apple.Y] = 2
            self.base = base
        else:
            x = self.snake.body[0].X
            y = self.snake.body[0].Y
            self.base[x, y] = 1

        state = list()

        body_up = []
        body_right = []
        body_down = []
        body_left = []
        if len(self.snake.body) > 3:
            for body in self.snake.body[3:]:
                if self.snake.get_distance(body) == 1:
                    if body.Y < self.snake.y:
                        body_up.append(1)
                    elif body.Y > self.snake.y:
                        body_down.append(1)
                    if body.X < self.snake.x:
                        body_left.append(1)
                    elif body.X > self.snake.x:
                        body_right.append(1)

        if len(body_up) > 0:
            body_up = 1
        else:
            body_up = 0
        if len(body_right) > 0:
            body_right = 1
        else:
            body_right = 0

        if len(body_down) > 0:
            body_down = 1
        else:
            body_down = 0
        if len(body_left) > 0:
            body_left = 1
        else:
            body_left = 0

        traverse_up = []
        traverse_down = []
        traverse_left = []
        traverse_right = []

        if len(self.snake.body) > 3:
            for body in self.snake.body[3:]:
                if body.Y == self.snake.y and body.X < self.snake.x:
                    traverse_left.append(1)
                elif body.Y == self.snake.y and body.X > self.snake.x:
                    traverse_right.append(1)
                if body.X == self.snake.x and body.Y < self.snake.y:
                    traverse_up.append(1)
                elif body.X == self.snake.x and body.Y > self.snake.y:
                    traverse_down.append(1)

        if len(traverse_up) > 0:
            traverse_up = 1
        else:
            traverse_up = 0
        if len(traverse_right) > 0:
            traverse_right = 1
        else:
            traverse_right = 0

        if len(traverse_down) > 0:
            traverse_down = 1
        else:
            traverse_down = 0
        if len(traverse_left) > 0:
            traverse_left = 1
        else:
            traverse_left = 0

        state.append(int(self.snake.y == 0 or body_up == 1))    # препятствие выше
        state.append(int(self.snake.x == 0 or body_left == 1))  # препятствие слева
        state.append(int(self.snake.y == self.size_y - 1 or body_down == 1))  # препятствие ниже
        state.append(int(self.snake.x == self.size_x - 1 or body_right == 1))  # препятствие справа

        state.append(int(self.snake.y > self.apple.Y))  # яблоко левее
        state.append(int(self.snake.x > self.apple.X))  # яблоко выше
        state.append(int(self.snake.y < self.apple.Y))  # яблоко правее
        state.append(int(self.snake.x < self.apple.X))  # яблоко ниже

        state.append(int(traverse_up))  # хвост на траверзе сверху
        state.append(int(traverse_down))  # хвост на траверсе снизу
        state.append(int(traverse_left))  # хвост на траверсе слева
        state.append(int(traverse_right))  # хвост на траверсе справа

        state.append(int(self.snake.direction == "Up"))
        state.append(int(self.snake.direction == "Left"))
        state.append(int(self.snake.direction == "Down"))
        state.append(int(self.snake.direction == "Right"))
        # s = base.flatten().tolist()
        return np.array(state)  # +s

    def reset(self):
        self.snake = Snake(int(self.base.shape[0] / 2),
                           int(self.base.shape[1] / 2),
                           "black")
        apple_x, apple_y = self.random_coordinates()
        self.apple = Segment(apple_x, apple_y)
        self.prev_distance = 1000
        self.done = False
        self.score = 0
        return self.get_state()

    def out_of_borders(self):
        if self.snake.x >= 20 or self.snake.x < 0 or self.snake.y >= 20 or self.snake.y < 0:
            return True

    def body_check_snake(self):
        if len(self.snake.body) > 1:
            for body in self.snake.body[1:]:
                if self.snake.get_distance(body) == 0:
                    return True

    def picked_apple(self):
        if self.snake.get_distance(self.apple) == 0:
            self.snake.body = [self.apple] + self.snake.body
            apple_x, apple_y = self.random_coordinates()
            self.apple = Segment(apple_x, apple_y)
            return True

    def step(self, action):
        if action == 0:
            self.snake.go_up()
        if action == 1:
            self.snake.go_right()
        if action == 2:
            self.snake.go_down()
        if action == 3:
            self.snake.go_left()
        self.run_game()

        state = self.get_state()
        return state, self.reward, self.done, self.score

    def run_game(self):
        reward_given = False
        self.snake.move_snake()

        if self.picked_apple():
            self.reward = 10
            reward_given = True

        if self.out_of_borders() or self.body_check_snake():
            self.reward = -150
            reward_given = True
            self.done = True
            if self.human:
                self.reset()
                print("Game Over!")

        self.snake.move_body(reward_given)

        if not self.done:
            self.prev_distance = self.distance
            self.distance = self.snake.get_distance(self.apple)
        if not reward_given:
            if self.distance < self.prev_distance:
                self.reward = 1
            else:
                self.reward = -5
        if self.human:
            time.sleep(0.05)
            self.base = self.get_state()

        self.score += self.reward

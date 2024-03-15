import random
from game_settings import BLOCK_SIZE
from game_utils import Point


class Obstacles:
    def __init__(self, game) -> None:
        self.obstacles = []
        self.game = game

    def place_random_obstacles(self, obstacles_quantity):
        self.obstacles.clear()
        
        for _ in range(obstacles_quantity):
            is_valid_point = False
            while not is_valid_point:
                x = random.randint(0, (self.game.width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
                y = random.randint(0, (self.game.height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE

                obstacle_point = Point(x, y)
                is_valid_point = True

                for obstacle in self.obstacles:
                    if obstacle_point == obstacle:
                        is_valid_point = False
                        break

            self.obstacles.append(obstacle_point)

    def is_point_at_obstacle(self, point):
        for obstacle in self.obstacles.obstacles:
            if point == obstacle:
                return True
        return False
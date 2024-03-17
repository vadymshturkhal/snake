import random
from game_settings import BLOCK_SIZE
from game_utils import Point, calculate_distance, normalize_distance


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
        for obstacle in self.obstacles:
            if point == obstacle:
                return True
        return False

    def get_distance_to_closest_obstacle(self, point):
        min_distance = float('inf')  # Initialize with infinity

        for obstacle in self.obstacles:
            distance = calculate_distance(point, obstacle)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_distance_to_all_obstacles(self, point):
        distances = []
        for obstacle in self.obstacles:
            distance = calculate_distance(point, obstacle)
            distances.append(distance)
        return distances

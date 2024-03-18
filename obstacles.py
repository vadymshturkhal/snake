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

    def place_obstacle_at_point(self, x, y):
        """Place an obstacle at a specified point."""
        obstacle_point = Point(x, y)
        if not self.is_point_at_obstacle(obstacle_point):
            self.obstacles.append(obstacle_point)

    def save_obstacles(self, filename='obstacles.txt'):
        with open(filename, 'w') as f:
            for obstacle in self.obstacles:
                f.write(f'{obstacle.x},{obstacle.y}\n')

    def load_obstacles_from_file(self, filename='obstacles.txt'):
        """Load obstacles from a file and place them in the game."""
        with open(filename, 'r') as f:
            for line in f:
                x, y = line.strip().split(',')
                self.place_obstacle_at_point(int(x), int(y))

import random
from game_settings import BLOCK_SIZE
from game_utils import Point, calculate_distance, normalize_distance


class Obstacles:
    def __init__(self, game) -> None:
        self.game = game
        self.obstacles = []

    def place_random_obstacles(self, obstacles_quantity: int):
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

    def is_point_at_obstacle(self, point: Point) -> bool:
        for obstacle in self.obstacles:
            if point == obstacle:
                return True
        return False

    def get_distance_to_closest_obstacle(self, point: Point):
        min_distance = float('inf')  # Initialize with infinity

        for obstacle in self.obstacles:
            distance = calculate_distance(point, obstacle)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_distance_to_all_obstacles(self, point: Point):
        distances = []
        for obstacle in self.obstacles:
            distance = calculate_distance(point, obstacle)
            distances.append(distance)
        return distances

    def place_obstacle_at_point(self, point: Point):
        """Place an obstacle at a specified point."""
        if not self.is_point_at_obstacle(point):
            self.obstacles.append(point)

    def remove_obstacle_at_point(self, point_to_remove: Point):
        """Remove an obstacle at a specified point, if present."""
        # Create a Point object for the x, y coordinates
        # Use a list comprehension to filter out the obstacle at the clicked point
        self.obstacles = [obstacle for obstacle in self.obstacles if obstacle != point_to_remove]

    def save_obstacles(self, filename='obstacles.txt'):
        with open(filename, 'w') as f:
            for obstacle in self.obstacles:
                f.write(f'{obstacle.x},{obstacle.y}\n')

    def load_obstacles_from_file(self, filename='obstacles.txt'):
        """Load obstacles from a file and place them in the game."""
        try:
            with open(filename, 'r') as f:
                for line in f:
                    x, y = line.strip().split(',')
                    self.place_obstacle_at_point(int(x), int(y))
        except FileNotFoundError:
            # If the file doesn't exist, create it by opening it in write mode and then closing it.
            # This is useful if you want to ensure the file exists for future operations.
            open(filename, 'w').close()

    def __iter__(self):
        """Make the Obstacles class iterable over its obstacle items."""
        # This method returns an iterator for the container. 
        # In this case, we can simply yield from self.obstacles since lists are already iterable.
        for obstacle in self.obstacles:
            yield obstacle

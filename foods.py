import random
from game_settings import BLOCK_SIZE, FOOD_QUANTITY
from game_utils import Point, calculate_distance


class Foods:
    def __init__(self, game):
        self.game = game
        self.foods = []

    @property
    def is_empty(self):
        return len(self.foods) == 0

    def place_food(self, random_place=True):
        if random_place:
            for _ in range(FOOD_QUANTITY):
                is_valid_point = False
                while not is_valid_point:
                    x = random.randint(0, (self.game.width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
                    y = random.randint(0, (self.game.height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE

                    food_point = Point(x, y)
                    is_valid_point = True

                    if food_point == self.game.snake.head:
                        is_valid_point = False

                    if self.game.obstacles.is_point_at_obstacle(food_point):
                        is_valid_point = False
                self.foods.append(food_point)
        else:
            # Load from file
            pass

    def get_closest_food(self, point):
        min_distance = float('inf')  # Initialize with infinity
        closest_food = None
        for food in self.foods:
            distance = calculate_distance(point, food)
            if distance < min_distance:
                min_distance = distance
                closest_food = food
        return closest_food

    def place_food_at_point(self, point):
        """Place a food item at the specified point."""
        if point not in self.foods:  # Avoid duplicate food points
            self.foods.append(point)

    def remove_food_at_point(self, food_point):
        """Remove a food item at the specified point, if present."""
        if food_point in self.foods:
            self.foods.remove(food_point)

    def __iter__(self):
        """Make the Foods class iterable over its food items."""
        # This method returns an iterator for the container. In this case, we can
        # simply yield from self.foods since lists are already iterable.
        for food in self.foods:
            yield food

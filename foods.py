import ast
import csv
import random
from game_settings import BLOCK_SIZE, FOOD_QUANTITY
from game_utils import Point, calculate_distance


class Foods:
    def __init__(self, game, load_from_filename=None):
        self.game = game
        self.foods = []
        self._load_from_filename = load_from_filename

    @property
    def is_empty(self):
        return len(self.foods) == 0

    def is_food_at_point(self, point):
        return point in self.foods

    def clear(self):
        self.foods.clear()

    def place_food(self):
        self.foods.clear()

        if self._load_from_filename is not None:
            self.load_foods_from_file()

        if len(self.foods) < FOOD_QUANTITY:
            self._place_food_at_random_place(FOOD_QUANTITY - len(self.foods))
        
        if len(self.foods) <= 0:
            raise Exception("Can't place random food with FOOD_QUANTITY <= 0")

    def _place_food_at_random_place(self, quantity: int):
        """
        A helper method used to place a specified number of food items at random
        locations within the game environment. Ensures food items do not overlap
        with the snake's head or obstacles.

        Parameters:
        - quantity (int): The number of food items to place randomly in the
            game environment.

        Details:
        This method attempts to find suitable locations for placing the specified
        number of food items by randomly selecting positions within the bounds of
        the game environment. A position is considered suitable if it does not
        coincide with the snake's head or any existing obstacles.

        For each food item, the method:
        1. Randomly generates x and y coordinates within the game boundaries,
            aligning them with the BLOCK_SIZE grid.
        2. Checks if the generated position coincides with the snake's head or
            any obstacles.
        3. If the position is suitable, adds the food item to the `foods` list.
            If not, repeats the process.

        Note: This method directly modifies the `foods` list attribute of the
        class by appending new Point instances representing the positions of the
        newly placed food items.
        """

        for _ in range(quantity):
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

    def get_closest_food(self, point: Point) -> Point:
        min_distance = float('inf')  # Initialize with infinity
        closest_food = None
        for food in self.foods:
            distance = calculate_distance(point, food)
            if distance < min_distance:
                min_distance = distance
                closest_food = food
        return closest_food

    def place_food_at_point(self, point: Point):
        """Place a food item at the specified point."""
        if point not in self.foods:  # Avoid duplicate food points
            self.foods.append(point)

    def remove_food_at_point(self, food_point: Point):
        """Remove a food item at the specified point, if present."""
        if food_point in self.foods:
            self.foods.remove(food_point)

    def save_foods(self):
        with open(self._load_from_filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Foods'])
            for food in self.foods:
                # Write the coordinates as a single string in the format '(x, y)'
                position = f'({food.x}, {food.y})'
                writer.writerow([position])

    def load_foods_from_file(self):
        """Load obstacles from a file and place them in the game."""
        try:
            with open(self._load_from_filename, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Parse the position string to extract x and y
                    position = ast.literal_eval(row['Foods'])
                    food_point = Point(*position)
                    self.foods.append(food_point)
                    self.place_food_at_point(food_point)
        except FileNotFoundError:
            # If the file doesn't exist, create it by opening it in write mode and then closing it.
            # This is useful if you want to ensure the file exists for future operations.
            self.save_foods()

    def __iter__(self):
        """Make the Foods class iterable over its food items."""
        # This method returns an iterator for the container. In this case, we can
        # simply yield from self.foods since lists are already iterable.
        for food in self.foods:
            yield food

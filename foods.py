from game_utils import Point, calculate_distance


class Foods:
    def __init__(self, game):
        self.game = game
        self.foods = []

    def get_closest_food(self, point):
        min_distance = float('inf')  # Initialize with infinity
        closest_food = None
        for food in self.foods:
            distance = calculate_distance(point, food)
            if distance < min_distance:
                min_distance = distance
                closest_food = food
        return closest_food

    def place_food_at_point(self, x, y):
        """Place a food item at the specified point."""
        food_point = Point(x, y)
        if food_point not in self.food:  # Avoid duplicate food points
            self.food.append(food_point)

    def remove_food_at_point(self, x, y):
        """Remove a food item at the specified point, if present."""
        food_point = Point(x, y)
        if food_point in self.food:
            self.food.remove(food_point)
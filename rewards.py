from game_settings import IS_ADD_OBSTACLES, REWARD_CRAWLING, REWARD_ROTATION
from game_settings import REWARD_WIN, REWARD_LOOSE


class Rewards:
    def __init__(self, game):
        self.game = game

    def get_snake_reward(self, action):
        # Goal reached
        if self.game.is_eaten():
            return REWARD_WIN

        # Crashed
        if self.game.snake_is_crashed:
            return REWARD_LOOSE

        if action == [0, 0, 1]:
            snake_reward = REWARD_CRAWLING
        else:
            snake_reward = REWARD_ROTATION

        # if IS_ADD_OBSTACLES:
        #     distance_to_all_obstacles = self.game.obstacles.get_distance_to_all_obstacles(self.game.snake.head)

        #     if min(distance_to_all_obstacles) <= 3:
        #         snake_reward -= 1/min(distance_to_all_obstacles) * 0.1

        return snake_reward

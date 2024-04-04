from agents.qlearning import QLearning
from game import SnakeGameAI
from game_settings import IS_ADD_OBSTACLES, MAPS_FOLDER, SNAKE_WEIGHTS_FILENAME, SCORE_DATA_FILENAME
from game_settings import GAME_SPEED, SNAKE_SPEED, FOOD_SPEED_MULTIPLIER, FRAME_RESTRICTION
import time
from rewards import Rewards
from game_utils import Timer


class TrainAgent:
    def __init__(self):
        self.game = SnakeGameAI(is_rendering, game_speed, IS_ADD_OBSTACLES, foods_to_load, is_place_food=True)
        self.snake_agent = QLearning(*[is_load_weights_snake, SNAKE_WEIGHTS_FILENAME, games_to_play, is_load_n_games])
        self.rewards = Rewards(self.game)

    def assure_data_csv(self):
        if is_load_weights_snake:
            return

        with open(SCORE_DATA_FILENAME, 'w') as file:
            file.write('Score, Time, Reward, Epsilon, Bumps, Steps, Rotations\n')

    def scores_to_csv(self, score, game_duration, snake_reward, snake_epsilon, bumps, steps, rotations):
        with open(SCORE_DATA_FILENAME, 'a') as file:
            file.write(f'{str(score)}, {game_duration:.4f}, {snake_reward:.4f}, {snake_epsilon:.4f}, {bumps}, {steps}, {rotations}\n')

    def train(self, obstacles_to_load=None):
        self.assure_data_csv()

        self.assure_data_csv()

        timer = Timer()

        max_reward = float('-inf')
        snake_game_reward = 0
        bumps = 0
        steps = 0
        rotations = 0

        timer.start()

        if obstacles_to_load is not None:
            self.game.obstacles.load_obstacles_from_file(obstacles_to_load)

        while self.game.counter <= games_to_play:
            # Snake Agent
            state_old = self.game.get_snake_state()
            snake_action = self.snake_agent.get_action(state_old)
            self.game.snake_apply_action(snake_action)

            snake_reward = self.rewards.get_snake_reward(action=snake_action)
            state_new = self.game.get_snake_state()
            snake_game_reward += snake_reward
            self.snake_agent.last_reward = snake_reward

            # done = self.game.is_eaten_food
            # done = self.game.snake_is_crashed
            done = FRAME_RESTRICTION - self.game.frame_iteration == 0
            self.game.play_step()

            # if self.game.is_eaten_food:
                # self.game.foods.place_food()

            # Train snake
            
            self.snake_agent.train_short_memory(state_old, snake_action, snake_reward, state_new, done)
            self.snake_agent.remember(state_old, snake_action, snake_reward, state_new, done)

            if self.game.snake_is_crashed:
                bumps += 1

            if snake_action == [0, 0, 1]:
                steps += 1
            else:
                rotations += 1

            if done:
                elapsed_time = timer.get_elapsed_time()
                timer.reset()

                self.snake_agent.n_games += 1
                self.snake_agent.train_long_memory()

                # Save snake model
                if snake_game_reward >= max_reward:
                    max_reward = snake_game_reward
                    self.snake_agent.model.save(epoch=self.snake_agent.n_games, filename=SNAKE_WEIGHTS_FILENAME)

                self.scores_to_csv(self.game.score, elapsed_time, snake_game_reward, self.snake_agent.epsilon, bumps, steps, rotations)
                snake_game_reward = 0
                bumps = 0
                steps = 0
                rotations = 0
                self.game.reset()
                timer.start()

is_load_weights_snake = True
is_load_n_games = True
is_rendering = True
game_speed = 40
games_to_play = 120
obstacles_to_load = MAPS_FOLDER + './level_0/obstacles.csv'
foods_to_load = MAPS_FOLDER + './level_0/foods.csv'

if __name__ == '__main__':
    train_agent = TrainAgent()
    train_agent.train(obstacles_to_load)

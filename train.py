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
        self._states = []
        self._actions = []
        self._rewards = []

        self._snake_game_reward = 0
        self._bumps = 0
        self._steps = 0
        self._rotations = 0

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

        if obstacles_to_load is not None:
            self.game.obstacles.load_obstacles_from_file(obstacles_to_load)

        for _ in range(games_to_play):
            self._train_single_game()
            game_loss = self.snake_agent.train_episode(self._states, self._actions, self._rewards)
            self._clear_game_data()

    def _train_single_game(self):
        timer = Timer()
        timer.start()

        max_reward = float('-inf')

        while True:
            # Snake Agent
            snake_state = self.game.get_snake_state()
            snake_action = self.snake_agent.get_action(snake_state)
            self.game.snake_apply_action(snake_action)
            snake_reward = self.rewards.get_snake_reward(action=snake_action)

            self._states.append(snake_state)
            self._actions.append(snake_action)
            self._rewards.append(snake_reward)

            self._snake_game_reward += snake_reward
            self.snake_agent.last_reward = snake_reward

            done = FRAME_RESTRICTION - self.game.frame_iteration == 0
            self.game.play_step()

            if self.game.snake_is_crashed:
                self._bumps += 1

            if snake_action == [0, 0, 1]:
                self._steps += 1
            else:
                self._rotations += 1

            if done:
                elapsed_time = timer.get_elapsed_time()
                timer.reset()

                self.snake_agent.n_games += 1

                # Save snake model
                if self._snake_game_reward >= max_reward:
                    max_reward = self._snake_game_reward
                    self.snake_agent.model.save(epoch=self.snake_agent.n_games, filename=SNAKE_WEIGHTS_FILENAME)

                self.scores_to_csv(self.game.score, elapsed_time, self._snake_game_reward, self.snake_agent.epsilon, self._bumps, self._steps, self._rotations)
                break

    def _clear_game_data(self):
        self._snake_game_reward = 0
        self._bumps = 0
        self._steps = 0
        self._rotations = 0

        self.game.reset()

        self._states.clear()
        self._actions.clear()
        self._rewards.clear()


is_load_weights_snake = False
is_load_n_games = False
is_rendering = False
game_speed = 40
games_to_play = 400
obstacles_to_load = MAPS_FOLDER + './level_0/obstacles.csv'
foods_to_load = MAPS_FOLDER + './level_0/foods.csv'

if __name__ == '__main__':
    train_agent = TrainAgent()
    train_agent.train(obstacles_to_load)

from agent_snake import SnakeAgent
from game import SnakeGameAI
from game_settings import FRAME_RESTRICTION, IS_ADD_OBSTACLES, MAPS_FOLDER, SNAKE_WEIGHTS_FILENAME, SCORE_DATA_FILENAME
from game_settings import GAME_SPEED, SNAKE_SPEED, FOOD_SPEED_MULTIPLIER
import time

from game_utils import Timer
from rewards import Rewards


def assure_data_csv(filename, is_load_weights):
    if is_load_weights:
        return

    with open(filename, 'w') as file:
        file.write('Score, Time, Reward, Epsilon, Bumps\n')

def scores_to_csv(filename, scores, game_duration, snake_reward, snake_epsilon, bumps):
    with open(filename, 'a') as file:
        file.write(f'{str(scores[-1])}, {game_duration:.4f}, {snake_reward:.4f}, {snake_epsilon:.4f}, {bumps}\n')

def train(snake_agent, game: SnakeGameAI, score_data_filename, games_to_play=0, food_agent=None, obstacles_to_load=None, foods_to_load=None):
    scores = []
    mean_scores = []
    total_score = 0

    counter = 0
    last_snake_update = time.time()
    last_food_update = last_snake_update

    rewards = Rewards(game)
    snake_game_reward = 0
    bumps = 0

    timer = Timer()

    if obstacles_to_load is not None:
        game.obstacles.load_obstacles_from_file(MAPS_FOLDER + obstacles_to_load)

    while counter < games_to_play:
        timer.start()
        timer.stop()

        current_time = time.time()
        if current_time - last_snake_update >= SNAKE_SPEED:
            timer.continue_timer()

            last_snake_update = current_time

            state_old = snake_agent.get_state(game)
            snake_action = snake_agent.get_action(state_old, is_train=False)
            game.snake_apply_action(snake_action)

            snake_reward = rewards.get_snake_reward(action=snake_action)
            snake_game_reward += snake_reward

            score = game.score

            game.play_step()

            timer.stop()

            if game.snake_is_crashed:
                bumps += 1

            if game.is_eaten():
            # if game.snake_is_crashed:
                game.reset()
                snake_agent.n_games += 1

                scores.append(score)
                total_score += score
                mean_score = total_score / snake_agent.n_games
                mean_scores.append(mean_score)

                counter += 1
                scores_to_csv(score_data_filename, scores, timer.get_elapsed_time(), snake_game_reward, snake_agent.epsilon, bumps)
                snake_game_reward = 0
                bumps = 0
                timer.reset()

        if current_time - last_food_update >= SNAKE_SPEED * FOOD_SPEED_MULTIPLIER:
            last_food_update = current_time


is_load_weights_snake = True
is_load_weights_food = False
is_load_n_games = True
is_rendering = True
game_speed = 40
games_to_play = 10
obstacles_to_load = './level_2/obstacles.csv'
foods_to_load = MAPS_FOLDER + './level_2/foods.csv'


assure_data_csv(SCORE_DATA_FILENAME, is_load_weights_snake)

snake_agent = SnakeAgent(*[is_load_weights_snake, SNAKE_WEIGHTS_FILENAME, games_to_play, is_load_n_games])
food_agent = None

game = SnakeGameAI(is_rendering, game_speed, IS_ADD_OBSTACLES, foods_to_load)
train(snake_agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent, obstacles_to_load)
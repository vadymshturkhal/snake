from agent_snake import SnakeAgent
from game import SnakeGameAI
from game_settings import IS_ADD_OBSTACLES, MAPS_FOLDER, REWARD_WIN, SNAKE_WEIGHTS_FILENAME, SCORE_DATA_FILENAME
from game_settings import GAME_SPEED, SNAKE_SPEED, FOOD_SPEED_MULTIPLIER, FRAME_RESTRICTION
import time
from rewards import Rewards
from game_utils import Timer


def assure_data_csv(filename, is_load_weights):
    if is_load_weights:
        return

    with open(filename, 'w') as file:
        file.write('Score, Time, Reward, Epsilon, Bumps, Steps, Rotations\n')

def scores_to_csv(filename, scores, game_duration, snake_reward, snake_epsilon, bumps, steps, rotations):
    with open(filename, 'a') as file:
        file.write(f'{str(scores[-1])}, {game_duration:.4f}, {snake_reward:.4f}, {snake_epsilon:.4f}, {bumps}, {steps}, {rotations}\n')

def train(snake_agent, game: SnakeGameAI, score_data_filename, games_to_play=0, food_agent=None, obstacles_to_load=None, foods_to_load=None):
    scores = []

    last_snake_update = time.time()

    rewards = Rewards(game)
    timer = Timer()

    max_reward = float('-inf')
    snake_game_reward = 0
    bumps = 0
    steps = 0
    rotations = 0

    timer.start()

    if obstacles_to_load is not None:
        game.obstacles.load_obstacles_from_file(MAPS_FOLDER + obstacles_to_load)

    while game.counter <= games_to_play:
        current_time = time.time()

        if current_time - last_snake_update >= SNAKE_SPEED:
            last_snake_update = current_time

            # Snake Agent
            state_old = snake_agent.get_state(game)
            snake_action = snake_agent.get_action(state_old)
            game.snake_apply_action(snake_action)

            snake_reward = rewards.get_snake_reward(action=snake_action)
            snake_game_reward += snake_reward

            score = game.score

            game.play_step()

            done = game.is_eaten()

            # Train snake
            state_new = snake_agent.get_state(game)
            snake_agent.train_short_memory(state_old, snake_action, snake_reward, state_new, done)
            snake_agent.remember(state_old, snake_action, snake_reward, state_new, done)

            if game.snake_is_crashed:
                bumps += 1

            if snake_action == [0, 0, 1]:
                steps += 1
            else:
                rotations += 1

            if done:
            # if game.frame_iteration > FRAME_RESTRICTION and any([game.snake_is_crashed, snake_reward == REWARD_WIN]):
            # if game.snake_is_crashed:
            # if score == game.counter // 10 + 1:
            # if score == 4:
                elapsed_time = timer.get_elapsed_time()
                timer.reset()

                game.reset()
                snake_agent.n_games += 1
                snake_agent.train_long_memory()

                if snake_game_reward >= max_reward:
                    max_reward = snake_game_reward
                    snake_agent.model.save(epoch=snake_agent.n_games, filename=SNAKE_WEIGHTS_FILENAME)

                scores.append(score)

                scores_to_csv(score_data_filename, scores, elapsed_time, snake_game_reward, snake_agent.epsilon, bumps, steps, rotations)
                snake_game_reward = 0
                bumps = 0
                steps = 0
                rotations = 0
                timer.start()


is_load_weights_snake = False
is_load_n_games = True
is_rendering = False
game_speed = 40
games_to_play = 100
obstacles_to_load = './level_1/obstacles.csv'
foods_to_load = MAPS_FOLDER + './level_1/foods.csv'

assure_data_csv(SCORE_DATA_FILENAME, is_load_weights_snake)

snake_agent = SnakeAgent(*[is_load_weights_snake, SNAKE_WEIGHTS_FILENAME, games_to_play, is_load_n_games])
food_agent = None

game = SnakeGameAI(is_rendering, game_speed, IS_ADD_OBSTACLES, foods_to_load)
train(snake_agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent, obstacles_to_load)

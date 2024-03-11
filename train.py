from agent_snake import SnakeAgent
from agent_food import FoodAgent
from game import SnakeGameAI
from collections import namedtuple
from game_settings import REWARD_WIN, SNAKE_WEIGHTS_FILENAME, FOOD_WEIGHTS_FILENAME, SCORE_DATA_FILENAME
from game_settings import GAME_SPEED, SNAKE_SPEED, FOOD_SPEED_MULTIPLIER, FRAME_RESTRICTION
import time
from rewards import Rewards
from game_utils import Timer


# Extend the Transition namedtuple with a 'done' field
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def assure_data_csv(filename, is_load_weights):
    if is_load_weights:
        return

    with open(filename, 'w') as file:
        file.write('Score, Time, Reward, Epsilon\n')

def scores_to_csv(filename, scores, game_duration, snake_reward, snake_epsilon):
    with open(filename, 'a') as file:
        file.write(f'{str(scores[-1])}, {game_duration:.4f}, {snake_reward:.4f}, {snake_epsilon}\n')

def train(snake_agent, game, score_data_filename, games_to_play=0, food_agent=None):
    scores = []
    total_score = 0
    record = 0

    last_snake_update = time.time()
    last_food_update = last_snake_update

    rewards = Rewards(game)
    timer = Timer()

    snake_game_reward = 0
    timer.start()

    while game.counter < games_to_play:
        current_time = time.time()

        if current_time - last_snake_update >= SNAKE_SPEED:
            last_snake_update = current_time

            # Snake Agent
            state_old = snake_agent.get_state(game)
            snake_next_move = snake_agent.get_action(state_old)

            game.snake_move(snake_next_move)

            snake_reward = rewards.get_snake_reward()
            snake_game_reward += snake_reward

            score = game.score

            game.play_step()

            # Train snake
            state_new = snake_agent.get_state(game)
            snake_agent.train_short_memory(state_old, snake_next_move, snake_reward, state_new)
            snake_agent.remember(state_old, snake_next_move, snake_reward, state_new)

            if game.frame_iteration > FRAME_RESTRICTION and any([game.snake_is_crashed, snake_reward == REWARD_WIN]):
            # if game.snake_is_crashed:
            # if score == game.counter // 10 + 1:
            # if score == 4:
                elapsed_time = timer.get_elapsed_time()
                timer.reset()

                game.reset()
                snake_agent.n_games += 1
                snake_agent.train_long_memory()

                if score > record:
                    record = score
                    snake_agent.model.save(epoch=agent.n_games, filename=SNAKE_WEIGHTS_FILENAME)

                scores.append(score)
                total_score += score

                scores_to_csv(score_data_filename, scores, elapsed_time, snake_game_reward, snake_agent.epsilon)
                snake_game_reward = 0
                timer.start()
        # if current_time - last_food_update >= SNAKE_SPEED * FOOD_SPEED_MULTIPLIER:
            # last_food_update = current_time
            # Random
            # game.food_move()

        # game.scores_to_csv(score_data_filename, scores)


is_load_weights_snake = False
is_load_weights_food = False
is_rendering = False
game_speed = 20
games_to_play = 150

assure_data_csv(SCORE_DATA_FILENAME, is_load_weights_snake)

agent = SnakeAgent(is_load_weights=is_load_weights_snake, weights_filename=SNAKE_WEIGHTS_FILENAME, epochs=games_to_play)
food_agent = FoodAgent(is_load_weights=is_load_weights_food, weights_filename=FOOD_WEIGHTS_FILENAME, epochs=games_to_play)

game = SnakeGameAI(is_rendering=is_rendering, game_speed=game_speed)
train(agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent)
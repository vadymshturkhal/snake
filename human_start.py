from agent_snake import SnakeAgent
from game import SnakeGameAI
from game_settings import FRAME_RESTRICTION, SNAKE_WEIGHTS_FILENAME, FOOD_WEIGHTS_FILENAME, SCORE_DATA_FILENAME
from game_settings import GAME_SPEED, SNAKE_SPEED, FOOD_SPEED_MULTIPLIER
import time
import pygame

from game_utils import Direction, Timer


def assure_data_csv(filename, is_load_weights):
    if is_load_weights:
        return

    with open(filename, 'w') as file:
        file.write('Score, Time, Reward, Epsilon\n')

def scores_to_csv(filename, scores, game_duration, snake_reward, snake_epsilon):
    with open(filename, 'a') as file:
        file.write(f'{str(scores[-1])}, {game_duration:.4f}, {snake_reward:.4f}, {snake_epsilon:.4f}\n')

def train(snake_agent, game, score_data_filename, games_to_play=0, food_agent=None):
    scores = []
    mean_scores = []
    total_score = 0

    counter = 0
    last_snake_update = time.time()
    last_food_update = last_snake_update

    timer = Timer()
    
    while counter < games_to_play:
        timer.start()
        timer.stop()

        current_time = time.time()
        if current_time - last_snake_update >= SNAKE_SPEED:
            timer.continue_timer()

            last_snake_update = current_time

            state_old = snake_agent.get_state(game)
            # snake_move = snake_agent.get_action(state_old, is_train=False)
            # game.snake_move(snake_move)

            action = [0, 0, 0]

            for event in pygame.event.get():
                # action = [0, 0, 0]  # for continuous

                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = [1, 0, 0]
                    elif event.key == pygame.K_RIGHT:
                        action = [0, 1, 0]
                    elif event.key == pygame.K_UP:
                        action = [0, 0, 1]

            game.snake_apply_action(action, is_human=True)
            score = game.score

            game.is_eaten()
            game.play_step()

            timer.stop()
            if game.snake_is_crashed:
                game.reset()
                snake_agent.n_games += 1

                scores.append(score)
                total_score += score
                mean_score = total_score / snake_agent.n_games
                mean_scores.append(mean_score)

                counter += 1
                timer.reset()

        if current_time - last_food_update >= SNAKE_SPEED * FOOD_SPEED_MULTIPLIER:
            last_food_update = current_time


is_load_weights = True
is_rendering = True
game_speed = 10
games_to_play = 3

assure_data_csv(SCORE_DATA_FILENAME, is_load_weights)

snake_agent = SnakeAgent(is_load_weights=is_load_weights, weights_filename=SNAKE_WEIGHTS_FILENAME)
food_agent = None
game = SnakeGameAI(is_rendering=is_rendering, game_speed=game_speed)
train(snake_agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent)
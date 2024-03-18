import pygame
from game_settings import BLOCK_SIZE
from agent_snake import SnakeAgent
from game import SnakeGameAI
from game_settings import SNAKE_WEIGHTS_FILENAME, SCORE_DATA_FILENAME
from game_settings import SNAKE_SPEED, FOOD_SPEED_MULTIPLIER
import time
import pygame

from game_utils import Timer


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
                    game.obstacles.save_obstacles()
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = [1, 0, 0]
                    elif event.key == pygame.K_RIGHT:
                        action = [0, 1, 0]
                    elif event.key == pygame.K_UP:
                        action = [0, 0, 1]
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Get mouse position and adjust it to your BLOCK_SIZE
                    x, y = event.pos
                    x = (x // BLOCK_SIZE) * BLOCK_SIZE
                    y = (y // BLOCK_SIZE) * BLOCK_SIZE

                    if event.button == 1:  # Left click
                        game.obstacles.place_obstacle_at_point(x, y)
                    elif event.button == 3:  # Right click
                        game.obstacles.remove_obstacle_at_point(x, y)

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


snake_agent = SnakeAgent(is_load_weights=is_load_weights, weights_filename=SNAKE_WEIGHTS_FILENAME)
food_agent = None
game = SnakeGameAI(is_rendering=is_rendering, game_speed=game_speed, is_add_obstacles=True)
train(snake_agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent)

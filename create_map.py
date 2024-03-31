import pygame
from game_settings import BLOCK_SIZE, MAPS_FOLDER
from agent_snake import SnakeAgent
from game import SnakeGameAI
from game_settings import SNAKE_WEIGHTS_FILENAME, SCORE_DATA_FILENAME
from game_settings import SNAKE_SPEED, FOOD_SPEED_MULTIPLIER
import time
import pygame

from game_utils import Point, Timer


def train(snake_agent, game, score_data_filename, games_to_play=0, food_agent=None, obstacles_to_load='new_obstacles.csv', foods_to_load='new_foods.csv'):
    scores = []
    mean_scores = []
    total_score = 0

    counter = 0
    last_snake_update = time.time()
    last_food_update = last_snake_update

    timer = Timer()
    game.obstacles.load_obstacles_from_file(MAPS_FOLDER + obstacles_to_load)
    while counter < games_to_play:
        timer.start()
        timer.stop()

        current_time = time.time()
        if current_time - last_snake_update >= SNAKE_SPEED:
            timer.continue_timer()

            last_snake_update = current_time

            action = [0, 0, 0]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.obstacles.save_obstacles(MAPS_FOLDER + obstacles_to_load)
                    game.foods.save_foods()
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
                    point = Point(x, y)

                    if event.button == 1:  # Left click
                        if not game.obstacles.is_point_at_obstacle(point):
                            game.obstacles.place_obstacle_at_point(point)
                        else:
                            game.obstacles.remove_obstacle_at_point(point)
                    elif event.button == 2:  # Middle click
                        # Toggle placing and deleting food
                        if not game.foods.is_food_at_point(point):
                            game.foods.place_food_at_point(point)
                        else:
                            game.foods.remove_food_at_point(point)

            state_old = snake_agent.get_state(game)
            game.snake_apply_action(action, is_human=True)
            score = game.score

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
obstacles_to_load = './level_1/obstacles.csv'
foods_to_load = MAPS_FOLDER + './level_1/foods.csv'

snake_agent = SnakeAgent(is_load_weights=is_load_weights, weights_filename=SNAKE_WEIGHTS_FILENAME)
food_agent = None
game = SnakeGameAI(is_rendering=is_rendering, game_speed=game_speed, is_add_obstacles=True, foods_to_load=foods_to_load)
train(snake_agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent, obstacles_to_load)

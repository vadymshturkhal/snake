import pygame
from game_settings import BLOCK_SIZE, MAPS_FOLDER
from agents.qlearning import QLearning
from game import SnakeGameAI
from game_settings import SNAKE_WEIGHTS_FILENAME, SCORE_DATA_FILENAME
from game_settings import SNAKE_SPEED, FOOD_SPEED_MULTIPLIER

from game_utils import Point


pygame.init()


def train(snake_agent, game, score_data_filename, games_to_play=0, food_agent=None, obstacles_to_load='new_obstacles.csv', foods_to_load='new_foods.csv'):
    total_score = 0

    counter = 0

    game.obstacles.load_obstacles_from_file(MAPS_FOLDER + obstacles_to_load)
    game.play_step()
    while counter < games_to_play:
        action = [0, 0, 0]
        activate_play_step = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.obstacles.save_obstacles(MAPS_FOLDER + obstacles_to_load)
                game.foods.save_foods()
                pygame.quit()
                print('Quit')
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = [1, 0, 0]
                elif event.key == pygame.K_RIGHT:
                    action = [0, 1, 0]
                elif event.key == pygame.K_UP:
                    action = [0, 0, 1]
                
                activate_play_step = True
            
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
                activate_play_step = True

        score = game.score

        if activate_play_step:
            snake_state = game.get_snake_state()
            snake_action = snake_agent.get_action(snake_state)
            game.snake_apply_action(action, is_human=True)
            game.play_step()

        if game.snake_is_crashed:
            game.reset()
            snake_agent.n_games += 1

            scores.append(score)
            total_score += score

            counter += 1


is_load_weights = True
is_rendering = True
game_speed = 10
games_to_play = 3
obstacles_to_load = './level_2/obstacles.csv'
foods_to_load = MAPS_FOLDER + './level_2/foods.csv'

snake_agent = QLearning(is_load_weights=is_load_weights, weights_filename=SNAKE_WEIGHTS_FILENAME)
food_agent = None
game = SnakeGameAI(is_rendering=is_rendering, game_speed=game_speed, is_add_obstacles=True, foods_to_load=foods_to_load, is_place_food=False)
train(snake_agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent, obstacles_to_load)

import pygame
import random
from foods import Foods
from obstacles import Obstacles
from snake import Snake

from game_stats import GameStats
from game_utils import Direction, Point, WHITE, RED, BLACK
from game_settings import BLOCK_SIZE, MAPS_FOLDER, SCREEN_W, SCREEN_H, OBSTACLES_QUANTITY


class SnakeGameAI:
    def __init__(self, is_rendering=False, game_speed=20, is_add_obstacles=False, foods_to_load=None, is_place_food=False):
        self.counter = 0
        self.snake_steps = 0
        self.is_eaten_food = False
        self.snake_is_crashed = False
        self.width = SCREEN_W
        self.height = SCREEN_H
        self.is_rendering = is_rendering
        self.game_speed = game_speed
        self.is_add_obstacles = is_add_obstacles
        self.is_place_food = is_place_food

        self.snake = Snake(head=Point(BLOCK_SIZE, BLOCK_SIZE), game=self, init_direction=Direction.UP)
        self.obstacles = Obstacles(self)
        self.foods = Foods(self, load_from_filename=foods_to_load)
        self.stats = GameStats(self)

        # init display
        if self.is_rendering:
            pygame.init()
            self.font = pygame.font.SysFont('arial', 25)
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

        self.reset()

    def get_snake_state(self):
        return self.stats.get_snake_state()

    # init game state
    def reset(self):
        self.counter += 1
        self.score = 0
        self.snake_steps = 0

        if self.is_add_obstacles:
            self.obstacles.place_random_obstacles(OBSTACLES_QUANTITY)

        self._place_snake(random_place=False)
        self.foods.place_food()
        self.frame_iteration = 0
        self.snake_is_crashed = False

    def _place_snake(self, random_place=True):
        if random_place:
            is_valid_point = False
            while not is_valid_point:
                x = random.randint(0, (self.width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
                y = random.randint(0, (self.height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE

                snake_point = Point(x, y)
                is_valid_point = True

                if self.obstacles.is_point_at_obstacle(snake_point):
                    is_valid_point = False
        else:
            snake_point = Point(0, (self.height - BLOCK_SIZE) // 2)
        self.snake.head = snake_point

    def snake_apply_action(self, action, is_human=False):
        if action == [1, 0, 0]: 
            self.snake.rotate_snake('left')
        elif action == [0, 1, 0]:
            self.snake.rotate_snake('right')
        elif action == [0, 0, 1]:
            self.snake_is_crashed = any([self.snake.move_after_rotation(), self.obstacles.is_point_at_obstacle(self.snake.head)])
            self.snake_steps += 1
        else:
            if not is_human:
                raise Exception(f'Unknown action for snake: {action}')

    def play_step(self):
        # Update UI and clock
        if self.is_rendering:
            self._update_ui()
            self.clock.tick(self.game_speed)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print('Quit')
                    quit()

        self.frame_iteration += 1

        if self.foods.is_empty:
            if self.is_place_food:
                self._place_snake(random_place=False)
                self.foods.place_food()

        if self.foods.is_food_at_point(self.snake.head):
            self.score += 1
            self.foods.remove_food_at_point(self.snake.head)

            self.is_eaten_food = self.foods
        else:
            self.is_eaten_food =  False



    def _update_ui(self):
        self.display.fill(BLACK)

        # Determine the rotation of the snake sprite
        sprite_rect = self.snake.sprite_rotated.get_rect(center=(self.snake.head.x + BLOCK_SIZE / 2, self.snake.head.y + BLOCK_SIZE / 2))
        self.display.blit(self.snake.sprite_rotated, sprite_rect.topleft)

        # Draw food
        for food in self.foods:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw obstacles
        for ob in self.obstacles:
                pygame.draw.rect(self.display, (128, 128, 128), pygame.Rect(ob.x, ob.y, BLOCK_SIZE, BLOCK_SIZE))  # Draw obstacles in gray

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

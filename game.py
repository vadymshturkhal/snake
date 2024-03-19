import pygame
import random
from foods import Foods
from obstacles import Obstacles
from snake import Snake

from game_utils import Direction, Point, WHITE, RED, BLACK
from game_settings import BLOCK_SIZE, MAPS_FOLDER, SCREEN_W, SCREEN_H, OBSTACLES_QUANTITY


pygame.init()
font = pygame.font.SysFont('arial', 25)
FOODS_TO_LOAD = './level_2/foods.csv'

class SnakeGameAI:
    def __init__(self, is_rendering=False, game_speed=20, is_add_obstacles=False):
        self.width = SCREEN_W
        self.height = SCREEN_H
        self.is_rendering = is_rendering
        self.game_speed = game_speed
        self.is_add_obstacles = is_add_obstacles
        self.counter = 0
        self.snake_steps = 0
        self.snake = Snake(head=Point(BLOCK_SIZE, BLOCK_SIZE), game=self, init_direction=Direction.UP)
        self.obstacles = Obstacles(self)
        self.foods = Foods(self, load_from_filename=MAPS_FOLDER + FOODS_TO_LOAD)

        self.previous_angle = None
        self.snake_is_crashed = False

        # init display
        if self.is_rendering:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

        self.reset()

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

        self.previous_angle = None

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

    def is_eaten(self):
        if self.foods.is_food_at_point(self.snake.head):
            self.score += 1
            self.foods.remove_food_at_point(self.snake.head)

            if self.foods.is_empty:
                self._place_snake(random_place=False)
                self.foods.place_food()
            return True
        return False

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                print('Quit')
                quit()

        self.frame_iteration += 1

        # Update UI and clock
        if self.is_rendering:
            self._update_ui()
            self.clock.tick(self.game_speed)

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

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

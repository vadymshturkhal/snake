import pygame
import random
import math
from obstacles import Obstacles
from snake import Snake
from food import Food

from game_utils import Direction, Point, WHITE, RED, BLACK
from game_settings import BLOCK_SIZE, SCREEN_W, SCREEN_H
from game_settings import OBSTACLES_QUANTITY


pygame.init()
font = pygame.font.SysFont('arial', 25)

class SnakeGameAI:
    def __init__(self, is_rendering=False, game_speed=20):
        self.width = SCREEN_W
        self.height = SCREEN_H
        self.is_rendering = is_rendering
        self.game_speed = game_speed
        self.counter = 0
        self.snake_steps = 0

        self.max_possible_distance = math.sqrt(SCREEN_W**2 + SCREEN_H**2) // BLOCK_SIZE
        self.prev_distance = self.max_possible_distance
        self.food_move_counter = 0
        self.food = Food(position=Point(SCREEN_W / 2, SCREEN_H / 2))
        self.snake = Snake(head=Point(SCREEN_W / 2, SCREEN_H / 2), game=self, init_direction=Direction.UP)
        self.obstacles = Obstacles(self)

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

        self.obstacles.place_random_obstacles(OBSTACLES_QUANTITY)
        self._place_snake(random_place=True)
        self._place_food(random_place=True)
        self.frame_iteration = 0
        self.snake_is_crashed = False

        self.previous_angle = None

    def _place_food(self, random_place=True):
        if random_place:
            is_valid_point = False
            while not is_valid_point:
                x = random.randint(0, (self.width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
                # x = self.w-BLOCK_SIZE
                y = random.randint(0, (self.height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE

                food_point = Point(x, y)
                is_valid_point = True

                if food_point == self.snake.head:
                    is_valid_point = False

                if self.obstacles.is_point_at_obstacle(food_point):
                    is_valid_point = False
        else:
            food_point = Point(self.width - BLOCK_SIZE, self.height // 2 - BLOCK_SIZE)
        self.food.position = food_point
    
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
            snake_point = Point(0, self.height // 2 - BLOCK_SIZE)
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
        if self.food.position == self.snake.head:
            self.score += 1
            self._place_food()
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
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.position.x, self.food.position.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw obstacles
        for ob in self.obstacles.obstacles:
                pygame.draw.rect(self.display, (128, 128, 128), pygame.Rect(ob.x, ob.y, BLOCK_SIZE, BLOCK_SIZE))  # Draw obstacles in gray

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

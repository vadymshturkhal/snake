import pygame
import random
import numpy as np
import math
from snake import Snake
from food import Food

from game_utils import Direction, Point, WHITE, RED, BLUE1, BLACK
from game_settings import BLOCK_SIZE, SCREEN_W, SCREEN_H
from game_settings import DIRECTIONS_QUANTITY, OBSTACLES_QUANTITY


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
        self.snake = Snake(head=Point(SCREEN_W / 2, SCREEN_H / 2), init_direction=Direction.RIGHT)

        self.previous_angle = None
        self.obstacles_quantity = OBSTACLES_QUANTITY

        self.snake_is_crashed = False

        # init display
        if self.is_rendering:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

        self.obstacles = []
        self.reset()

    # init game state
    def reset(self):
        self.counter += 1
        self.score = 0
        self.snake_steps = 0

        # self.obstacles.clear()
        # self._place_random_obstacles()
        self._place_snake(random_place=True)
        self._place_food(random_place=True)
        self.frame_iteration = 0

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

                for obstacle in self.obstacles:
                    if food_point == obstacle:
                        is_valid_point = False
                        break
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

                for obstacle in self.obstacles:
                    if snake_point == obstacle:
                        is_valid_point = False
                        break
        else:
            snake_point = Point(0, self.height // 2 - BLOCK_SIZE)
        self.snake.head = snake_point

    def snake_move(self, action, is_human=False):
        self.snake_is_crashed = any([self.snake.move(action, is_human=is_human), self.is_collision_with_obstacle()])
        self.snake_steps += 1

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

    def is_collision_with_obstacle(self, pt=None):
        if pt is None:
            pt = self.snake.head

        # Check if head hits any obstacle
        for obstacle in self.obstacles:
            if obstacle == pt:
                return True

        return False

    def _place_random_obstacles(self):
        self.obstacles = []
        
        for _ in range(self.obstacles_quantity):
            is_valid_point = False
            while not is_valid_point:
                x = random.randint(0, (self.width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
                y = random.randint(0, (self.height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE

                obstacle_point = Point(x, y)
                is_valid_point = True

                for obstacle in self.obstacles:
                    if obstacle_point == obstacle:
                        is_valid_point = False
                        break

            self.obstacles.append(obstacle_point)

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw snake
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.snake.head.x, self.snake.head.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.position.x, self.food.position.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw obstacles
        for ob in self.obstacles:
                pygame.draw.rect(self.display, (128, 128, 128), pygame.Rect(ob.x, ob.y, BLOCK_SIZE, BLOCK_SIZE))  # Draw obstacles in gray

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        # Current direction index
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % DIRECTIONS_QUANTITY
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        elif np.array_equal(action, [0, 0, 1]): # [0, 0, 1]
            next_idx = (idx - 1) % DIRECTIONS_QUANTITY
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        else:
            raise Exception('Unknown direction', )

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

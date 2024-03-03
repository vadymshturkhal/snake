import pygame
import random
import numpy as np
import math
from snake import Snake
from food import Food
import time

from game_utils import Point, Direction, WHITE, RED, BLUE1, BLUE2, BLACK
from game_utils import calculate_distance, calculate_angle
from game_settings import BLOCK_SIZE, DIRECTIONS_QUANTITY, FRAME_RESTRICTION
from game_settings import SCREEN_W, SCREEN_H
from game_settings import REWARD_WRONG_DIRECTION, REWARD_CORECT_DIRECTION, REWARD_WIN, REWARD_LOOSE
from game_settings import SNAKE_ANGLE_REWARD, SNAKE_ANGLE_PUNISH


pygame.init()
font = pygame.font.SysFont('arial', 25)

class SnakeGameAI:
    def __init__(self, is_rendering=False, game_speed=20):
        self.w = SCREEN_W
        self.h = SCREEN_H
        self.is_rendering = is_rendering
        self.game_speed = game_speed

        self.max_possible_distance = math.sqrt(SCREEN_W**2 + SCREEN_H**2)
        self.prev_distance = self.max_possible_distance
        self.food_move_counter = 0
        self.food = Food(head=Point(SCREEN_W / 2, SCREEN_H / 2))

        self.previous_angle = None

        # init display
        if self.is_rendering:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

        self.reset()

    # init game state
    def reset(self):
        self.snake = Snake(head=Point(self.w/2, self.h/2), init_direction=Direction.RIGHT)
        
        self.score = 0
        self._place_food()
        self.frame_iteration = 0

        self.previous_angle = None

    def _place_food(self, random_place=True):
        if random_place:
            while True:
                x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
                y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE

                if x == self.snake.head.x and y == self.snake.head.y:
                    continue

                self.food.head = Point(x, y)
                break

    def scores_to_csv(self, filename, scores):
        with open(filename, 'a') as file:
            file.write(f'{str(scores[-1])} \n')

    def snake_move(self, action):
        self.snake.move(action)

    def food_move(self, action=None):
        # Assuming snake_head and food_position are Point objects with x and y attributes
        distance = calculate_distance(self.snake.head, self.food.head)

        if distance >= self.prev_distance:
            reward = 2
        else:
            reward = -0.1

        self.food.move(action)

        return reward

    def is_eaten(self):
        if self.food.head == self.snake.head:
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

        # Punish snake if game is over
        game_over = False
        if self.is_collision() or self.frame_iteration > FRAME_RESTRICTION:
            game_over = True
            return REWARD_LOOSE, game_over

        # Update UI and clock
        if self.is_rendering:
            self._update_ui()
            self.clock.tick(self.game_speed)

        # Return 0 reward if game is not over.
        return 0, game_over


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw snake
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.snake.head.x, self.snake.head.y, BLOCK_SIZE, BLOCK_SIZE))

        # ?
        pygame.draw.rect(self.display, BLUE2, pygame.Rect(self.snake.head.x+4, self.snake.head.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.head.x, self.food.head.y, BLOCK_SIZE, BLOCK_SIZE))

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

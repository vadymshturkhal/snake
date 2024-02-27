import pygame
import random
import numpy as np
import math
from snake import Snake
from food import Food

from game_utils import Point, Direction, WHITE, RED, BLUE1, BLUE2, BLACK
from game_utils import calculate_distance_and_angle, normalize_distance
from game_settings import BLOCK_SIZE, DIRECTIONS_QUANTITY, FRAME_RESTRICTION
from game_settings import SCREEN_W, SCREEN_H
from game_settings import REWARD_WRONG_DIRECTION, REWARD_CORECT_DIRECTION, REWARD_WIN, REWARD_LOOSE


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
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        # self.food_direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])
        # self._move_food()

    def _place_food(self, random_place=True):
        if random_place:
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            self.food = Food(x, y)
            
            if self.food.coordinates == self.snake.head:
                self._place_food()

    def scores_to_csv(self, filename, scores):
        with open(filename, 'a') as file:
            for score in scores:
                file.write(f'{str(score)} \n')

    def play_step(self, action):
        self.frame_iteration += 1

        # Assuming snake_head and food_position are Point objects with x and y attributes
        distance, angle = calculate_distance_and_angle(self.snake.head, self.food)

        # Example usage
        normalized_distance = normalize_distance(distance, self.max_possible_distance)

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                print('Quit')
                quit()
        
        # 2. move
        self.snake.move(action)

        if self.prev_distance > distance:
            reward = REWARD_CORECT_DIRECTION
        else:
            reward = REWARD_WRONG_DIRECTION

        self.prev_distance = distance

        # 3. check if game is over
        game_over = False
        if self.is_collision() or self.frame_iteration > FRAME_RESTRICTION:
            game_over = True
            reward = REWARD_LOOSE
            return reward, game_over, self.score

        # 4. place new food
        if self.food.coordinates == self.snake.head:
            self.score += 1
            reward = REWARD_WIN
            self._place_food()

        # 5. update ui and clock
        if self.is_rendering:
            self._update_ui()
            self.clock.tick(self.game_speed)

        # 6. return game over and score
        return reward, game_over, self.score


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
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

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
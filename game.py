import pygame
import random
from game_utils import Point, Direction
import numpy as np
from snake import Snake
from game_settings import BLOCK_SIZE, DIRECTIONS_QUANTITY, FRAME_RESTRICTION


# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)


pygame.init()
font = pygame.font.SysFont('arial', 25)

class SnakeGameAI:
    def __init__(self, w=640, h=480, is_rendering=False, game_speed=20):
        self.w = w
        self.h = h
        self.is_rendering = is_rendering
        self.game_speed = game_speed

        
        # init display
        if self.is_rendering:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head]

        self.snake_class = Snake(head=Point(self.w/2, self.h/2), init_direction=Direction.RIGHT)
        print('snake_class')
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        self.food_direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])
        # self._move_food()

    def _place_food(self, random_place=True):
        if random_place:
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            self.food = Point(x, y)
            if self.food in self.snake:
                self._place_food()
    
    def _move_food(self):
        print('before', self.food)
        directions = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]
        self.food_direction = random.choice(directions)  # Randomly change direction
        # self.food_direction = Direction.RIGHT  # Randomly change direction
        
        x, y = self.food
        if self.food_direction == Direction.RIGHT:
            x = min(x + BLOCK_SIZE, self.w - BLOCK_SIZE)
        elif self.food_direction == Direction.LEFT:
            x = max(x - BLOCK_SIZE, 0)
        elif self.food_direction == Direction.UP:
            y = max(y - BLOCK_SIZE, 0)
        elif self.food_direction == Direction.DOWN:
            y = min(y + BLOCK_SIZE, self.h - BLOCK_SIZE)
        
        self.food = Point(x, y)
        print('after', self.food)

    def scores_to_csv(self, filename, scores):
        with open(filename, 'a') as file:
            for score in scores:
                file.write(f'{str(score)} \n')

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                print('Quit')
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > FRAME_RESTRICTION:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        self.snake.pop()


        # 5. update ui and clock
        if self.is_rendering:
            self._update_ui()
            self.clock.tick(self.game_speed)

            # Consistent speed
            # self._move_food()
        # else:
        self._move_food()

        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

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
# Files
SNAKE_WEIGHTS_FILENAME = './model/model_snake.pth'
FOOD_WEIGHTS_FILENAME = './model/model_food.pth'
SCORE_DATA_FILENAME = './data/latest.csv'

# Screen
SCREEN_W = 200
SCREEN_H = 200

# Game
BLOCK_SIZE = 20
DIRECTIONS_QUANTITY = 4
FRAME_RESTRICTION = 400

# Obstacles
OBSTACLES_QUANTITY = 20

# Speed
GAME_SPEED = 40
SNAKE_SPEED = 0.000004
FOOD_SPEED_MULTIPLIER = 2

# Train
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
AVAILABLE_SNAKE_DIRECTIONS_QUANTITY = 3
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-5
SNAKE_MIN_EPSILON = 0.05

# Snake layers
SNAKE_INPUT_LAYER_SIZE = 12
SNAKE_HIDDEN_LAYER_SIZE1 = 128
SNAKE_HIDDEN_LAYER_SIZE2 = 128
SNAKE_OUTPUT_LAYER_SIZE = 3

# Snake rewards
REWARD_CRAWLING = 0.01
REWARD_CORECT_DIRECTION = 0.1
REWARD_WRONG_DIRECTION = -0.1
REWARD_WIN = 20
REWARD_LOOSE = -10
SNAKE_ANGLE_REWARD = 0.1
SNAKE_ANGLE_PUNISH = -0.1

# Food
FOOD_INPUT_LAYER_SIZE = 14
FOOD_HIDDEN_LAYER_SIZE1 = 128
FOOD_HIDDEN_LAYER_SIZE2 = 64
FOOD_OUTPUT_LAYER_SIZE = 4

# Food rewards
BASE_REWARD = 0.1
REWARD_INCREASE_DISTANCE = 1

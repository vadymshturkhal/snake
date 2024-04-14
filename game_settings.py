# Files
SNAKE_WEIGHTS_FILENAME = './model/model_snake.pth'
FOOD_WEIGHTS_FILENAME = './model/model_food.pth'
SCORE_DATA_FILENAME = './data/latest.csv'

# SPRITES
SNAKE_SPRITE_PATH = './sprites/snake_head.png'

# Screen
SCREEN_W = 210
SCREEN_H = 210

# Game
BLOCK_SIZE = 30
DIRECTIONS_QUANTITY = 4
FRAME_RESTRICTION = 300

# Obstacles
MAPS_FOLDER = './maps/'
OBSTACLES_QUANTITY = 0
IS_ADD_OBSTACLES = False

# Speed
GAME_SPEED = 40
SNAKE_SPEED = 0.000004
FOOD_SPEED_MULTIPLIER = 2

# Train
MAX_MEMORY = 100_000
BATCH_SIZE = 1024
SNAKE_ACTION_LENGTH = 3
DROPOUT_RATE = 0.2
LR = 0.0001
WEIGHT_DECAY = 1e-5
SNAKE_GAMMA = 0.95
SNAKE_START_EPSILON = 1
SNAKE_MIN_EPSILON = 0.1
EPSILON_SHIFT = 0
SNAKE_ENERGY = 400
TRAINER_STEPS = 3

# Snake vision
CODE_SNAKE = -1
CODE_UNKNOWN = 0
CODE_OBSTACLES = 1
CODE_FOOD = -2
SNAKE_VISION_RANGE = 3

# Snake layers
SNAKE_INPUT_LAYER_SIZE = 57
SNAKE_HIDDEN_LAYER_SIZE1 = 256
SNAKE_HIDDEN_LAYER_SIZE2 = 256
SNAKE_OUTPUT_LAYER_SIZE = 3

# Snake rewards
REWARD_CRAWLING = 0
REWARD_ROTATION = 0
REWARD_WIN = 10
REWARD_LOOSE = -4

# Food
FOOD_QUANTITY = 1
FOOD_INPUT_LAYER_SIZE = 14
FOOD_HIDDEN_LAYER_SIZE1 = 128
FOOD_HIDDEN_LAYER_SIZE2 = 128
FOOD_OUTPUT_LAYER_SIZE = 4

# Food rewards
BASE_REWARD = 0.1
REWARD_INCREASE_DISTANCE = 1

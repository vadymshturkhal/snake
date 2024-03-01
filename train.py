from agent_snake import SnakeAgent
from agent_food import FoodAgent
from game import SnakeGameAI
from collections import namedtuple
from game_settings import REWARD_WIN, SNAKE_WEIGHTS_FILENAME, FOOD_WEIGHTS_FILENAME, SCORE_DATA_FILENAME


# Extend the Transition namedtuple with a 'done' field
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def assure_data_csv(filename, is_load_weights):
    if is_load_weights:
        return

    with open(filename, 'w') as file:
        file.write('Score, \n')

def train(snake_agent, game, score_data_filename, games_to_play=0, food_agent=None):
    scores = []
    total_score = 0
    record = 0

    game_counter = 0
    while game_counter < games_to_play:
        # Snake Agent
        state_old = snake_agent.get_state(game)
        snake_next_move = snake_agent.get_action(state_old)
        snake_reward, score = game.snake_move(snake_next_move)

        if game.is_eaten():
            snake_reward += REWARD_WIN

        punishment, done = game.play_step()

        # Pubish snake if game is lost
        snake_reward += punishment

        # Train snake
        state_new = snake_agent.get_state(game)
        snake_agent.train_short_memory(state_old, snake_next_move, snake_reward, state_new, done)
        snake_agent.remember(state_old, snake_next_move, snake_reward, state_new, done)

        if done:
            game.reset()
            snake_agent.n_games += 1
            snake_agent.train_long_memory()

            food_agent.train_long_memory()

            if score > record:
                record = score
                snake_agent.model.save(epoch=agent.n_games, filename=SNAKE_WEIGHTS_FILENAME)
                food_agent.model.save(epoch=agent.n_games, filename=FOOD_WEIGHTS_FILENAME)

            scores.append(score)
            total_score += score

            game_counter += 1
        else:
            # Food Agent
            # Random
            # game.food_move()

            food_state_old = food_agent.get_state(game)
            food_next_move = food_agent.get_action(food_state_old)
            food_reward = game.food_move(food_next_move)

            # Train food
            food_state_new = food_agent.get_state(game)
            food_agent.train_short_memory(food_state_old, food_next_move, food_reward, food_state_new)
            food_agent.remember(food_state_old, food_next_move, food_reward, food_state_new)

    game.scores_to_csv(score_data_filename, scores)


is_load_weights = False
is_rendering = False
game_speed = 40
games_to_play = 100

assure_data_csv(SCORE_DATA_FILENAME, is_load_weights)

agent = SnakeAgent(is_load_weights=is_load_weights, weights_filename=SNAKE_WEIGHTS_FILENAME, epochs=games_to_play)
food_agent = FoodAgent(is_load_weights=False, weights_filename=FOOD_WEIGHTS_FILENAME, epochs=games_to_play)

game = SnakeGameAI(is_rendering=is_rendering, game_speed=game_speed)
train(agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent)
from agent import SnakeAgent, FoodAgent
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

def train(agent, game, score_data_filename, games_to_play=0, food_agent=None):
    scores = []
    total_score = 0
    record = 0

    game_counter = 0
    while game_counter < games_to_play:
        state_old = agent.get_state(game)
        snake_move = agent.get_action(state_old)
        snake_reward, score = game.snake_move(snake_move)

        if game.is_eaten():
            snake_reward += REWARD_WIN

        punishment, done = game.play_step()

        # Pubish snake if game lost
        snake_reward += punishment
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, snake_move, snake_reward, state_new, done)
        agent.remember(state_old, snake_move, snake_reward, state_new, done)

        # Food Agent
        # food_state_old = food_agent.get_state(game)
        # food_next_move = food_agent.get_action(food_state_old)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(epoch=agent.n_games)

            scores.append(score)
            total_score += score

            game_counter += 1

    game.scores_to_csv(score_data_filename, scores)


is_load_weights = False
is_rendering = False
game_speed = 40
games_to_play = 120

assure_data_csv(SCORE_DATA_FILENAME, is_load_weights)

agent = SnakeAgent(is_load_weights=is_load_weights, weights_filename=SNAKE_WEIGHTS_FILENAME)
food_agent = FoodAgent(is_load_weights=False, weights_filename=FOOD_WEIGHTS_FILENAME)

game = SnakeGameAI(is_rendering=is_rendering, game_speed=game_speed)
train(agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent)
from agent import SnakeAgent, FoodAgent
from game import SnakeGameAI
from game_settings import SNAKE_WEIGHTS_FILENAME, FOOD_WEIGHTS_FILENAME, SCORE_DATA_FILENAME


def assure_data_csv(filename, is_load_weights):
    if is_load_weights:
        return

    with open(filename, 'w') as file:
        file.write('Score, \n')

def train(snake_agent, game, score_data_filename, games_to_play=0, food_agent=None):
    scores = []
    mean_scores = []
    total_score = 0

    counter = 0
    while counter < games_to_play:
        # get old state
        state_old = snake_agent.get_state(game)

        # get move
        snake_move = snake_agent.get_action(state_old)

        # perform move and get new state
        snake_reward, score = game.snake_move(snake_move)
        punishment, done = game.play_step()

        if done:
            game.reset()
            snake_agent.n_games += 1

            scores.append(score)
            total_score += score
            mean_score = total_score / snake_agent.n_games
            mean_scores.append(mean_score)

            counter += 1

    game.scores_to_csv(score_data_filename, scores)


is_load_weights = True
is_rendering = True
game_speed = 40
games_to_play = 10

assure_data_csv(SCORE_DATA_FILENAME, is_load_weights)

snake_agent = SnakeAgent(is_load_weights=is_load_weights, weights_filename=SNAKE_WEIGHTS_FILENAME)
food_agent = FoodAgent(is_load_weights=False, weights_filename=FOOD_WEIGHTS_FILENAME)
game = SnakeGameAI(is_rendering=is_rendering, game_speed=game_speed)
train(snake_agent, game, SCORE_DATA_FILENAME, games_to_play, food_agent)
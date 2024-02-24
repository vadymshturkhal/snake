from agent import Agent
from game import SnakeGameAI
from collections import namedtuple


# Extend the Transition namedtuple with a 'done' field
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

GAME_SPEED = 20
WEIGHTS_FILENAME = './model/model.pth'
SCORE_DATA_FILENAME = './data.csv'

def assure_data_csv(filename, is_load_weights):
    if is_load_weights:
        return

    with open(filename, 'w') as file:
        file.write('Score, \n')

def train(agent, game, score_data_filename, games_to_play=0):
    scores = []
    total_score = 0
    record = 0

    counter = 0
    while counter < games_to_play:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(epoch=agent.n_games)

            # print('Game', agent.n_games, 'Score', score, 'Record:', record)

            scores.append(score)
            total_score += score

            # game.record_score(score_data_filename, score)
            # plot(plot_scores, plot_mean_scores)
            counter += 1

    game.scores_to_csv(score_data_filename, scores)


is_load_weights = False
is_rendering = False
game_speed = 40
games_to_play = 100

assure_data_csv(SCORE_DATA_FILENAME, is_load_weights)

agent = Agent(is_load_weights=is_load_weights, weights_filename=WEIGHTS_FILENAME)
game = SnakeGameAI(is_rendering=is_rendering, game_speed=game_speed)
train(agent, game, SCORE_DATA_FILENAME, games_to_play)
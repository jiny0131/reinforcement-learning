from agents.myTeam import myAgent
from Splendor.splendor_model import SplendorGameRule as GameRule
from Splendor.splendor_model import SplendorState
from Splendor.splendor_model import Card
import numpy as np

if __name__ == '__main__':
    myteam = myAgent(1)
    # myteam.train()

    game = GameRule(2)
    # myteam.self_play(40)

    print(game.current_game_state.board.nobles)
    print(myteam.find_nearest_noble(game.current_game_state, 0))
    # for action in actions:
    #     print(action)
    #     print(myteam.get_actiondata(action))
    #     # print(myteam.reward(game.current_game_state, action, game.getCurrentAgentIndex(), game))
    #     print("-------------------------------------------------------")

    # action = {'type': 'collect_same', 'collected_gems': {"black" : 2}, 'returned_gems': None, 'noble': None}
    # print(action)
    # print(myteam.reward(game.current_game_state, action, 0, game))
    
    # sd = np.array(myteam.get_statedata(state))
    # ad = np.array(myteam.get_actiondata(action))
    # label = np.array(label)
    # model = myteam.create_model()
    # model.fit(sd, ad, label, epochs = 1)
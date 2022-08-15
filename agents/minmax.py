from template import Agent
import time
import random
from Splendor.splendor_model import SplendorGameRule as GameRule

THINKTIME = 0.95


class Node(object):
    def __init__(self, state, action):
        self.state = state
        self.action = action

def ab_minimax(node, alpha, beta, current_depth, game, id, search_depth):
        current_depth += 1
        if current_depth == search_depth:
        #if current_depth == 3:
            board_value = evaluation(node.state, id)
            if current_depth % 2 == 0:
                # pick smallest number, where root is black and odd depth
                if (beta > board_value):
                    beta = board_value
                return beta
                
            else:
                # pick largest number, where root is black and even depth
                if (alpha < board_value):
                    alpha = board_value
                return alpha

        if current_depth % 2 == 0:
            # min player's turn
            opp_id = abs(id -1)
            queue = expand_node(node, game, opp_id)
            for child_node in queue:
                if alpha < beta:
                    board_value = ab_minimax(child_node,alpha, beta, current_depth, search_depth)
                    if beta > board_value:
                        beta = board_value
            return beta
            
        else:
            # max player's turn
            queue = expand_node(node, game, id)
            for child_node in queue:
                if alpha < beta:
                    board_value = ab_minimax(child_node, alpha, beta, current_depth, search_depth)
                    if alpha < board_value:
                        alpha = board_value

            return alpha

def expand_node(node, game, id):
    actions = game.getLegalActions(node.state, id)
    node_l = []
    for action in actions:
        new_state = game.generateSuccessor(node.state, action, id)
        node = Node(new_state, action)
        node_l.append(node)
    return node_l

def evaluation(state, id):
    opp_id = abs(id -1)
    board_state = state.board
    my_state = state.agents[id]
    opp_state = state.agents[opp_id]
    total_score = -card_score(my_state,board_state) + card_score(opp_state, board_state)
    return total_score

def card_score(agent_state, board_state):
    cards = agent_state.cards
    score = 0
    for tier in board_state.decks:
        for card in tier:
            cost = card.cost
            for key in cost:
                if(cost[key] > len(cards[key])):
                    score += cost[key] - len(cards[key])
    for noble in board_state.nobles:
        cost = noble[1]
        for key in cost:
            if(cost[key] > len(cards[key])):
                score += 1*(cost[key] - len(cards[key]))
    return score


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
    
    def SelectAction(self,actions,game_state):
        start_time = time.time()
        game = GameRule(2)
        alpha = -9999
        beta = 9999
        while time.time()-start_time < THINKTIME:
            for action in actions:
                node = Node(game.generateSuccessor(game_state, action, self.id), action)
                board_value = ab_minimax(node, alpha, beta, 0, game, self.id, search_depth = 1)
                if alpha < board_value:
                    alpha = board_value
                    best_node = node 
            return best_node.action
        return random.choice(actions)

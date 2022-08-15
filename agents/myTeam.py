from re import L
from tensorflow.python.keras.models import Sequential
from tensorflow.python.platform.tf_logging import error
from game import Game
from tensorflow.python.autograph.pyct.parser import MAX_SIZE
from template import Agent
import random

from Splendor.splendor_model import SplendorGameRule as GameRule
from Splendor.splendor_model import SplendorState

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.optimizers import Adam
from collections import deque

colour_dict = {'black':0, 'red':1, 'green':2, 'blue':3, 'white':4, 'yellow':5}

MEMORY_SIZE = 500
MIN_MEMORY_SIZE = 150
BATCH_SIZE = 64
DISCOUNT = 0.9
EPSILON = 0.95

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game = GameRule(2)
        self.model = self.create_model()

        # we use it to predict q value every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.memory = deque(maxlen= MEMORY_SIZE)
        
        self.target_updata_counter = 0

    def self_play(self, round):
        for i in range(round):
            done = False
            game = GameRule(2)
            epsido_reward, step = 0, 0
            while not done:

                current_state, player = game.current_game_state, game.getCurrentAgentIndex()
                if(random.random() > EPSILON):
                    q_value, action = self.get_future_q(current_state, player, self.model)
                else:
                    action = random.choice(game.getLegalActions(current_state, player))

                game.update(action)
                reward  = self.reward(game.current_game_state, action, player, game)
                epsido_reward += reward

                step += 1

                q, opp_action = self.get_future_q(game.current_game_state, game.current_agent_index, self.model)
                game.update(opp_action)

                done, new_state = game.gameEnds(), game.current_game_state
                self.memory.append((current_state, action, reward, new_state, player, done))
                step += 1

                if(step % 10 == 0):
                    self.train()
                
            print(epsido_reward, step)
            print("contain memory", len(self.memory))
            if(i > 4):
                print("----------------------------SAVE MODEL------------------------------")
                self.model.save('saved_model/model_with_epsido_reward_%d_total_step_%d'%(epsido_reward, step))


    # The reward for player 0 is positive and reward for player 1 is negetive
    # When choosing action by q value, player 0 always choose the biggest q and player 1 always choose the smallest q
    def reward(self, state, action, player, game):
        reward = 0
        if(action['type'] == 'pass'):
            reward = 0
        elif(action['type'] == 'buy_available' or action['type'] == 'buy_reserve'):
            card = action['card']
            reward += card.points * 5
            if(action['noble']):
                reward += 15
            colour = card.colour
            
            for noble in state.board.nobles:
                if(colour in noble[1] and len(state.agents[player].cards[colour]) < noble[1][colour]):
                    reward += 2
        elif(action['type'] == 'reserve' and action['collected_gems']):
            if(game.resources_sufficient(state.agents[abs(player -1)], action['card'].cost) and action['card'].points > 2):
                reward += 5
        else:
            reward += sum(action['collected_gems'][colour] for colour in action['collected_gems'])
            if(action['returned_gems']):
                reward = -100
        return reward

    def find_nearest_card(self, state, player):
        agent = state.agents[player]
        dealt = state.board.dealt
        card_list, min_diff = [], 1000
        for deck in dealt:
            for card in deck:
                diff = 0
                for colour in card.cost:
                    if(card.cost[colour] - agent.gems[colour] - len(agent.cards[colour]) > 0):
                        diff += card.cost[colour] - agent.gems[colour] - len(agent.cards[colour])
                if(diff <= min_diff):
                    if(diff == min_diff):
                        card_list.append(card)
                    else:
                        min_diff = diff
                        card_list = []
                        card_list.append(card)
        return card_list
    
    def find_nearest_noble(self, state, player):
        agent = state.agents[player]
        nobles = state.board.nobles
        noble_list, min_diff = [], 1000
        for noble in nobles:
            diff = 0
            for colour in noble[1]:
                if(noble[1][colour] - len(agent.cards[colour]) > 0):
                    diff += noble[1][colour] - len(agent.cards[colour])
            if(diff <= min_diff):
                if(diff == min_diff):
                    noble_list.append(noble)
                else:
                    min_diff = diff
                    noble_list = []
                    noble_list.append(noble)
        return noble_list

                
        
    def train(self):
        if(len(self.memory) < MIN_MEMORY_SIZE):
            return
        batch = random.sample(self.memory, BATCH_SIZE)

        x1,x2, y = [], [],[]
        for (current_state, action, reward, new_state, player, done) in batch:
            current_state_data = self.get_statedata(current_state, player)
            current_action_data = self.get_actiondata(action)
            if not done:
                # 应该是对手的action 的qvalue，取最小的？更改更改更改更改更改更改更改更改更改更改
                max_future_q, max_future_action = self.get_future_q(new_state, player, self.target_model)
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            x1.append(current_state_data)
            x2.append(current_action_data)
            y.append(new_q)

        self.model.fit([np.array(x1), np.array(x2)], np.array(y), batch_size = BATCH_SIZE)

        self.target_updata_counter += 1
        if(self.target_updata_counter > 5):
            self.target_model.set_weights(self.model.get_weights())
            self.target_updata_counter = 0
        # game = GameRule(2)
        # state = SplendorState(2)
        # action = game.getLegalActions(state, 1)[0]
        # label = [1]

        # sd = np.array([self.get_statedata(state)])
        # ad = np.array([self.get_actiondata(action)])
        # label = np.array(label)
        # model = self.create_model()
        # model.fit([sd, ad], label, epochs= 5)

    
    def create_model(self):
        state_inpt = layers.Input(shape = (133,))
        state_layer = layers.Dense(64, activation = 'relu')(state_inpt)
        state_output = layers.Dense(32, activation = 'relu')(state_layer)
        state_model = Model(inputs = state_inpt, outputs = state_output)

        action_inpt = layers.Input(shape = (3,))
        # action_layer = layers.Dense(8, activation = 'relu')(action_inpt)
        action_output = layers.Dense(1, activation = 'relu')(action_inpt)
        action_model = Model(inputs = action_inpt, outputs = action_output)

        combined = layers.concatenate([state_model.output, action_model.output])
        combined = layers.Dense(16, activation = 'relu')(combined)
        result = layers.Dense(1, activation = 'relu')(combined)

        model = Model(inputs = [state_model.input, action_model.input], outputs = result)
        model.compile(loss='mse',optimizer=Adam(lr=0.01))
        return model

    def get_actiondata(self, action):
        # Score, gem collected, gem return
        if(action['type'] == 'pass'):
            data = [0,0,0]
        elif(action['type'] == 'collect_diff' or action['type'] == 'collect_same' or action['type']  == 'reserve'):
            data = [0]
            data.append(sum(action['collected_gems'][colour] for colour in action['collected_gems']))
            data.append(sum(action['returned_gems'][colour] for colour in action['returned_gems']))
        else:
            score = action['card'].points
            if(action['noble']):
                score += 3
            data = [score, 0]
            data.append(sum(action['returned_gems'][colour] for colour in action['returned_gems']))
        # # 16 features ID [0,1], Score, Colour, cost/gem collect(6), return(6), noble
        # if(action['type'] == 'pass'):
        #     data = [0]*15
        #     data.append(1 if action['noble'] else 0)
        # elif(action['type'] == 'collect_diff' or action['type'] == 'collect_same'):
        #     data = [0, 0, 6]
        #     for colour in colour_dict:
        #         data.append(action['collected_gems'][colour] if colour in action['collected_gems'] and action['collected_gems'][colour] else 0)
        #     for colour in colour_dict:
        #         data.append(action['returned_gems'][colour] if colour in action['returned_gems'] and action['returned_gems'][colour] else 0)
        #     data.append(1 if action['noble'] else 0)
        # elif(action['type']  == 'reserve'):
        #     data = [0, action['card'].points, colour_dict[action['card'].colour]]
        #     for colour in colour_dict:
        #         data.append(1 if colour in action['collected_gems'] and action['collected_gems'][colour] else 0)
        #     for colour in colour_dict:
        #         data.append(1 if colour in action['returned_gems'] and action['returned_gems'][colour] else 0)
        #     data.append(1 if action['noble'] else 0)
        # else:
        #     data = [1, action['card'].points, colour_dict[action['card'].colour], 0, 0, 0, 0, 0, 0]
        #     for colour in colour_dict:
        #         data.append(action['returned_gems'][colour] if colour in action['returned_gems'] and action['returned_gems'][colour] else 0)
        #     data.append(1 if action['noble'] else 0)
        return data



    def get_statedata(self, state, player):
        data = []

        boardstate = state.board
        # dealt data, 12 cards in total, each card have 7 features
        for deck in boardstate.dealt:
            for card in deck:
                card_data = [card.points, colour_dict[card.colour]]
                cost_data = []
                for key in colour_dict:
                    if(key == 'yellow'):
                        break
                    cost_data.append(card.cost[key] if key in card.cost else 0)
                card_data.extend(cost_data)
                data.extend(card_data)
        # state gem data, have 6 colour gem
        for key in colour_dict:
            data.append(boardstate.gems[key] if key in boardstate.gems else 0)
        # noble data, with noble costs, every noble have 3 points so no need to record
        noble_count = 0
        for noble in boardstate.nobles:
            noble_count += 1
            for key in colour_dict:
                if(key == 'yellow'):
                    break
                data.append(noble[1][key] if key in noble[1] else 0)
        if(noble_count != 3):
            data.extend([0,0,0,0,0]*(3-noble_count))
        
        
        agentstate = state.agents
        # agent data, with agent score, cards, gems and nobles
        count, curplayer= 0, player
        agent_data = []
        while count < 2:
            agent = agentstate[curplayer]
            agent_data.append(agent.score)
            for colour in colour_dict:
                agent_data.append(len(agent.cards[colour]))
            for colour in colour_dict:
                agent_data.append(agent.gems[colour])
            agent_data.append(len(agent.nobles))
            count += 1
            curplayer = abs(curplayer -1)

        data.extend(agent_data)
        
        return data

    def get_future_q(self, state, player, model):
        actions = self.game.getLegalActions(state, player)
        q_list = []
        for action in actions:
            state_data = np.array([self.get_statedata(state, player)])
            action_data = np.array([self.get_actiondata(action)])
            q_list.append(model.predict([state_data, action_data])[0][0])
        # if(player == 0):
        #     q = max(q_list)
        # else:
        #     q = min(q_list)
        q = max(q_list)
        action = actions[q_list.index(q)]
        return q, action

    def SelectAction(self,actions,game_state):
        model = load_model('saved_model/model_with_epsido_reward_-845_total_step_118')
        q_list = []
        for action in actions:
            state_data = np.array([self.get_statedata(game_state, self.id)])
            action_data = np.array([self.get_actiondata(action)])
            q_value = model.predict([state_data, action_data])[0][0]
            print(action)
            print(q_value)
            q_list.append(q_value)
        # if(self.id == 0):
        #     q = max(q_list)
        # else:
        #     q = min(q_list)
        q = max(q_list)
        action = actions[q_list.index(q)]
        return action
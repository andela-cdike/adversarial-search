import pickle
import random
from collections import defaultdict, Counter

from isolation.isolation import Isolation, _WIDTH, _HEIGHT, _ACTIONSET, Action
from my_custom_player import CustomPlayer


NUM_ROUNDS = 1500


def build_table(num_rounds=NUM_ROUNDS):
    book = defaultdict(Counter)
    for _ in range(num_rounds):
        state = Isolation()
        build_tree(state, book)
    return {k: max(v, key=v.get) for k, v in book.items()}


def build_tree(state, book, depth=5):
    if depth <= 0 or state.terminal_test():
        return -simulate(state)

    agent = CustomPlayer(state.player())
    action = agent.minimax_with_alpha_beta_pruning(state, depth)
    reward = build_tree(state.result(action), book, depth - 1)

    book[state][action] += reward
    return -reward


def simulate(state):
    player_id = state.player()
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
    return -1 if state.utility(player_id) < 0 else 1


if __name__ == '__main__':
    opening_book = build_table(NUM_ROUNDS)
    print(opening_book)
    with open('data.pickle', 'wb') as f:
        pickle.dump(opening_book, f)

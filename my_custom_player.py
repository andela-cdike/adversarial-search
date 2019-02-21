
from isolation.isolation import Isolation
from sample_players import DataPlayer


HEIGHT = 9
WIDTH = 11
CENTER_LOCATION = 57

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        if state.ply_count < 1:
            self.queue.put(CENTER_LOCATION)
        elif state.ply_count < 4:
            if state in self.data:
                self.queue.put(self.data[state])
            else:
                self.queue.put(random.choice(state.actions()))
        else:
            depth_limit = 3
            for depth in range(1, depth_limit):
                best_move = self.minimax_with_alpha_beta_pruning(state, depth)
            self.queue.put(best_move)

    def minimax_with_alpha_beta_pruning(self, state, depth):

        def min_value(state, alpha, beta, depth):
    
            if state.terminal_test():
                return state.utility(self.player_id)

            if depth <= 0:
                return self.score(state)

            value = float('inf')
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                beta = min(beta, value)

            return value

        def max_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)

            if depth <= 0:
                return self.score(state)

            value = float('-inf')
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)

            return value

        alpha = float('-inf')
        beta = float('inf')
        best_score = float('-inf')
        best_move = None
        for action in state.actions():
            value = min_value(state.result(action), alpha, beta, depth - 1)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_move = action
        return best_move

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        own_score = len(own_liberties)
        opp_score = len(opp_liberties)
        for loc in own_liberties:
            own_score += len(state.liberties(loc))
        for loc in opp_liberties:
            opp_score += len(state.liberties(loc))
        if own_score != opp_score:
            return own_score - opp_score
        # what if I only do this when the board is a certain percentage filled. Don't care initially, only later
        else:
            center_loc_cartesian = ind2xy(CENTER_LOCATION)
            own_loc_cartesian = ind2xy(own_loc)
            opp_loc_cartesian = ind2xy(opp_loc)
            own_distance_from_center = abs(own_loc_cartesian[0] - center_loc_cartesian[0]) + abs(own_loc_cartesian[1] - center_loc_cartesian[1])
            opp_distance_from_center = abs(opp_loc_cartesian[0] - center_loc_cartesian[0]) + abs(opp_loc_cartesian[1] - center_loc_cartesian[1])
            # divide by 10 so it doesn't have the same weight as the difference in number of moves
            return (opp_distance_from_center - own_distance_from_center) / 10


def ind2xy(ind):
        """
        From isolation.isolation.DebugState
        Convert from board index value to xy coordinates

        The coordinate frame is 0 in the bottom right corner, with x increasing
        along the columns progressing towards the left, and y increasing along
        the rows progressing towards teh top.
        """
        return (ind % (WIDTH + 2), ind // (WIDTH + 2))

"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return combinedHeuristic(game,player)

def diffSquaredScores(game, player):
    ## This heuristic returns squared difference of player moves  
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    playerMoves = len(game.get_legal_moves(player))
    opponentMoves = len(game.get_legal_moves(game.get_opponent(player)))
    
    return float(playerMoves*playerMoves - opponentMoves*opponentMoves)


def maximizingWinningChances(game, player):
    ## This heuristic returns ratio of player moves left to the opponent moves left
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    playerMoves = len(game.get_legal_moves(player))
    opponentMoves = len(game.get_legal_moves(game.get_opponent(player)))

    if opponentMoves>0:
        return float(playerMoves/opponentMoves)
    else:
        return float(playerMoves)

def combinedHeuristic(game, player):
    if (len(game.get_blank_spaces()) > game.height*game.width/2):
        return maximizingWinningChances(game, player) # Try initially to increase the difference squared scores
    else:
        return diffSquaredScores(game, player) # After less blank spaces are left, try maximizing winning chances



class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = getattr(self,method)
        self.time_left = None
        self.TIMER_THRESHOLD = timeout - 0.5

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        self.time_left = time_left
        iterDepth=1
        ## Check if any legal moves available
        if not legal_moves:
            return (-1, -1)
        try:
        ## If iterative deepening is allowed
            if self.iterative:
                for iterDepth in range(1,game.width*game.height):
                    # The search method call (alpha beta or minimax) should happen in
                    # here in order to avoid timeout. The try/except block will
                    # automatically catch the exception raised by the search method
                    # when the timer gets close to expiring
                    score, move = self.method(game,iterDepth)
                    if score == float('inf'):
                        break
            else:
                score, move = self.method(game,self.search_depth)
        except Timeout:
            pass
        
        if move == (-1, -1):
            move = legal_moves[random.randint(0, len(legal_moves) - 1)]
        
        # Return the best move from the last completed search iteration
        return move
    
    def cutoffTest(self,depth,plyCount,game):
        """ 
        cutoffTest to stop minimax or AlphaBeta search
        Tests if any legal moves are remaining or if the plyCount is equal to depth for returning True
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        plyCount : int
            Current number of plies done
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (bool)
            Cutoff Test is True or False
            
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        if depth == plyCount or game.get_legal_moves()==False:
            return True
        
        else:
            return False


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        def max_value (game, plyCount):
            
            ## If the allowed depth is already reached then return the score based on evaluation function
            if self.cutoffTest(depth,plyCount,game):
                return self.score(game, self)
            
            v = float("-inf")
            
            ## Enumerate over all allowed moves in the current game state 
            ## and Minimize over their resulting states recursively
            for a in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(a),plyCount+1))
            
            return v
        
        def min_value(game, plyCount):
            ## If the allowed depth is already reached then return the score based on evaluation function
            if self.cutoffTest(depth,plyCount,game):
                return self.score(game, self)

            v = float("inf")
            
            ## Enumerate over all allowed moves in the current game state 
            ## and Maximize over their resulting states recursively
            for a in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(a), plyCount+1))
            return v
            
        plyCount =0
        if maximizing_player:
            # Since it is a maximizing player, we take max of all the score,action tuples
            # The subsequent would then be for the adversary i.e. minimizer
            return max([(min_value(game.forecast_move(m), plyCount+1), m) for m in game.get_legal_moves()])
        else:
            return min([(max_value(game.forecast_move(m), plyCount+1), m) for m in game.get_legal_moves()])


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        
        def max_value (game, plyCount, alpha, beta):
            
            if self.cutoffTest(depth,plyCount,game):
                return self.score(game, self)
            
            v = float("-inf")
            
            for a in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(a),plyCount+1, alpha, beta))
                
                ## If the value of v for the current game state is more than beta 
                ## then return from there itself as there is no point in going further down the tree
                if v >= beta:
                    return v
                alpha = max(alpha, v)
                
            return v
        
        def min_value(game, plyCount, alpha, beta):

            if self.cutoffTest(depth,plyCount,game):
                return self.score(game, self)

            v = float("inf")
            
            for a in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(a), plyCount+1, alpha, beta))
                
                ## If the value of v for the current game state is less than alpha 
                ## then return from there itself as there is no point in going further down the tree
                if v <= alpha:
                    return v
                beta = min(beta, v)
                
            return v
            
        plyCount =0

        if maximizing_player:
            
            # Body of alphabeta_search:
            best_score = alpha
            best_action = None
            for m in game.get_legal_moves():
                v = min_value(game.forecast_move(m), plyCount+1, best_score, beta)
                if v > best_score:
                    best_score = v
                    best_action = m
            
            return best_score, best_action
        
        else:
            # Body of alphabeta_search:
            best_score = alpha
            best_action = None
            for m in game.get_legal_moves():
                v = max_value(game.forecast_move(m), plyCount+1, best_score, beta)
                if v < best_score:
                    best_score = v
                    best_action = m
            
            return best_score, best_action


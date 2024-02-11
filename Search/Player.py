#!/usr/bin/env python3
import random
import numpy as np

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
import time


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


def is_terminal(node):
    if len(node.state.get_fish_positions()) == 0:
        return True
    return False


def evaluate_move(node):
    player_position = node.state.get_hook_positions()[0]  # Your boat's position
    fish_positions = node.state.get_fish_positions()
    fish_scores = node.state.get_fish_scores()

    best_value = 0
    for fish, position in fish_positions.items():
        fish_value = fish_scores[fish]
        distance = abs(player_position[0] - position[0]) + abs(player_position[1] - position[1])
        value = fish_value / (distance + 1)  # Adding 1 to avoid division by zero
        if value > best_value:
            best_value = value

    return best_value


def get_children(node):
    children = node.compute_and_get_children()
    sorted_children = sorted(children, key=lambda child: evaluate_move(child), reverse=True)
    return children # sorted_children


def compute_heuristic(node):
    player_pos = node.state.get_hook_positions()  # dict of x, y tuple
    fish_pos = node.state.get_fish_positions()  # dict of x, y tuple
    fish_scores = node.state.get_fish_scores()  # dict of scores
    score_diff = node.state.player_scores[0] - node.state.player_scores[1]

    h = 0
    for fish in fish_pos:
        dx_0 = min(abs(player_pos[0][0] - fish_pos[fish][0]), 20 - abs(player_pos[0][0] - fish_pos[fish][0]))
        dx_1 = min(abs(player_pos[1][0] - fish_pos[fish][0]), 20 - abs(player_pos[1][0] - fish_pos[fish][0]))

        dy_0 = abs(player_pos[0][1] - fish_pos[fish][1])
        dy_1 = abs(player_pos[1][1] - fish_pos[fish][1])

        d_max = dx_0 + dy_0 + 1
        d_min = dx_1 + dy_1 + 1

        h += fish_scores[fish] * (1 / d_max - 1 / d_min)

    return h + score_diff * 100


def init_table():
    # Initialize a Zobrist table
    # Dimension: 2D
    # Size: 20x20
    # Cells contain: 64-bit integers

    hash_table = np.zeros((20, 20), dtype=np.uint64)
    for i in range(20):
        for j in range(20):
            hash_table[i][j] = random.getrandbits(64)
    return hash_table


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.Zobrist_table = init_table()
        self.transposition_table = {}

    def compute_zobrist_hash(self, node):
        # Compute the Zobrist hash of a node
        # The hash is the XOR of the hashes of the positions of the boats
        # and the hash of the position of the fish

        hash_value = 0
        for hook in node.state.get_hook_positions().values():
            hash_value ^= int(self.Zobrist_table[hook[0]][hook[1]])
        for fish in node.state.get_fish_positions().values():
            hash_value ^= int(self.Zobrist_table[fish[0]][fish[1]])

        return hash_value

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find the best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        self.transposition_table.clear()
        initial_time = time.time()
        depth = 7
        best_move = None
        num_fish_left = len(initial_tree_node.state.get_fish_positions())
        time_limit = 0.06 if num_fish_left < 5 else 0.05  # More time if fewer fish

        while time.time() - initial_time < time_limit:
            current_best_move = self.iterative_deepening(initial_tree_node, depth, initial_time)
            if current_best_move is not None:
                best_move = current_best_move
            depth_increment = 2 if num_fish_left < 5 else 1
            depth += depth_increment
        return best_move

    def iterative_deepening(self, node, depth, initial_time):
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        best_score = float('-inf')

        for child in get_children(node):
            score = self.alpha_beta(child, depth, alpha, beta, False, initial_time)
            if score > best_score:
                best_score = score
                best_move = child.move
        return ACTION_TO_STR[best_move]

    def alpha_beta(self, node, depth, alpha, beta, maximizing_player, initial_time):
        # Compute the Zobrist hash of the current node
        zobrist_hash = self.compute_zobrist_hash(node)

        # Check if the hash is in the transposition table
        if zobrist_hash in self.transposition_table:
            return self.transposition_table[zobrist_hash]

        if depth == 0 or is_terminal(node):
            heuristic_value = compute_heuristic(node)
            # Store the heuristic value in the transposition table
            self.transposition_table[zobrist_hash] = heuristic_value
            return heuristic_value

        if time.time() - initial_time > 0.06:
            heuristic_value = compute_heuristic(node)
            # Store the heuristic value in the transposition table
            self.transposition_table[zobrist_hash] = heuristic_value
            return heuristic_value

        if maximizing_player:
            max_eval = float('-inf')
            for child in get_children(node):
                current_value = self.alpha_beta(child, depth - 1, alpha, beta, False, initial_time)
                max_eval = max(max_eval, current_value)
                alpha = max(alpha, current_value)
                if beta <= alpha:
                    break
            # Store the computed value in the transposition table before returning
            self.transposition_table[zobrist_hash] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for child in get_children(node):
                current_value = self.alpha_beta(child, depth - 1, alpha, beta, True, initial_time)
                min_eval = min(min_eval, current_value)
                beta = min(beta, current_value)
                if beta <= alpha:
                    break
            # Store the computed value in the transposition table before returning
            self.transposition_table[zobrist_hash] = min_eval
            return min_eval

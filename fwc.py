import numpy as np
import time

# Four Way Countdown
# 4 player game where the goal is to get rid of all the numbers from 1-12 using two dice rolls and a field operation
# Each player takes it in turns to get rid of a tile based on the dice roles
# If 2 sixes a rolled then the player's tiles reset

# I want to test different strategies and determine what the best strategy is
# Make a NN and train it using NEAT to find a good strategy

# Game Logic

class FWC:
    def __init__(self) -> None:
        # The number of dice rolls that can make the number
        self.numProbs = [15, 13, 12, 9, 8, 9, 6, 7, 5, 5, 2, 4]
    
    def rollDice(self) -> tuple:
        dice_rolls = np.random.randint(1, 7, 2)
        return dice_rolls
    
    def possNums(self, dice_roll: tuple) -> list:
        a = dice_roll[0]
        b = dice_roll[1]
        outcomes = []
        outcomes.append(a+b)
        if a != b:
            outcomes.append(abs(a-b))
        else:
            outcomes.append(0)
        if a * b < 13:
            outcomes.append(a * b)
        else:
            outcomes.append(0)
        if max(a, b) % min(a, b) == 0:
            outcomes.append(int(max(a/b, b/a)))
        else:
            outcomes.append(0)
        return outcomes

    # Picks a random output
    def strat_one(self, moves: list, tiles: list) -> int:
        picks = []
        for i in range(4):
            if moves[i] > 0:
                if tiles[moves[i] - 1] == 0:
                    picks.append(moves[i])
        # If there a no possible moves
        if len(picks) == 0:
            return -1
        else:
            x = np.random.randint(0, len(picks))
            return picks[x]

    # Picks the number that is the hardest to make
    def strat_two(self, moves: list, tiles: list) -> int:
        picks = []
        for i in range(4):
            if moves[i] > 0:
                if tiles[moves[i] - 1] == 0:
                    picks.append(moves[i])
        # If there a no possible moves
        if len(picks) == 0:
            return -1
        # Chosing Logic
        elif len(picks) == 1:
            return picks[0]
        else:
            selection = picks[0]
            for p in picks:
                if self.numProbs[p-1] < self.numProbs[selection - 1]:
                    selection = p
            return p

    def simGames(self, games: int) -> None:
        player_strats = {"player 0":self.strat_one, "player 1":self.strat_two, "player 2":self.strat_one, "player 3":self.strat_one}
        # Game Stats
        player_positions = {f"player {i}":[0]*4 for i in range(4)}
        game_turns = []
        winning_turns = []
        max_resets = []
        game_resets = []
        winning_resets = []
        start_time = time.perf_counter()
        for _ in range(games):
            player_tiles = {f"player {i}": [0]*12 for i in range(4)}
            finished_players = []
            turn_counter = [0, 0, 0, 0]
            player_turn = 0
            player_resets = [0, 0, 0, 0]
            playing = True
            while playing:
                if player_turn not in finished_players:
                    player = f"player {player_turn}"
                    turn_counter[player_turn] += 1
                    dice_roll = self.rollDice()
                    # Tiles reset if double 6 is rolled
                    if sum(dice_roll) == 12:
                        player_resets[player_turn] += 1
                        player_tiles[player] = [0] * 12
                    else:
                        outcomes = self.possNums(dice_roll)
                        move = player_strats[player](outcomes, player_tiles[player])
                        if move > 0:
                            player_tiles[player][move - 1] = 1
                            # Check if the player has finished
                            if sum(player_tiles[player]) == 12:
                                finished_players.append(player_turn)
                                l = len(finished_players)
                                player_positions[player][l-1] += 1
                                # Append winning stats
                                if l == 1:
                                    winning_turns.append(turn_counter[player_turn])
                                    winning_resets.append(player_resets[player_turn])
                                elif l == 3:
                                    game_turns.append(turn_counter[player_turn])
                                    # End the game if only one player is left
                                    for i in range(4):
                                        if i not in finished_players:
                                            finished_players.append(i)
                                            player_positions[f"player {i}"][3] += 1
                                            max_resets.append(max(player_resets))
                                            game_resets.append(sum(player_resets))
                                            playing = False
                                            break
                                    break
                player_turn += 1
                player_turn = player_turn % 4
        # Print stats after all games have been played
        print("=" * 50)
        time_elapsed = time.perf_counter() - start_time
        mins_elapsed = time_elapsed // 60
        secs_elapsed = round(time_elapsed - 60 * mins_elapsed, 3)
        if mins_elapsed < 10:
            print(f"Total time elapsed for {games} games: 0{int(mins_elapsed)}:{secs_elapsed}")
        else:
            print(f"Total time elapsed for {games} games: {int(mins_elapsed)}:{secs_elapsed}")
        print("=" * 50)
        print(f"Longest game: {max(game_turns)} turns")
        print(f"Shortest game: {min(game_turns)} turns")
        print(f"Median game length: {np.median(game_turns)} turns")
        print("=" * 50)
        print(f"Longest win: {max(winning_turns)} turns")
        print(f"Shortest win: {min(winning_turns)} turns")
        print(f"Median win: {np.median(winning_turns)} turns")
        print("=" * 50)
        print(f"Most resets in a game: {max(game_resets)} resets")
        print(f"Most resets for a single player: {max(max_resets)} resets")
        print(f"Most resets for winner: {max(winning_resets)} resets")
        print(f"Median resets per game: {np.median(game_resets)} resets")
        print("Player Positions:")
        print("=" * 50)
        print(" player  |  1st  |  2nd  |  3rd  |  4th")
        for i in range(4):
            record = player_positions[f"player {i}"]
            print(f"player {i} | {round(100 * record[0] / games, 1)}% | {round(100 * record[1] / games, 1)}% | "
            f"{round( 100 * record[2] / games, 1)}% | {round(100 * record[3] / games, 1)}%")

# Simulate 100,000 games
FWC().simGames(100_000)

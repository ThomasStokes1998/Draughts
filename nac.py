# Naughts and Crosses Game for testing alpha-zero type algorithm
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Input
from keras.models import load_model


def decodeBoard(encode: int, listformat=False):
    if listformat:
        newboard = []
    else:
        newboard = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for n in range(9):
        y = int((encode % (3 ** (n + 1))) / (3 ** n))
        sq = int(y * (5 - 3 * y) / 2)
        if listformat:
            newboard.append(sq)
        else:
            newboard[n // 3][n % 3] = sq
    return newboard


class Node:
    def __init__(self, priorprob: float, val: float):
        self.wins = 0
        self.draws = 0
        self.visits = 0
        # Estimated probability of winning from this position
        self.priorprob = priorprob
        self.expanded = False
        # Estimation for the value of the rest of the decision tree
        self.val = val
        # Stores data about child nodes for creating the training data quicker
        self.childkeys = {}

    def searchScore(self, parent):
        u = self.priorprob
        m = parent.visits
        n = self.visits
        return u + math.sqrt(m) / (n + 1)

    def score(self):
        if self.visits == 0:
            return 0.5
        else:
            return (self.wins + 0.5 * self.draws) / self.visits

    def value(self):
        if self.visits == 0:
            return self.val
        else:
            return self.wins / self.visits


class NAC:
    def __init__(self) -> None:
        self.board = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]
        self.player1 = 1
        self.player2 = -1

    def resetBoard(self) -> None:
        self.board = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]

    def printBoard(self, board=None) -> None:
        if board is None:
            board = self.board
        pboard = {0: "   ", self.player1: " X ", self.player2: " O "}
        for i, row in enumerate(board):
            rowstring = ""
            for j, col in enumerate(row):
                rowstring += pboard[col]
                if j < 2:
                    rowstring += "|"
            print(rowstring)

    def move(self, square: int, turn: int, board=None, newboard=False) -> None:
        if board is None:
            board = self.board
        r = square // 3
        c = square % 3
        if board[r][c] == 0:
            if newboard:
                nboard = []
                for i, row in enumerate(board):
                    nrow = []
                    for j, col in enumerate(row):
                        if i == r and j == c:
                            nrow.append(turn)
                        else:
                            nrow.append(col)
                    nboard.append(nrow)
            else:
                board[r][c] = turn
        if board is not None and newboard:
            return nboard

    def checkWin(self, turn: int, board=None) -> bool:
        if board is None:
            board = self.board
        for i in range(3):
            # Check rows
            if sum(board[i]) == 3 * turn:
                return True
            # Check columns
            if sum([board[j][i] for j in range(3)]) == 3 * turn:
                return True
        # Check diagonals
        if sum([board[i][i] for i in range(3)]) == 3 * turn:
            return True
        if sum([board[i][2 - i] for i in range(3)]) == 3 * turn:
            return True
        return False

    def encodeBoard(self, board=None) -> int:
        e = 0
        if board is None:
            board = self.board
        for i, row in enumerate(board):
            for j, col in enumerate(row):
                n = j + 3 * i
                e += 3 ** n * col * (3 * col - 1) / 2
        return int(e)

    def flipBoard(self, board=None):
        if board is None:
            board = self.board
        fboard = []
        for i, row in enumerate(board):
            nrow = []
            for j, col in enumerate(row):
                if col == 0:
                    nrow.append(0)
                else:
                    nrow.append(self.player1 + self.player2 - col)
            fboard.append(nrow)
        return fboard

    def getLegalMoves(self, board=None) -> list:
        moves = []
        if board is None:
            board = self.board
        for i, row in enumerate(board):
            for j, col in enumerate(row):
                if col == 0:
                    moves.append(j + 3 * i)
        return moves

    def playGame(self, showhist=False):
        move_hist = []
        self.resetBoard()
        turn_counter = 0
        turn = self.player1
        playing = True
        while playing:
            if turn == self.player1:
                turn_counter += 1
            print(f"Turn {turn_counter} Player {round((3 - turn) / 2)}")
            self.printBoard()
            sq = int(input(f"Enter a square ({self.getLegalMoves()}):"))
            while self.board[sq // 3][sq % 3] != 0:
                sq = int(input(f"Enter a square ({self.getLegalMoves()}):"))
            self.move(sq, turn)
            move_hist.append(sq)
            if turn_counter >= 3:
                iswon = self.checkWin(turn)
                if iswon:
                    print(f"Player {round((3 - turn) / 2)} wins!")
                    self.printBoard()
                    playing = False
                elif turn_counter == 5 and turn == self.player1:
                    print("Game over. Draw!")
                    self.printBoard()
                    playing = False
            turn = self.player1 + self.player2 - turn
        if showhist:
            return move_hist


class TrainNetwork:
    def __init__(self):
        self.nac = NAC()
        self.nodes = {}

    def simGame(self, policy):
        self.nac.resetBoard()
        turn_counter = 0
        playing = True
        # Stores game data
        movekeys = []
        while playing:
            evals = policy.predict([self.nac.board])[0]
            boardcopy = self.nac.board.copy()
            bckey = self.nac.encodeBoard(boardcopy)
            movekeys.append(bckey)
            if bckey not in self.nodes:
                self.nodes[bckey] = Node(0, evals[-1])
            lmoves = self.nac.getLegalMoves()
            maxmove = 0
            maxeval = 0
            gkeys = []
            for l in lmoves:
                e = evals[l]
                unflippedboard = self.nac.move(l, 1, boardcopy, True)
                flippedboard = self.nac.flipBoard(unflippedboard)
                gkey = self.nac.encodeBoard(flippedboard)
                gkeys.append(gkey)
                if gkey not in self.nodes:
                    fboardval = policy.predict([flippedboard])[0][-1]
                    self.nodes[gkey] = Node(e, fboardval)
                if l not in self.nodes[bckey].childkeys:
                    self.nodes[bckey].childkeys[l] = gkey
                if self.nodes[gkey].searchScore(self.nodes[bckey]) > maxeval:
                    maxmove = l
                    maxeval = self.nodes[gkey].searchScore(self.nodes[bckey])

            self.nodes[bckey].expanded = True
            self.nac.move(maxmove, self.nac.player1)
            if turn_counter >= 4:
                iswon = self.nac.checkWin(self.nac.player1)
                if iswon:
                    wdllist = []
                    for i, mk in enumerate(movekeys):
                        self.nodes[mk].visits += 1
                        if i % 2 == turn_counter % 2:
                            self.nodes[mk].wins += 1
                            wdllist.append(1)
                        else:
                            wdllist.append(0)
                    playing = False
                elif turn_counter == 8:
                    wdllist = []
                    for i, mk in enumerate(movekeys):
                        self.nodes[mk].draws += 1
                        self.nodes[mk].visits += 1
                        wdllist.append(0.5)
                    playing = False
            turn_counter += 1
            self.nac.board = self.nac.flipBoard()
        return movekeys, wdllist

    def playGames(self, games: int, policy, trainprop: float):
        movekeys_ = []
        wdllists_ = []
        lastint = 0
        t0 = time.time()
        for g in range(games):
            movekeys, wdllist = self.simGame(policy)
            
            if g % 10 == 9:
                dt = time.time() - t0
                print(f"Finished training on {g+1} games. {round(dt, 3)}seconds")
                t0 = time.time()
            if round(g * trainprop) > lastint:
                for i, m in enumerate(movekeys):
                    movekeys_.append(m)
                    wdllists_.append(wdllist[i])
                lastint = round(g * trainprop)
        return movekeys_, wdllists_

    def trainModel(self, generations: int, games: int, policy, trainprop: float = 0.9, batchsize: int = 32,
                   epochs: int = 5, startgen: int = 0):
        for gen in range(generations):
            # Reset training environment
            self.nodes = {}
            # Get all the raw training data
            t0 = time.time()
            movekeys_, wdllists_ = self.playGames(games, policy, trainprop)
            dt1 = round(time.time() - t0)
            print(f"Took {dt1 // 60}:{dt1 % 60} to play {games} games.")
            print(f"Average time per game: {round(dt1 / games, 2)} seconds.")
            x_train = []
            y_train = []
            t1 = time.time()
            for i, m in enumerate(movekeys_):
                x_train.append(decodeBoard(m))
                m_train = []
                if not self.nodes[m].expanded:
                    for l in range(9):
                        m_train.append(wdllists_[i])
                else:
                    for l in range(9):
                        if l in self.nodes[m].childkeys:
                            ck = self.nodes[m].childkeys[l]
                            m_train.append(self.nodes[ck].value())
                        else:
                            m_train.append(0)
                m_train.append(wdllists_[i])
                y_train.append(m_train)
            # Train the policy network
            dt = time.time() - t1
            print(f"{round(dt, 3)} seconds.")
            policy.fit(x_train, y_train, batch_size=batchsize, epochs=epochs)
            # Saves trained model before the next cycle
            policy.save(f"nacnn{startgen + gen + 1}.h5")

    def run(self, iterations: int, policy, board: list):
        pass


if __name__ == "__main__":
    tn = TrainNetwork()
    #policy = buildModel()f"
    policy = load_model("nacnn4.h5")
    tn.trainModel(1, 15000, policy, trainprop=1)




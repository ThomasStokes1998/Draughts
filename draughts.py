# Game rules
# regular pieces only move once diagonally forward
# promote to king if reach the end of the board
# kings can move in any direction diagonally once
# you must take if you can
# If you have no legal moves then you forfeit your turn
# If both players have no legal moves the match ends in a draw
# win by eliminating all of your opponents pieces
# draw if three repeat board positions or 20* consecutive moves with no pieces taken or no king promotion
# *might change number
# (draw logic not implemented)

import numpy as np

# Global variables
PLAYER1 = 1
PLAYER2 = -1
# For encoding board positions
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class Draughts:
    def __init__(self) -> None:
        self.board = [[PLAYER2, 0, PLAYER2, 0, PLAYER2, 0, PLAYER2, 0],
                      [0, PLAYER2, 0, PLAYER2, 0, PLAYER2, 0, PLAYER2],
                      [PLAYER2, 0, PLAYER2, 0, PLAYER2, 0, PLAYER2, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, PLAYER1, 0, PLAYER1, 0, PLAYER1, 0, PLAYER1],
                      [PLAYER1, 0, PLAYER1, 0, PLAYER1, 0, PLAYER1, 0],
                      [0, PLAYER1, 0, PLAYER1, 0, PLAYER1, 0, PLAYER1]]
        self.turn = PLAYER1
        self.pieces = [12, 12]
        self.playing = False
        self.valid_moves = ["LU", "LD", "RU", "RD"]

    def printBoard(self, board: list = None) -> None:
        pboard = {0: "   ", PLAYER1: " a ", PLAYER2: " b ", 2 * PLAYER1: " A ", 2 * PLAYER2: " B "}
        if board is None:
            board = self.board
        yaxis = "x|"
        for i in range(8):
            yaxis += f" {i} "
        print(yaxis + "|")
        for i, row in enumerate(board):
            rowstring = f"{i}|"
            for j, col in enumerate(row):
                if (i + j) % 2 == 0 and col == 0:
                    rowstring += " o "
                else:
                    rowstring += pboard[col]
            print(rowstring + "|")

    def evalBoard(self, player: int, board: list = None) -> float:
        if board is None:
            board = self.board
        score = 0
        for y, row in enumerate(board):
            for x, p in enumerate(row):
                if p == PLAYER1:
                    score += 1 + (7 - y) / 7
                elif p == PLAYER2:
                    score -= 1 + y / 7
                elif p == 2 * PLAYER1:
                    score += 2
                elif p == 2 * PLAYER2:
                    score -= 2

        return round(np.tanh(score * (PLAYER1 + PLAYER2 - 2 * player) / (4 * (PLAYER2 - PLAYER1))), 2)

    def encodeBoard(self, board=None, letters: str = "aAbBn", lcode: bool = False):
        if board is None:
            board = self.board
        # 32 character encoding
        longcode = ""
        for y, row in enumerate(board):
            for x, col in enumerate(row):
                if (x + y) % 2 == 0:
                    if col == PLAYER1:
                        longcode += letters[0]
                    elif col == 2 * PLAYER1:
                        longcode += letters[1]
                    elif col == PLAYER2:
                        longcode += letters[2]
                    elif col == 2 * PLAYER2:
                        longcode += letters[3]
                    elif col == 0:
                        longcode += letters[4]
        # 16 character encoding
        shortcode = ""
        for l in range(16):
            x = longcode[2 * l]
            y = longcode[2 * l + 1]
            i = 6
            j = 6
            for a in range(5):
                if letters[a] == x:
                    i = a
                if letters[a] == y:
                    j = a
                if i < 6 and j < 6:
                    break
            shortcode += ALPHABET[5 * i + j]
        if lcode:
            return shortcode, longcode
        else:
            return shortcode

    def decodeBoard(self, encoding: str, letters: str = "aAbBn"):
        newboard = []
        piece_dict = {0: PLAYER1, 1: 2 * PLAYER1, 2: PLAYER2, 3: 2 * PLAYER2, 4: 0}
        if len(encoding) == 16:
            for y in range(8):
                row = []
                for x in range(8):
                    if y % 2 == 0 and x % 4 == 0 or y % 2 == 1 and x % 4 == 1:
                        i = 2 * y + (x - (y % 2)) / 4
                        e = encoding[int(i)]
                        l = 0
                        for j in range(25):
                            if ALPHABET[j] == e:
                                l = j
                                break
                        s1 = l // 5
                        s2 = l % 5
                        if y % 2 == 0:
                            row.append(piece_dict[s1])
                            row.append(0)
                            row.append(piece_dict[s2])
                            row.append(0)
                        else:
                            row.append(0)
                            row.append(piece_dict[s1])
                            row.append(0)
                            row.append(piece_dict[s2])
                newboard.append(row)
        elif len(encoding) == 32:
            for y in range(8):
                row = []
                for x in range(8):
                    if (x + y) % 2 == 0:
                        i = 4 * y + (x - (y % 2)) / 2
                        e = encoding[int(i)]
                        l = 0
                        while letters[l] != e:
                            l += 1
                            if letters[l] == e:
                                break
                        row.append(piece_dict[l])
                    else:
                        row.append(0)
                newboard.append(row)
        return newboard

    def isValMove(self, piece: list, move_type: str):
        x = piece[0]
        y = piece[1]
        p = self.board[y][x]
        if "L" in move_type:
            x_ = -1
        else:
            x_ = 1
        if "U" in move_type:
            y_ = -1
        else:
            y_ = 1
        # Check if piece can move in that direction
        if self.turn == (PLAYER1 + PLAYER2 + (PLAYER1 - PLAYER2) * y_) / 2 and p == self.turn:
            return False, False
        # Check if piece will go off the board
        elif x == 7 * (1 + x_) / 2 or y == 7 * (1 + y_) / 2:
            return False, False
        # Check if move obstructed by friendly piece
        elif self.board[y + y_][x + x_] != 0 and (self.board[y + y_][x + x_] == self.turn
                                                  or self.board[y + y_][x + x_] == 2 * self.turn):
            return False, False
        # Check if a valid take
        elif self.board[y + y_][x + x_] in (PLAYER1 + PLAYER2 - self.turn, 2 * (PLAYER1 + PLAYER2 - self.turn)):
            if x == (5 * x_ + 7) / 2 or y == (5 * y_ + 7) / 2 or self.board[y + 2 * y_][x + 2 * x_] != 0:
                return False, True
            else:
                return True, True
        return True, False

    def legalMoves(self, player: int, board: list, piece=None) -> dict:
        legal_moves = {}
        l_take = False
        # Finds legal moves for a specific piece
        if piece is not None:
            l_take = True
            n = []
            for m in self.valid_moves:
                v, t = self.isValMove(piece, m)
                # If there is a take then the only legal moves are takes
                if v and l_take and t:
                    n.append(m)
            legal_moves[f"{piece[0]}{piece[1]}"] = n
        else:
            for y, row in enumerate(board):
                for x, col in enumerate(row):
                    if col != 0 and (col == player or col == 2 * player):
                        p = [x, y]
                        # List of legal moves for the piece
                        n = []
                        for m in self.valid_moves:
                            v, t = self.isValMove(p, m)
                            # If you can take you must
                            if v and t and not l_take:
                                # print(f"Entered loop with piece {p}, move {m}")
                                l_take = True
                                legal_moves = {}
                                n = []
                            # If there is a take then the only legal moves are takes
                            if v and l_take and t or v and not l_take:
                                n.append(m)
                        if len(n) > 0:
                            legal_moves[f"{x}{y}"] = n
        return legal_moves

    def move(self, piece: list, move_type: str, internalboard=None):
        if internalboard is None:
            internalboard = self.board
        if self.isValMove(piece, move_type):
            x = piece[0]
            y = piece[1]
            p = internalboard[y][x]
            if "L" in move_type:
                x_ = -1
            else:
                x_ = 1
            if "U" in move_type:
                y_ = -1
            else:
                y_ = 1
            # Checks players piece is being moved
            if p == self.turn or p == 2 * self.turn:
                # Empty Square
                if internalboard[y + y_][x + x_] == 0:
                    # Checks for king promotion
                    if y == (5 * self.turn + PLAYER2 - 6 * PLAYER1) / (PLAYER2 - PLAYER1) and p in (PLAYER1, PLAYER2):
                        internalboard[y + y_][x + x_] = p * 2
                    else:
                        internalboard[y + y_][x + x_] = p
                    # Turn current square to empty
                    internalboard[y][x] = 0
                    return 0, [x + x_, y + y_]
                # Taking a piece with a normal piece
                elif p in (PLAYER1, PLAYER2):
                    internalboard[y + y_][x + x_] = 0
                    self.pieces[round((PLAYER2 - self.turn) / (PLAYER2 - PLAYER1))] -= 1
                    # Checks for king promotion
                    if p == PLAYER1 and y + 2 * y_ == 0 or p == PLAYER2 and y + 2 * y_ == 7:
                        internalboard[y + 2 * y_][x + 2 * x_] = p * 2
                    else:
                        internalboard[y + 2 * y_][x + 2 * x_] = p
                    # Turn current square to empty
                    internalboard[y][x] = 0
                    # Find the new legal moves
                    l_m = self.legalMoves(self.turn, internalboard, [x + 2 * x_, y + 2 * y_])
                    n = l_m[f"{x + 2 * x_}{y + 2 * y_}"]
                    if len(n) == 0:
                        return 0, [x + 2 * x_, y + 2 * y_]
                    else:
                        return 1, [x + 2 * x_, y + 2 * y_]
                # Taking a piece with a king
                else:
                    internalboard[y + y_][x + x_] = 0
                    self.pieces[round((PLAYER2 - self.turn) / (PLAYER2 - PLAYER1))] -= 1
                    internalboard[y + 2 * y_][x + 2 * x_] = p
                    # Turn current square to empty
                    internalboard[y][x] = 0
                    # Find the new legal moves
                    l_m = self.legalMoves(self.turn, internalboard, [x + 2 * x_, y + 2 * y_])
                    n = l_m[f"{x + 2 * x_}{y + 2 * y_}"]
                    if len(n) == 0:
                        return 0, [x + 2 * x_, y + 2 * y_]
                    else:
                        return 1, [x + 2 * x_, y + 2 * y_]
        return 1, None

    def playGame(self) -> None:
        self.playing = True
        evalhist = []
        turn_count = 0
        nlm = 0
        while self.playing:
            p = None
            avail_moves = 1
            if self.turn == PLAYER1:
                evalhist.append(self.evalBoard(self.turn))
                turn_count += 1
            print(f"===== Player {self.turn} Turn {turn_count} Eval: {self.evalBoard(self.turn)} =====")
            self.printBoard()
            while avail_moves != 0:
                l_m = self.legalMoves(self.turn, self.board, p)
                # If player has no legal moves then skip turn
                if len(l_m) == 0:
                    print(f"No move info: p={p}, l_m={l_m}")
                    nlm += 1
                    break
                # If there is only one piece that can move the game auto selects it
                elif len(l_m) == 1:
                    p_ = [w for w in l_m][0]
                    piece = [int(p_[0]), int(p_[1])]
                    print(f"Only piece with a legal move is {piece}")
                    if nlm > 0:
                        nlm = 0
                # Asks player to input the piece they want to move
                else:
                    if nlm > 0:
                        nlm = 0
                    print(f"Legal pieces: {[w for w in l_m]}")
                    q = int(input("Select piece: "))
                    piece = [q // 10, q % 10]
                    while f"{piece[0]}{piece[1]}" not in l_m:
                        print(f"This piece has no legal moves select from: {[w for w in l_m]}")
                        q = int(input("Select piece: "))
                        piece = [q // 10, q % 10]
                # Select how the piece moves
                z = l_m[f"{piece[0]}{piece[1]}"]
                # If there is only one legal move then it is auto selected
                if len(z) == 1:
                    m = z[0]
                    print(f"Only legal move is {m}.")
                # Asks player to input the move they want if there are multiple legal moves
                else:
                    print(f"Legal moves: {z}")
                    m = input("Select move: ")
                    while m not in self.valid_moves:
                        print(f"This is not a legal move select from: {z}")
                        m = input("Select move: ")
                # Applies the move then outputs result
                avail_moves, p = self.move(piece, m)
            # End Conditions
            # One player has lost all their pieces
            if self.pieces[round((self.turn - PLAYER1) / (PLAYER2 - PLAYER1))] == 0:
                print(f"Game over player {PLAYER1 + PLAYER2 - self.turn} wins!")
                self.playing = False
            # Draw if both players have no legal moves (goes through two full cycles to make double sure)
            elif nlm == 4:
                print("Match ends in a draw because there are no legal moves for both players")
                self.playing = False
            # Start new turn
            else:
                self.turn = PLAYER1 + PLAYER2 - self.turn
        print(f"Evaluation History: {evalhist}")

    def getPossBoards(self, encodedboard: str, player: int) -> dict:
        defboard = self.decodeBoard(encodedboard)
        newboards = {}
        p = None
        avail_moves = 1
        lm = self.legalMoves(player, defboard, p)
        for piece in lm:
            for direction in lm[piece]:
                simboard = np.copy(defboard)
                newmove = piece + dir
                while avail_moves != 0:
                    avail_moves, p = self.move([int(piece[0]), int(piece[1])], direction, simboard)
                    # Trying to figure out what to do with multiple moves in a turn
                newboards[newmove] = self.encodeBoard(simboard)

        return newboards


if __name__ == "__main__":
    dr = Draughts()
    dr.playGame()

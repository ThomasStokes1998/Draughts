# Game rules
# regular pieces only move once diagonally forward
# promote to king if reach the end of the board
# kings can move in any direction diagonally once
# you must take if you can
# If you have no legal moves then you forfeit your turn
# If both players have no legal moves the match ends in a draw
# win by eliminating all of your opponents pieces
# draw if three repeat board positions or 20* consecutive moves with no pieces taken
# *might change number
# (draw logic not implemented)

# Global variables
PLAYER1 = 1
PLAYER2 = 2


class Draughts:
    def __init__(self) -> None:
        self.board = [[2, 0, 2, 0, 2, 0, 2, 0],
                      [0, 2, 0, 2, 0, 2, 0, 2],
                      [2, 0, 2, 0, 2, 0, 2, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0, 1, 0, 1],
                      [1, 0, 1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1, 0, 1]]
        self.turn = PLAYER1
        self.pieces = [12, 12]
        self.playing = False
        self.valid_moves = ["LU", "LD", "RU", "RD"]

    def printBoard(self) -> None:
        print("x", [i for i in range(8)])
        for i, row in enumerate(self.board):
            print(f"{i} {row}")

    def evalBoard(self, board: list, player: int) -> int:
        score = 0
        for y, row in enumerate(board):
            for x, col in enumerate(row):
                p = board[y][x]
                if p == player:
                    score += 1
                elif p == player + 2:
                    score += 2
                elif p == 3 - player:
                    score -= 1
                elif score == 5 - player:
                    score -= 2
        return score

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
        if (p - 1) // 2 == 0 and self.turn == (3 - y_) / 2:
            return False, False
        # Check if piece will go off the board
        elif x == 7 * (1 + x_) / 2 or y == 7 * (1 + y_) / 2:
            return False, False
        # Check if move obstructed
        elif self.board[y + y_][x + x_] != 0 and self.board[y + y_][x + x_] % 2 == self.turn % 2:
            return False, False
        # Check if a valid take
        elif self.board[y + y_][x + x_] == PLAYER1 + PLAYER2 - self.turn:
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
                    if col > 0 and col % 2 == player % 2:
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

    def move(self, piece: list, move_type: str):
        if self.isValMove(piece, move_type):
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
            # Checks players piece is being moved
            if p % 2 == self.turn % 2:
                # Empty Square
                if self.board[y + y_][x + x_] == 0:
                    # Checks for king promotion
                    if y == 5 * self.turn - 4 and (p - 1) // 2 == 0:
                        self.board[y + y_][x + x_] = p + 2
                    else:
                        self.board[y + y_][x + x_] = p
                    # Turn current square to empty
                    self.board[y][x] = 0
                    return 0, [x + x_, y + y_]
                # Taking a piece with a normal piece
                elif (p - 1) // 2 == 0:
                    self.board[y + y_][x + x_] = 0
                    self.pieces[2 - self.turn] -= 1
                    # Checks for king promotion
                    if y == 3 * self.turn - 1:
                        self.board[y + 2 * y_][x + 2 * x_] = p + 2
                    else:
                        self.board[y + 2 * y_][x + 2 * x_] = p
                    # Turn current square to empty
                    self.board[y][x] = 0
                    # Find the new legal moves
                    l_m = self.legalMoves(self.turn, self.board, [x + 2 * x_, y + 2 * y_])
                    n = l_m[f"{x + 2 * x_}{y + 2 * y_}"]
                    if len(n) == 0:
                        return 0, [x + 2 * x_, y + 2 * y_]
                    else:
                        return 1, [x + 2 * x_, y + 2 * y_]
                # Taking a piece with a king
                else:
                    self.board[y + y_][x + x_] = 0
                    self.pieces[2 - self.turn] -= 1
                    self.board[y + 2 * y_][x + 2 * x_] = p
                    # Turn current square to empty
                    self.board[y][x] = 0
                    # Find the new legal moves
                    l_m = self.legalMoves(self.turn, self.board, [x + 2 * x_, y + 2 * y_])
                    n = l_m[f"{x + 2 * x_}{y + 2 * y_}"]
                    if len(n) == 0:
                        return 0, [x + 2 * x_, y + 2 * y_]
                    else:
                        return 1, [x + 2 * x_, y + 2 * y_]
        return 1, None

    def playGame(self) -> None:
        self.playing = True
        turn_count = 0
        nlm = 0
        while self.playing:
            p = None
            avail_moves = 1
            if self.turn == 1:
                turn_count += 1
            print(f"===== Player {self.turn} Turn {turn_count} =====")
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
            if self.pieces[self.turn - 1] == 0:
                print(f"Game over player {3 - self.turn} wins!")
                self.playing = False
            # Draw if both players have no legal moves (goes through two full cycles to make double sure)
            elif nlm == 4:
                print("Match ends in a draw because there are no legal moves for both players")
                self.playing = False
            # Start new turn
            else:
                self.turn = 3 - self.turn


if __name__ == "__main__":
    Draughts().playGame()

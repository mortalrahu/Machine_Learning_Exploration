"""
This script contains the class Player
"""

from tile_game import Game, get_positive_integer
import time

class Player:
    """
       Represents a player in the game, capable of simulating moves and determining optimal strategies.

       The Player class interfaces with the Game class to simulate different moves on a copy of the game board.
       It uses a temporary board for simulations to decide the best move based on the current game state,
       aiming to maximize the number of connected tiles of the same color starting from the origin.

       Attributes:
           game (Game): An instance of the Game class representing the current game state.
           temp_board (list): A 2D list used to simulate moves without affecting the actual game board.
           n (int): The size of the board (n x n), inherited from the game instance.
           m (int): The number of available colors, inherited from the game instance.

       Methods:
           flood_fill(x, y, new_color, original_color): Simulates filling the connected area starting from (x, y) on the temp_board.
           temp_perform_move(color): Performs a move on the temporary board for simulation purposes.
           count_connected_tiles(x=0, y=0, checked=None): Counts tiles connected to the origin with the same color for a simulated state.
           find_largest_expansion_color(): Determines the color that, when chosen, maximizes the expansion of the connected component.
           play_game(): Executes the game strategy by continuously finding and performing the best moves until the game is over.
       """

    def __init__(self, game):
        """
            Initializes a Player object with a game instance.

            Args:
                game (Game): An instance of the Game class.

            Raises:
                TypeError: If the game is not an instance of Game.
            """

        if not isinstance(game, Game):
            raise TypeError("Expected a Game instance")

        self.game = game  # This is an instance of the Game class
        self.temp_board = [row[:] for row in self.game.board]
        self.n = self.game.n
        self.m = self.game.m

    def flood_fill(self, x, y, new_color, original_color):
        """
            A recursive method to flood fill the board starting from the tile at position (x, y).

            Args:
                x (int): The x-coordinate to start filling.
                y (int): The y-coordinate to start filling.
                new_color (int): The color to apply.
                original_color (int): The original color to change from.

            Raises:
                ValueError: If coordinates are out of bounds.
            """
        if x < 0 or x >= self.n or y < 0 or y >= self.n or self.temp_board[x][y] != original_color:
            return

        self.temp_board[x][y] = new_color
        self.flood_fill(x + 1, y, new_color, original_color)
        self.flood_fill(x - 1, y, new_color, original_color)
        self.flood_fill(x, y + 1, new_color, original_color)
        self.flood_fill(x, y - 1, new_color, original_color)

    def temp_perform_move(self, color):
        """
            Performs a temporary move on the board for simulation purposes.

            Args:
                color (int): The color to change the connected component to.

            Raises:
                ValueError: If the color is not within the valid range.
            """
        if not (1 <= color <= self.m):
            raise ValueError("Color out of allowed range")
        original_color = self.temp_board[0][0]
        if original_color != color:
            # Only manipulate the Origin Tile at (0,0)
            self.flood_fill(0, 0, color, original_color)

    def count_connected_tiles(self, x=0, y=0, checked=None):

        """
           Counts the number of tiles connected to the given tile that have the same color as the origin.

           Args:
               x (int): The x-coordinate to start counting from, default is 0.
               y (int): The y-coordinate to start counting from, default is 0.
               checked (set): A set of already checked tile coordinates.

           Returns:
               int: Number of connected tiles.
           """

        if checked is None:
            checked = set()

        if (x, y) in checked or x < 0 or x >= self.game.n or y < 0 or y >= self.game.n:
            return 0
        checked.add((x, y))

        # Only count tiles connected with the same color as the origin
        if self.temp_board[x][y] != self.temp_board[0][0]:
            return 0

        # Recursive flood count
        return 1 + self.count_connected_tiles(x + 1, y, checked) \
               + self.count_connected_tiles(x - 1, y, checked) \
               + self.count_connected_tiles(x, y + 1, checked) \
               + self.count_connected_tiles(x, y - 1, checked)

    def find_largest_expansion_color(self):

        """
            Finds the color that leads to the largest expansion from the origin.

            Returns:
                int: The color that maximizes the expansion.
            """

        # Temporary switch each possible color and count connected components
        max_tiles = 0
        best_color = self.game.board[0][0]

        original_color = self.game.board[0][0]
        for color in range(1, self.game.m + 1):
            if color == original_color:
                continue

            # Simulate the move but on the temp board not the original board, so the moves don't count, it's like thinking
            self.temp_board = [row[:] for row in self.game.board]
            self.temp_perform_move(color)
            count = self.count_connected_tiles()

            # Revert the board after simulation
            #self.game.board = temp_board

            if count > max_tiles:
                max_tiles = count
                best_color = color

        return best_color

    def play_game(self):
        """
        Play the game on the actual game board

        Returns:
             int : Number of moves taken to finish the game
        """
        moves = 0
        while not self.game.is_game_over(print_off=False): # Make it True to turn off print statements
            best_color = self.find_largest_expansion_color()
            self.game.perform_move(best_color)
            moves += 1
        return moves

if __name__ == '__main__':

    # Command Prompt Testing Automated Player's Game Play

    n = get_positive_integer("Enter board size n: ")
    m = get_positive_integer("Enter number of colors m: ")

    start_time = time.time()

    game = Game(n, m)

    #game.print_board()

    player = Player(game=game)

    moves = player.play_game()

    end_time = time.time()

    game_time = end_time - start_time
    print(f"Game time: {game_time} seconds")

"""
This script contains the class Game
"""

import random
import time

class Game:
    """
       A class representing the game of tile flood.

       Attributes:
           n (int): The size of the board (n x n).
           m (int): The number of colors available.
           board (list): A 2D list representing the game board.
           num_moves (int): The number of moves played.

       Methods:
           print_board(): Prints the current state of the board.
           flood_fill(x, y, new_color, original_color): Fills connected regions of the board from (x, y) with new_color.
           perform_move(color): Performs a move by changing the color of connected components from the origin.
           check_win(): Checks if all board tiles are the same color.
           is_game_over(): Prints game status and checks for a win condition.
       """

    def __init__(self, n, m, seed=None):
        """
        Initializes a Game with a board of size n x n and m colors.

        Args:
            n (int): The size of the board (must be a positive integer).
            m (int): The number of colors (must be a positive integer).

        Raises:
            TypeError: If n or m is not an integer.
            ValueError: If n or m is less than or equal to 0.
        """

        # Edge case
        if not isinstance(n, int) or not isinstance(m, int):
            raise TypeError("Both board size and number of colors must be integers.")
        # Edge case
        if n <= 0 or m <= 0:
            raise ValueError("Board size and number of colors must be greater than zero.")

        if seed is not None:
            random.seed(seed)
        self.n = n  # board dimensions n x n
        self.m = m  # number of colors
        self.board = [[random.randint(1, m) for _ in range(n)] for _ in range(n)] # board intialization with random colors
        self.num_moves = 0

    def print_board(self):
        """
        Prints the current state of the board.

        Each row of the board is printed as a series of numbers, where each number
        represents a color. Rows are printed one after the other, separated by new lines.
        """
        for row in self.board:
            print(' '.join(str(x) for x in row))
        print()

    def flood_fill(self, x, y, new_color, original_color):
        """
            Recursively fills all connected tiles starting from the tile at position (x, y) with a new color,
            provided they are of the original color.

            Args:
                x (int): The x-coordinate (row index) of the starting tile.
                y (int): The y-coordinate (column index) of the starting tile.
                new_color (int): The new color to apply to the connected component.
                original_color (int): The color of tiles to be changed.

            Postconditions:
                Modifies the board to fill all connected tiles of the original color starting
                from (x, y) with the new color.
            """

        if x < 0 or x >= self.n or y < 0 or y >= self.n or self.board[x][y] != original_color:
            return
        self.board[x][y] = new_color
        self.flood_fill(x + 1, y, new_color, original_color)
        self.flood_fill(x - 1, y, new_color, original_color)
        self.flood_fill(x, y + 1, new_color, original_color)
        self.flood_fill(x, y - 1, new_color, original_color)

    def perform_move(self, color):
        """
        Perform a move in the game by changing the color of all connected tiles from the origin.

        Args:
            color (int): The color to change the connected component to.

        Postcondition:
            Updates the board state and increments the move counter.
        """
        original_color = self.board[0][0]
        if original_color != color:
            # Only manipulate the Origin Tile at (0,0)
            self.flood_fill(0, 0, color, original_color)
        self.num_moves += 1

    def check_win(self):
        """
            Checks if all tiles on the board are the same color.

            Returns:
                bool: True if all tiles are of the same color, indicating a win; False otherwise.
            """
        first_color = self.board[0][0]
        return all(self.board[x][y] == first_color for x in range(self.n) for y in range(self.n))

    def is_game_over(self, print_off=False):
        """
        Checks the game state to determine if the game is over. It checks if a win condition
        is met or provides a prompt for the next move.

        Args:
            print_off (bool): Only prints the print statements if False

        Returns:
            bool: True if the game is over (all tiles are the same color); False if the game
                  should continue.
        """
        if self.check_win():
            print("GAME OVER !!!")
            print("Number of moves played : ", self.num_moves)
            return True
        else:
            if self.num_moves != 0:
                if not print_off:
                    print("Game not over yet, play your next move...")
            return False

def get_positive_integer(prompt):
    """
    Ensures the prompt given to the function is a postive integer

    Args:
        prompt: prompt by the user

    Returns:
         int: value of the prompt if it is a positive integere, else raise error messages
    """
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                raise ValueError("Please enter a positive integer.")
            return value
        except ValueError as e:
            print(f"Invalid input: {e}")

if __name__ == '__main__':

    # Command Prompt Testing Manual Game Play

    n = get_positive_integer("Enter board size n: ")
    m = get_positive_integer("Enter number of colors m: ")

    start_time = time.time()

    game = Game(n, m)
    game.print_board()

    while not game.is_game_over():
        try:
            next_color = get_positive_integer(f"Choose the next color (1-{m}): ")
            if next_color > m:
                print(f"Color must be between 1 and {m}, please try again.")
                continue
            game.perform_move(next_color)
            game.print_board()
        except ValueError as e:
            print(f"Error: {e}")

    end_time = time.time()

    game_time = end_time - start_time
    print(f"Game time: {game_time} seconds")

"""
This script contains unit tests for the class Game and Player
"""

from tile_game import Game
from automated_player import Player
import unittest


class TestGame(unittest.TestCase):
    """
    Unit tests for the Game class from the tile_game module.
    """

    def setUp(self):
        """
        Set up method for preparing the test fixture before exercising it.
        """
        self.game = Game(5, 3)  # Example: 5x5 board, 3 colors

    def test_board_initialization(self):
        """
        Test to ensure the board is initialized with correct dimensions.
        """
        self.assertEqual(len(self.game.board), 5)
        self.assertEqual(all(len(row) == 5 for row in self.game.board), True)

    def test_invalid_initialization(self):
        """
        Test initialization failures with invalid inputs.
        """
        with self.assertRaises(ValueError):
            Game(0, 3)
        with self.assertRaises(ValueError):
            Game(5, 0)
        with self.assertRaises(TypeError):
            Game("5", "3")

    def test_flood_fill_functionality(self):
        """
        Test the flood fill functionality to ensure it only fills the correct areas.
        """
        self.game.board = [[1, 1, 2, 2, 3],
                           [1, 1, 2, 3, 3],
                           [4, 1, 2, 3, 3],
                           [4, 4, 4, 3, 3],
                           [4, 4, 4, 4, 4]]
        self.game.flood_fill(0, 0, 3, 1)  # Change color 1 to 3 at the top-left corner
        self.assertEqual(self.game.board[0][0], 3)
        self.assertEqual(self.game.board[1][0], 3)
        self.assertEqual(self.game.board[0][1], 3)

    def test_perform_move(self):
        """
        Test that performing a move actually changes the board and increments move counter.
        """
        initial_color = self.game.board[0][0]
        new_color = (initial_color % self.game.m) + 1  # Ensure a different color
        self.game.perform_move(new_color)
        self.assertNotEqual(self.game.board[0][0], initial_color)
        self.assertEqual(self.game.num_moves, 1)

    def test_game_over(self):
        """
        Test the game over conditions are correctly identified.
        """
        self.game.board = [[1]*5]*5  # All tiles the same
        self.assertTrue(self.game.is_game_over())

    def test_minimal_board_size(self):
        """
        Test the game with the smallest possible board size.
        """
        game = Game(1, 3)
        self.assertEqual(len(game.board), 1)
        self.assertEqual(len(game.board[0]), 1)
        game.perform_move(game.board[0][0])  # Perform a move with the only available color
        self.assertTrue(game.is_game_over())

    def test_large_board_size(self):
        """
        Test the game with a very large board size to ensure no performance degradation or errors.
        """
        game = Game(100, 3)  # Large board with moderate number of colors
        self.assertEqual(len(game.board), 100)
        self.assertEqual(len(game.board[0]), 100)
        # Further tests could assess performance or handle specific large-scale logic

    def test_large_color_range(self):
        """
           Test the game with the unusual number of colors.
           """
        game = Game(20, 100)  # Increase board size significantly
        color_set = set(color for row in game.board for color in row)
        self.assertTrue(len(color_set) > 90)  # Adjust expectation based on empirical data

    def test_single_color(self):
        """
        Test the game with the minimum number of colors.
        """
        game_min_colors = Game(5, 1)  # All tiles will have the same color
        self.assertTrue(game_min_colors.is_game_over())  # Should be true at start


class TestPlayer(unittest.TestCase):
    """
    Unit tests for the Player class from the automated_player module.
    """

    def setUp(self):
        """
        Set up method for preparing the player test fixture before exercising it.
        """
        self.game = Game(5, 3)
        self.player = Player(self.game)

    def test_player_initialization(self):
        """
        Test player initialization and the initial state of temp_board.
        """
        for i in range(self.game.n):
            for j in range(self.game.n):
                self.assertEqual(self.game.board[i][j], self.player.temp_board[i][j])

    def test_find_largest_expansion_color(self):
        """
        Test that the player correctly identifies the color leading to the largest expansion.
        """
        self.game.board = [[2]*5]*5  # All tiles the same but one
        self.game.board[0][0] = 1
        best_color = self.player.find_largest_expansion_color()
        self.assertEqual(best_color, 2)

    def test_count_connected_tiles(self):
        """
        Test that counting connected tiles works correctly.
        """
        self.game.board = [[1] * 5 for _ in range(5)]  # Ensure all tiles are the same and independent of initialization
        self.player.temp_board = [row[:] for row in self.game.board]  # Re-initialize temp_board for safety
        count = self.player.count_connected_tiles()
        print("Counted tiles:", count)
        print("Board state at test:", self.player.temp_board)
        self.assertEqual(count, 25)  # Entire board should be connected

if __name__ == '__main__':
    unittest.main()

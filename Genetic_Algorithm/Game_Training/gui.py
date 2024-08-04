import tkinter as tk
from tile_game import Game
from automated_player import Player

class GameGUI:
    def __init__(self, game, player):
        self.game = game
        self.player = player
        self.root = tk.Tk()
        self.root.title("Tile Color Game")
        self.buttons = [[None for _ in range(game.n)] for _ in range(game.n)]
        self.setup_grid()

    def setup_grid(self):
        for i in range(self.game.n):
            for j in range(self.game.n):
                color = self.get_color(self.game.board[i][j])
                button = tk.Button(self.root, bg=color, width=10, height=3,
                                   command=lambda x=i, y=j: self.on_button_click(x, y))
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

    def on_button_click(self, x, y):
        chosen_color = self.game.board[x][y]
        self.game.perform_move(chosen_color)  # Direct call to game's perform_move
        self.update_colors()
        if self.game.is_game_over():
            print("Game Over! All tiles are the same color.")
            self.root.destroy()
        else:
            # Optionally simulate the next move by the player
            self.simulate_player_move()

    def simulate_player_move(self):
        best_color = self.player.find_largest_expansion_color()
        self.game.perform_move(best_color)
        self.update_colors()

    def update_colors(self):
        for i in range(self.game.n):
            for j in range(self.game.n):
                new_color = self.get_color(self.game.board[i][j])
                self.buttons[i][j].config(bg=new_color)

    def get_color(self, value):
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
        return colors[(value-1) % len(colors)]

    def start_game(self):
        self.root.mainloop()

# Usage
if __name__ == "__main__":
    game = Game(5, 3)
    player = Player(game)
    gui = GameGUI(game, player)
    gui.start_game()

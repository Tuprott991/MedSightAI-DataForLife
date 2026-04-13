import tkinter as tk
from tkinter import messagebox
import math

class CaroGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Caro Game - Five in a Row")
        self.root.resizable(False, False)
        
        # Game settings
        self.board_size = 20    
        self.cell_size = 40
        self.padding = 50
        self.stone_radius = 15
        
        # Game state
        self.board = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = "black"  # black starts first
        self.game_over = False
        self.move_history = []
        
        # Colors
        self.bg_color = "#DEB887"  # Burlywood
        self.line_color = "#8B4513"  # SaddleBrown
        self.black_stone = "#000000"
        self.white_stone = "#FFFFFF"
        self.hover_color = "#FFD700"  # Gold
        self.last_move_color = "#FF0000"  # Red
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg="#2C3E50")
        main_frame.pack(padx=10, pady=10)
        
        # Title and info panel
        info_frame = tk.Frame(main_frame, bg="#2C3E50")
        info_frame.pack(pady=(0, 10))
        
        title_label = tk.Label(
            info_frame, 
            text="🎮 CARO GAME 🎮", 
            font=("Arial", 24, "bold"),
            fg="#ECF0F1",
            bg="#2C3E50"
        )
        title_label.pack()
        
        # Status panel
        status_frame = tk.Frame(main_frame, bg="#34495E", relief=tk.RAISED, borderwidth=3)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(
            status_frame,
            text="⚫ Black's Turn",
            font=("Arial", 16, "bold"),
            fg="#ECF0F1",
            bg="#34495E",
            pady=10
        )
        self.status_label.pack()
        
        # Canvas for game board
        canvas_width = self.board_size * self.cell_size + 2 * self.padding
        canvas_height = self.board_size * self.cell_size + 2 * self.padding
        
        self.canvas = tk.Canvas(
            main_frame,
            width=canvas_width,
            height=canvas_height,
            bg=self.bg_color,
            highlightthickness=2,
            highlightbackground="#8B4513"
        )
        self.canvas.pack()
        
        # Button panel
        button_frame = tk.Frame(main_frame, bg="#2C3E50")
        button_frame.pack(pady=(10, 0))
        
        self.new_game_btn = tk.Button(
            button_frame,
            text="🔄 New Game",
            font=("Arial", 12, "bold"),
            bg="#27AE60",
            fg="white",
            activebackground="#229954",
            activeforeground="white",
            relief=tk.RAISED,
            borderwidth=3,
            padx=20,
            pady=10,
            command=self.new_game,
            cursor="hand2"
        )
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        self.undo_btn = tk.Button(
            button_frame,
            text="↶ Undo",
            font=("Arial", 12, "bold"),
            bg="#E67E22",
            fg="white",
            activebackground="#D35400",
            activeforeground="white",
            relief=tk.RAISED,
            borderwidth=3,
            padx=20,
            pady=10,
            command=self.undo_move,
            cursor="hand2"
        )
        self.undo_btn.pack(side=tk.LEFT, padx=5)
        
        quit_btn = tk.Button(
            button_frame,
            text="❌ Quit",
            font=("Arial", 12, "bold"),
            bg="#E74C3C",
            fg="white",
            activebackground="#C0392B",
            activeforeground="white",
            relief=tk.RAISED,
            borderwidth=3,
            padx=20,
            pady=10,
            command=self.root.quit,
            cursor="hand2"
        )
        quit_btn.pack(side=tk.LEFT, padx=5)
        
        # Draw board
        self.draw_board()
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_hover)
        self.hover_circle = None
        
    def draw_board(self):
        """Draw the game board grid"""
        self.canvas.delete("all")
        
        # Draw grid lines
        for i in range(self.board_size):
            # Vertical lines
            x = self.padding + i * self.cell_size
            self.canvas.create_line(
                x, self.padding,
                x, self.padding + (self.board_size - 1) * self.cell_size,
                fill=self.line_color,
                width=2
            )
            
            # Horizontal lines
            y = self.padding + i * self.cell_size
            self.canvas.create_line(
                self.padding, y,
                self.padding + (self.board_size - 1) * self.cell_size, y,
                fill=self.line_color,
                width=2
            )
        
        # Draw star points (decorative dots)
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_points:
            x = self.padding + col * self.cell_size
            y = self.padding + row * self.cell_size
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill=self.line_color,
                outline=self.line_color
            )
        
        # Redraw all stones
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col]:
                    self.draw_stone(row, col, self.board[row][col])
        
        # Highlight last move
        if self.move_history:
            last_row, last_col = self.move_history[-1]
            x = self.padding + last_col * self.cell_size
            y = self.padding + last_row * self.cell_size
            self.canvas.create_oval(
                x - 6, y - 6, x + 6, y + 6,
                outline=self.last_move_color,
                width=3,
                tags="last_move"
            )
    
    def draw_stone(self, row, col, color):
        """Draw a stone at the specified position"""
        x = self.padding + col * self.cell_size
        y = self.padding + row * self.cell_size
        
        # Shadow effect
        shadow_offset = 2
        self.canvas.create_oval(
            x - self.stone_radius + shadow_offset,
            y - self.stone_radius + shadow_offset,
            x + self.stone_radius + shadow_offset,
            y + self.stone_radius + shadow_offset,
            fill="#555555",
            outline="#555555",
            tags=f"stone_{row}_{col}"
        )
        
        # Main stone
        stone_color = self.black_stone if color == "black" else self.white_stone
        outline_color = "#333333" if color == "white" else "#000000"
        
        self.canvas.create_oval(
            x - self.stone_radius,
            y - self.stone_radius,
            x + self.stone_radius,
            y + self.stone_radius,
            fill=stone_color,
            outline=outline_color,
            width=2,
            tags=f"stone_{row}_{col}"
        )
        
        # Highlight effect for white stones
        if color == "white":
            self.canvas.create_oval(
                x - self.stone_radius + 3,
                y - self.stone_radius + 3,
                x - self.stone_radius + 8,
                y - self.stone_radius + 8,
                fill="#FFFFFF",
                outline="#FFFFFF",
                tags=f"stone_{row}_{col}"
            )
    
    def on_hover(self, event):
        """Show hover effect"""
        if self.game_over:
            return
        
        # Remove previous hover circle
        if self.hover_circle:
            self.canvas.delete(self.hover_circle)
            self.hover_circle = None
        
        # Get grid position
        col = round((event.x - self.padding) / self.cell_size)
        row = round((event.y - self.padding) / self.cell_size)
        
        # Check if position is valid and empty
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            if self.board[row][col] is None:
                x = self.padding + col * self.cell_size
                y = self.padding + row * self.cell_size
                
                # Draw hover circle
                self.hover_circle = self.canvas.create_oval(
                    x - self.stone_radius - 2,
                    y - self.stone_radius - 2,
                    x + self.stone_radius + 2,
                    y + self.stone_radius + 2,
                    outline=self.hover_color,
                    width=3,
                    tags="hover"
                )
    
    def on_click(self, event):
        """Handle mouse click"""
        if self.game_over:
            return
        
        # Get grid position
        col = round((event.x - self.padding) / self.cell_size)
        row = round((event.y - self.padding) / self.cell_size)
        
        # Check if position is valid
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return
        
        # Check if position is empty
        if self.board[row][col] is not None:
            return
        
        # Place stone
        self.place_stone(row, col)
    
    def place_stone(self, row, col):
        """Place a stone and check for win"""
        # Update board
        self.board[row][col] = self.current_player
        self.move_history.append((row, col))
        
        # Draw stone
        self.draw_board()
        
        # Check for win
        if self.check_win(row, col):
            self.game_over = True
            winner = "Black" if self.current_player == "black" else "White"
            self.status_label.config(text=f"🎉 {winner} Wins! 🎉")
            
            # Show winner dialog and auto-reset
            def show_win_and_reset():
                messagebox.showinfo(
                    "Game Over",
                    f"🎉 {winner} Player Wins! 🎉\n\nCongratulations!\n\nStarting new game..."
                )
                self.new_game()
            
            self.root.after(100, show_win_and_reset)
            return
        
        # Check for draw
        if len(self.move_history) == self.board_size * self.board_size:
            self.game_over = True
            self.status_label.config(text="🤝 Draw!")
            
            # Show draw dialog and auto-reset
            def show_draw_and_reset():
                messagebox.showinfo(
                    "Game Over",
                    "🤝 It's a Draw!\n\nThe board is full!\n\nStarting new game..."
                )
                self.new_game()
            
            self.root.after(100, show_draw_and_reset)
            return
        
        # Switch player
        self.current_player = "white" if self.current_player == "black" else "black"
        turn_text = "⚫ Black's Turn" if self.current_player == "black" else "⚪ White's Turn"
        self.status_label.config(text=turn_text)
    
    def check_win(self, row, col):
        """Check if the current move wins the game"""
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal \
            (1, -1)   # Diagonal /
        ]
        
        for dr, dc in directions:
            count = 1  # Count the stone just placed
            
            # Check in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r][c] == self.current_player:
                    count += 1
                    r += dr
                    c += dc
                else:
                    break
            
            # Check in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r][c] == self.current_player:
                    count += 1
                    r -= dr
                    c -= dc
                else:
                    break
            
            # Win condition: 5 or more in a row
            if count >= 5:
                return True
        
        return False
    
    def undo_move(self):
        """Undo the last move"""
        if not self.move_history or self.game_over:
            return
        
        # Remove last move
        row, col = self.move_history.pop()
        self.board[row][col] = None
        
        # Switch player back
        self.current_player = "white" if self.current_player == "black" else "black"
        turn_text = "⚫ Black's Turn" if self.current_player == "black" else "⚪ White's Turn"
        self.status_label.config(text=turn_text)
        
        # Redraw board
        self.draw_board()
    
    def new_game(self):
        """Start a new game"""
        # Reset game state
        self.board = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = "black"
        self.game_over = False
        self.move_history = []
        
        # Update UI
        self.status_label.config(text="⚫ Black's Turn")
        self.draw_board()


def main():
    root = tk.Tk()
    root.configure(bg="#2C3E50")
    game = CaroGame(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()

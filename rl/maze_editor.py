# maze_editor.py
import pygame
import numpy as np
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Maze Editor')
    parser.add_argument('--size', type=int, default=10, help='Size of the maze (N x N)')
    parser.add_argument('--out_path', type=str, default=None, help='Path to save the maze file')
    return parser.parse_args()

class MazeEditor:
    def __init__(self, size=10, out_path=None):
        pygame.init()
        self.size = size
        self.cell_size = 50
        self.screen_size = self.size * self.cell_size
        self.out_path = out_path
        
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Maze Editor")
        
        # Initialize empty maze
        self.maze = np.zeros((size, size))
        self.start_pos = None
        self.goal_pos = None
        
        # Colors
        self.colors = {
            'wall': (0, 0, 0),
            'empty': (255, 255, 255),
            'start': (0, 255, 0),
            'goal': (255, 0, 0),
            'grid': (200, 200, 200)
        }
        
        # Current drawing mode
        self.modes = ['wall', 'start', 'goal']
        self.current_mode = 0
        
        # Help text
        self.help_text = [
            "Controls:",
            "Left Click: Draw/Remove walls",
            "Space: Switch mode (wall/start/goal)",
            "R: Clear maze",
            "S: Save maze",
            "Q: Quit",
            f"Current mode: {self.modes[self.current_mode]}"
        ]
        
    def draw_grid(self):
        self.screen.fill(self.colors['empty'])
        
        # Draw cells
        for i in range(self.size):
            for j in range(self.size):
                color = self.colors['wall'] if self.maze[i,j] == 1 else self.colors['empty']
                
                if (i, j) == self.start_pos:
                    color = self.colors['start']
                elif (i, j) == self.goal_pos:
                    color = self.colors['goal']
                    
                pygame.draw.rect(self.screen, color,
                               (j*self.cell_size, i*self.cell_size,
                                self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.colors['grid'],
                               (j*self.cell_size, i*self.cell_size,
                                self.cell_size, self.cell_size), 1)
                
        pygame.display.flip()
        
    def get_cell_coords(self, pos):
        x, y = pos
        return y // self.cell_size, x // self.cell_size
        
    def save_maze(self, filename):
        with open(filename, 'w') as f:
            f.write(f"{self.size}\n")
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) == self.start_pos:
                        f.write('S')
                    elif (i, j) == self.goal_pos:
                        f.write('G')
                    else:
                        f.write('#' if self.maze[i,j] == 1 else '.')
                f.write('\n')
                
    def clear_maze(self):
        """Clear the maze to empty state"""
        self.maze = np.zeros((self.size, self.size))
        self.start_pos = None
        self.goal_pos = None

    def draw_help(self):
        """Draw help text"""
        font = pygame.font.Font(None, 24)
        y_offset = 10
        for text in self.help_text:
            surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (self.screen_size + 10, y_offset))
            y_offset += 25

    def handle_wall_click(self, i, j):
        """Toggle wall state when clicked"""
        self.maze[i,j] = 0 if self.maze[i,j] == 1 else 1

    def run(self):
        print("Maze Editor")
        print('\n'.join(self.help_text))
        drawing = False
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    drawing = True
                    i, j = self.get_cell_coords(event.pos)
                    if 0 <= i < self.size and 0 <= j < self.size:
                        mode = self.modes[self.current_mode]
                        
                        if mode == 'wall':
                            self.handle_wall_click(i, j)
                        elif mode == 'start' and self.start_pos != (i,j):
                            self.start_pos = (i,j)
                            self.maze[i,j] = 0
                        elif mode == 'goal' and self.goal_pos != (i,j):
                            self.goal_pos = (i,j)
                            self.maze[i,j] = 0
                    
                elif event.type == pygame.MOUSEBUTTONUP:
                    drawing = False
                    
                elif event.type == pygame.MOUSEMOTION and drawing:
                    i, j = self.get_cell_coords(event.pos)
                    if 0 <= i < self.size and 0 <= j < self.size:
                        mode = self.modes[self.current_mode]
                        
                        if mode == 'wall':
                            self.handle_wall_click(i, j)
                        elif mode == 'start' and self.start_pos != (i,j):
                            self.start_pos = (i,j)
                            self.maze[i,j] = 0
                        elif mode == 'goal' and self.goal_pos != (i,j):
                            self.goal_pos = (i,j)
                            self.maze[i,j] = 0
                            
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Switch mode
                        self.current_mode = (self.current_mode + 1) % len(self.modes)
                        self.help_text[-1] = f"Current mode: {self.modes[self.current_mode]}"
                        print(f"Current mode: {self.modes[self.current_mode]}")
                    elif event.key == pygame.K_s:
                        # Save maze
                        if self.out_path:
                            self.save_maze(self.out_path)
                            print(f"Maze saved to {self.out_path}")
                        else:
                            # Save to mazes/NxN.txt
                            filename = f"mazes/{self.size}x{self.size}.txt"
                            self.save_maze(filename)
                            print(f"Maze saved to {filename}")
                    elif event.key == pygame.K_r:
                        # Clear maze
                        self.clear_maze()
                    elif event.key == pygame.K_q:
                        running = False
                        
            self.draw_grid()
            
        pygame.quit()

if __name__ == "__main__":
    args = parse_args()
    editor = MazeEditor(size=args.size)
    editor.run()
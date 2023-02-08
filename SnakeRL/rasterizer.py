import tkinter as tk
from environment import Field


class Drawer(tk.Tk):
    def __init__(self, x, y, line_width, name, field: Field):
        tk.Tk.__init__(self)
        self.x_size = x
        self.y_size = y
        self.line_width = line_width
        self.geometry(str(self.x_size + self.line_width * 2) + "x" + str(self.y_size + self.line_width * 2))
        self.draw_field = field
        self.title(name)

        self.cell_count_x = self.draw_field.base.shape[0]
        self.cell_count_y = self.draw_field.base.shape[1]

        self.x_step = self.x_size / self.cell_count_x
        self.y_step = self.y_size / self.cell_count_y

        self.canvas = tk.Canvas(self,
                                width=self.x_size + self.line_width,
                                height=self.y_size + self.line_width,
                                bg='white', )

        self.canvas.pack()
        self.draw_grid(line_width=self.line_width)
        self.canvas.pack()
        self.bind('<Left>', self.leftKey)
        self.bind('<Right>', self.rightKey)
        self.bind('<Up>', self.upKey)
        self.bind('<Down>', self.downKey)

    def draw_grid(self, line_width):

        self.canvas.create_rectangle(line_width,
                                     line_width,
                                     self.x_size + line_width,
                                     self.y_size + line_width,
                                     outline='black', width=self.line_width)

        for i in range(self.cell_count_x):
            self.canvas.create_line(self.x_step * i + line_width,
                                    line_width,
                                    self.x_step * i + line_width,
                                    self.y_size + line_width, fill="black", width=line_width)
        for i in range(self.cell_count_y):
            self.canvas.create_line(line_width,
                                    self.y_step * i + line_width,
                                    self.x_size + line_width,
                                    self.y_step * i + line_width, fill="black", width=line_width)

    def refresh(self):
        self.canvas.delete("all")
        self.draw_grid(self.line_width)
        self.draw_state()

    def leftKey(self, event):
        print("Left")
        self.draw_field.snake.go_left()

    def rightKey(self, event):
        print("Right")
        self.draw_field.snake.go_right()

    def upKey(self, event):
        print("Up")
        self.draw_field.snake.go_up()

    def downKey(self, event):
        print("Down")
        self.draw_field.snake.go_down()

    def draw_state(self):
        for i in range(self.cell_count_x):
            for j in range(self.cell_count_y):
                if self.draw_field.base[i, j] == 1:
                    self.canvas.create_rectangle(i * self.x_step + self.line_width,
                                                 j * self.y_step + self.line_width,
                                                 i * self.x_step + self.x_step + self.line_width,
                                                 j * self.y_step + self.y_step + self.line_width, outline="black",
                                                 fill="black")
                if self.draw_field.base[i, j] == 2:
                    self.canvas.create_oval(i * self.x_step + self.line_width,
                                            j * self.y_step + self.line_width,
                                            i * self.x_step + self.x_step + self.line_width,
                                            j * self.y_step + self.y_step + self.line_width, outline="black",
                                            fill="black")
                if self.draw_field.base[i, j] == 3:
                    self.canvas.create_oval(i * self.x_step + self.line_width-5,
                                            j * self.y_step + self.line_width-5,
                                            i * self.x_step + self.x_step + self.line_width+5,
                                            j * self.y_step + self.y_step + self.line_width+5, outline="black",
                                            fill="black")
                self.canvas.pack()

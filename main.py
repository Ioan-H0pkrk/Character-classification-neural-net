import numpy as np
from MNIST_reader import MnistDataloader
from os.path import join
from NeuronVisualiser import NetworkVisualizer

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Pixel canvas
class PixelDrawer:
    def __init__(self, nn, cell_size=30):
        self.nn = nn

        self.grid_size = 28
        self.cell_size = cell_size

        self.pixels = np.zeros((28, 28), dtype=float)

        self.root = tk.Tk()
        self.root.title("28x28 Pixel Drawer")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        canvas_size = self.grid_size * self.cell_size

        self.canvas = tk.Canvas(
            self.left_frame,
            width=canvas_size,
            height=canvas_size,
            bg="white"
        )
        self.canvas.pack()

        self.rects = []

        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                rect = self.canvas.create_rectangle(
                    x * self.cell_size,
                    y * self.cell_size,
                    (x + 1) * self.cell_size,
                    (y + 1) * self.cell_size,
                    fill="white",
                    outline="gray"
                )
                row.append(rect)
            self.rects.append(row)

        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        clear_button = tk.Button(self.left_frame, text="Clear", command=self.clear)
        clear_button.pack(fill="x")

        predict_button = tk.Button(self.left_frame, text="Predict digit", command=self.predict)
        predict_button.pack(fill="x")

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax_bar = self.fig.add_subplot(111)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plot(np.zeros(10))

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def draw(self, event):
        x = event.x // self.cell_size
        y = event.y // self.cell_size

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.pixels[y, x] = 1.0
            self.canvas.itemconfig(self.rects[y][x], fill="black")
        
    def update_plot(self, output):
        self.ax_bar.clear()

        self.ax_bar.bar(range(10), output)
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_xticks(range(10))
        self.ax_bar.set_xlabel("Digit", fontsize=12)
        self.ax_bar.set_ylabel("Probability", fontsize=12)
        self.ax_bar.set_title(f"Prediction: {np.argmax(output)}", fontsize=14)
        self.ax_bar.tick_params(axis="both", labelsize=11)

        self.fig.tight_layout()
        self.mpl_canvas.draw_idle()

    def clear(self):
        self.pixels[:, :] = 0.0

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.canvas.itemconfig(self.rects[y][x], fill="white")

        self.update_plot(np.zeros(10))

    def predict(self):
        image = self.pixels.ravel()
        output = self.nn.forward(image)
        prediction = np.argmax(output)

        self.update_plot(output)

        print("prediction:", prediction)
        print("output:", np.vstack(np.array([x for x in enumerate(output)]).round(2)))

    def run(self):
        self.root.mainloop()
    
    def close(self):
        self.root.quit()
        self.root.destroy()

# Loads training data
input_path = r"C:\Users\i_hopkirk22\Downloads\mnist"

training_images_filepath = join(input_path, "train-images.idx3-ubyte") 
training_labels_filepath = join(input_path, "train-labels.idx1-ubyte")
test_images_filepath = join(input_path, "t10k-images.idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels.idx1-ubyte")

mnist_dataloader = MnistDataloader(
    training_images_filepath,    
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath
)

x_train, y_train = mnist_dataloader.load_train()
x_test, y_test = mnist_dataloader.load_test()
     

# Neural network class definition
class NeuralNetwork:
    def __init__(self, input_width, hidden_layer_widths, a_funcs, output_width):
        self.input_width = input_width
        self.hidden_widths = hidden_layer_widths
        self.output_width = output_width

        layer_widths = [input_width] + hidden_layer_widths + [output_width]

        self.weights = []
        self.biases = []

        self.activations = []
        self.zs = []

        for previous_width, current_width in zip(layer_widths[:-1], layer_widths[1:]):
            self.weights.append(np.random.randn(current_width, previous_width))
            self.biases.append(np.random.randn(current_width))
        
        a_funcs_dict = {
            "sigmoid": self.sigmoid,
            "relu": self.relu,
            "tanh": np.tanh,
            "softmax": self.softmax
        }
        self.a_funcs = [a_funcs_dict[x] for x in a_funcs]

        d_funcs_dict = {
            "sigmoid": self.sigmoid_derivative,
            "relu": self.relu_derivative,
            "tanh": self.tanh_derivative,
            "softmax": None
        }
        self.d_funcs = [d_funcs_dict[x] for x in a_funcs]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0.0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def tanh_derivative(self, a):
        return 1 - a ** 2

    def relu_derivative(self, z):
        return (z > 0).astype(float)


    def forward(self, inputs):
        activation = np.array(inputs).ravel()
        
        self.activations = [activation]
        self.zs = []

        for i, (weights, bias, a_func) in enumerate(
            zip(self.weights, self.biases, self.a_funcs)
        ):
            z = np.dot(weights, activation) + bias

            self.zs.append(z)

            if i == len(self.weights) - 1:
                activation = self.softmax(z)
            else:
                activation = a_func(z)

            self.activations.append(activation)

        return activation
    
    def train(self, inputs, target, learning_rate=0.01):
        output = self.forward(inputs)
        target = np.array(target)

        loss = self.loss(output, target)

        # softmax + cross-entropy output delta
        delta = output - target

        for layer in range(len(self.weights) - 1, -1, -1):
            previous_activation = self.activations[layer]

            dW = np.outer(delta, previous_activation)
            db = delta

            if layer > 0:
                next_delta = self.weights[layer].T @ delta

                previous_activation_output = self.activations[layer]
                previous_z = self.zs[layer - 1]
                derivative_func = self.d_funcs[layer - 1]

                if derivative_func == self.relu_derivative:
                    next_delta *= derivative_func(previous_z)
                else:
                    next_delta *= derivative_func(previous_activation_output)
            else:
                next_delta = None

            self.weights[layer] -= learning_rate * dW
            self.biases[layer] -= learning_rate * db

            delta = next_delta

        return loss

    def one_hot(self, correct_index):
        target = np.zeros(self.output_width)
        target[correct_index] = 1
        return target
    
    def loss(self, output, target):
        output = np.clip(output, 1e-12, 1.0 - 1e-12)
        return -np.sum(target * np.log(output))


def main():
    # Inputted data
    # Expects: input_width {int}, hidden_layer_widths {list(int)}, a_funcs {list(str)}, output_width {int}, inputs {list(int)}
    in_width = 784
    hidden_layer_widths = [16, 16, 16, 12]
    a_funcs = ["sigmoid", "tanh", "sigmoid", "tanh", "softmax"]
    out_width = 10

    nn = NeuralNetwork(in_width, hidden_layer_widths, a_funcs, out_width)

    visualizer = NetworkVisualizer(
        layer_sizes=[in_width] + hidden_layer_widths + [out_width],
        max_neurons_per_layer=16
    )

    visualizer.update(nn.weights)

    print("Successfully created neural net\n")

    target = nn.one_hot(y_train[0])
    image = np.array(x_train[0]).ravel() / 255.0
    
    # An epoch is an entire pass of a training set
    for epoch in range(20):
        total_loss = 0

        for image, label in zip(x_train[:1000], y_train[:1000]):
            image = np.array(image).ravel() / 255.0
            target = nn.one_hot(label)

            total_loss += nn.train(image, target, learning_rate=0.1)

        print("epoch:", epoch, "loss:", total_loss / 1000)

        visualizer.update(nn.weights)

    mode = int(input("\nWhat do you want to do?\n1: Draw\n2: Use a test image\n\n\t"))

    match mode:
        case 1:
            drawer = PixelDrawer(nn)
            drawer.run()
        case 2:
            test_n = int(input("Pick a number from 0 to 6000: "))
            image = np.array(x_test[test_n]).ravel() / 255.0

            output = nn.forward(image)
            prediction = np.argmax(output)

            visualizer.update(nn.weights)

            fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8, 3))

            # Left: test image
            ax_img.imshow(np.array(x_test[test_n]), cmap="gray")
            ax_img.set_title(f"Actual: {y_test[test_n]}")
            ax_img.axis("off")

            # Right: prediction probabilities
            ax_bar.bar(range(10), output)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_xticks(range(10))
            ax_bar.set_xlabel("Digit")
            ax_bar.set_ylabel("Probability")
            ax_bar.set_title(f"Prediction: {np.argmax(output)}")

            plt.tight_layout()
            
            print("prediction:", prediction)
            print("actual:", y_test[test_n])
            print("accuracy:", round(output[y_test[test_n]], 2) * 100, "%")
            print("output:", np.vstack(np.array([x for x in enumerate(output)]).round(2)))

            plt.show()
            return
        
main()

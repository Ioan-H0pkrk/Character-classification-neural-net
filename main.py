import numpy as np
from random import randrange
from MNIST_reader import MnistDataloader
from NeuronVisualiser import NetworkVisualizer

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import yaml
import time

# Pixel canvas
class PixelDrawer:
    def __init__(self, nn, cell_size=15):
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

# Taken from stackoverflow to find the bounding box of non-zero data in a matrix
def bbox(img: np.ndarray):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax+1, xmin:xmax+1]

def randmove(img: np.ndarray):
    bound_y = img.shape[0]
    bound_x = img.shape[1]
    img = bbox(img)
    img = np.array(img)

    row_space = bound_y - img.shape[0]
    col_space = bound_x - img.shape[1]

    if row_space == 0:
        n_cols_left = randrange(0, col_space + 1) # Chooses how many cols to insert to the left
        n_cols_right = col_space - n_cols_left
        img = np.pad(img, ((0, 0), (n_cols_left, n_cols_right)), mode="constant")
        return img
    elif col_space == 0:
        n_rows_up = randrange(0, row_space + 1) # Chooses how many rows to insert above
        n_rows_down = row_space - n_rows_up
        img = np.pad(img, ((n_rows_up, n_rows_down), (0, 0)), mode="constant")
        return img
    elif row_space == 0 and col_space == 0:
        return img
    
    n_rows_up = randrange(0, row_space + 1) # Chooses how many rows to insert above
    n_cols_left = randrange(0, col_space + 1) # Chooses how many cols to insert to the left
    n_rows_down = row_space - n_rows_up
    n_cols_right = col_space - n_cols_left

    img = np.pad(img, ((n_rows_up, n_rows_down), (n_cols_left, n_cols_right)), mode="constant")

    return img

# Neural network class definition
class NeuralNetwork:
    def __init__(self, input_width, hidden_layer_widths, a_funcs, output_width, weights=None, biases=None):
        self.input_width = input_width
        self.hidden_widths = hidden_layer_widths
        self.output_width = output_width

        layer_widths = [input_width] + hidden_layer_widths + [output_width]

        expected_layers = len(layer_widths) - 1

        if len(a_funcs) != expected_layers:
            raise ValueError(
                f"Expected {expected_layers} activation functions, got {len(a_funcs)}: {a_funcs}"
            )

        if weights is None or weights == []:
            self.weights = [
                np.random.randn(current_width, previous_width) * np.sqrt(2 / previous_width)
                for previous_width, current_width in zip(layer_widths[:-1], layer_widths[1:])
            ]
        else:
            self.weights = weights

        if biases is None or biases == []:
            self.biases = [
                np.zeros(current_width)
                for current_width in layer_widths[1:]
            ]
        else:
            self.biases = biases

        self.activations = []
        self.zs = []

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

# Loads training data
training_images_filepath = "mnist/train-images.idx3-ubyte"
training_labels_filepath = "mnist/train-labels.idx1-ubyte"
test_images_filepath = "mnist/t10k-images.idx3-ubyte"
test_labels_filepath = "mnist/t10k-labels.idx1-ubyte"

mnist_dataloader = MnistDataloader(
    training_images_filepath,    
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath
)

x_train, y_train = mnist_dataloader.load_train()
x_test, y_test = mnist_dataloader.load_test()

x_train_flat = np.array([np.array(img).ravel() / 255.0 for img in x_train])
x_test_flat = np.array([np.array(img).ravel() / 255.0 for img in x_test])

# Loads existing values + config file
config_filepath = "config.yml"
weights_filepath = "weights.npy"
biases_filepath = "biases.npy"

def main():
    try:
        with open(config_filepath) as file:
            config = yaml.safe_load(file)
    except OSError:
        with open(config_filepath, 'w') as outfile:
            data = dict(
                in_width = 784,
                hidden_layer_widths = [32, 64, 32, 16],
                a_funcs = ["sigmoid", "tanh", "sigmoid", "tanh", "softmax"],
                out_width = 10
            )
            config = data
            yaml.dump(data, outfile, default_flow_style=False)

    weights = []
    biases = []
    option = str(input("Load existing neural net (Y/N)? "))
    match option:
        case "Y" | "y":
            in_width = config["in_width"]
            hidden_layer_widths = config["hidden_layer_widths"]
            a_funcs = config["a_funcs"]
            out_width = config["out_width"]

            try:
                weights = np.load(weights_filepath, allow_pickle=True).tolist()
            except (EOFError, FileNotFoundError):
                layer_widths = [in_width] + hidden_layer_widths + [out_width]
                weights = [
                    np.random.randn(current_width, previous_width) * np.sqrt(2 / previous_width)
                    for previous_width, current_width in zip(layer_widths[:-1], layer_widths[1:])
                ]
                np.save(weights_filepath, np.array(weights, dtype=object), allow_pickle=True)
            try:
                biases = np.load(biases_filepath, allow_pickle=True).tolist()
            except (EOFError, FileNotFoundError):
                layer_widths = [in_width] + hidden_layer_widths + [out_width]
                biases = [
                    np.zeros(current_width)
                    for current_width in layer_widths[1:]
                ]
                np.save(biases_filepath, np.array(biases, dtype=object), allow_pickle=True)
                
        case "N" | "n":
            # Inputted data
            # Expects: input_width {int}, hidden_layer_widths {list(int)}, a_funcs {list(str)}, output_width {int}, inputs {list(int)}
            with open(config_filepath, 'w') as outfile:
                in_width = int(input("In_width: "))
                hidden_layer_widths = list(map(int, input("Hidden_layer_widths: ").split()))
                a_funcs = input("Activation_functions: ").split()
                out_width = int(input("Out_width: "))

                data = dict(
                    in_width = in_width,
                    hidden_layer_widths = hidden_layer_widths,
                    a_funcs = a_funcs,
                    out_width = out_width
                )
                config = data

                yaml.dump(data, outfile, default_flow_style=False)
        
        case _:
            raise ValueError("Option must be Y or N")

    nn = NeuralNetwork(in_width, hidden_layer_widths, a_funcs, out_width, weights, biases)

    print("Successfully (re)created neural net\n")

    show_visual = True
    option = str(input("Show neural net visualiser (Y/N)? "))

    match option:
        case "Y" | "y":
            visualizer = NetworkVisualizer(
                layer_sizes=[in_width] + hidden_layer_widths + [out_width],
                max_neurons_per_layer=16
            )

            visualizer.update(nn.weights)
        case "N" | "n":
            show_visual = False
    
    epoch_count = int(input("Epoch train count: "))
    learn_rate = float(input("Learning rate: "))
    train_count = int(input(f"Training image count, max {len(x_train)}: "))
    
    total_start = time.perf_counter()

    # An epoch is an entire pass of a training set
    for epoch in range(epoch_count):
        epoch_start = time.perf_counter()

        total_loss = 0

        indices = np.random.permutation(len(x_train_flat))[:train_count]

        for i in indices:
            #image = randmove(x_train_flat[i])
            image = np.array(x_train_flat[i])

            target = nn.one_hot(y_train[i])
            total_loss += nn.train(image, target, learning_rate=learn_rate)

        epoch_time = time.perf_counter() - epoch_start

        print(
            "epoch:", epoch,
            "loss:", total_loss / train_count,
            "time:", round(epoch_time, 2), "s"
        )

        if show_visual and epoch % 5 == 0: 
            visualizer.update(nn.weights)

    total_time = time.perf_counter() - total_start

    print("total training time:", round(total_time, 2), "s")
    print("average epoch time:", round(total_time / epoch_count, 2), "s")

    np.save(weights_filepath, np.array(nn.weights, dtype=object), allow_pickle=True)
    np.save(biases_filepath, np.array(nn.biases, dtype=object), allow_pickle=True)

    correct = 0
    total = 1000

    for image, label in zip(x_test[:total], y_test[:total]):
        image = np.array(image).ravel() / 255.0
        output = nn.forward(image)
        prediction = np.argmax(output)

        if prediction == label:
            correct += 1

    print("test accuracy:", correct / total * 100, "%")

    mode = int(input("\nWhat do you want to do?\n1: Draw a number\n2: Use a test image\n\n\t"))

    match mode:
        case 1:
            drawer = PixelDrawer(nn)
            drawer.run()
        case 2:
            test_n = int(input("Pick a number from 0 to 9999: "))
            image = np.array(x_test[test_n]).ravel() / 255.0

            output = nn.forward(image)
            prediction = np.argmax(output)

            if show_visual: visualizer.update(nn.weights)

            _, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8, 3))

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
            
            print("prediction:", prediction)
            print("actual:", y_test[test_n])
            print("accuracy:", round(output[y_test[test_n]], 2) * 100, "%")
            print("output:", np.vstack(np.array([x for x in enumerate(output)]).round(2)))

            plt.tight_layout()
            plt.show()

            plt.close("all")
            return
        
main()

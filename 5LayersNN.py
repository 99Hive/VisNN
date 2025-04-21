import tkinter as tk
from tkinter import ttk
import numpy as np
import math

class NeuralNetworkVisualization:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Visualization")
        self.root.geometry("1200x900")

        # Scrollable canvas setup
        self.main_canvas = tk.Canvas(self.root)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.main_canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.main_canvas.configure(yscrollcommand=v_scrollbar.set)

        self.scrollable_frame = ttk.Frame(self.main_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )

        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Network attributes
        self.layers = [2, 4, 3, 4, 1]
        self.weights = []
        self.biases = []
        self.activations = []
        self.inputs = [0.5, 0.8]
        self.active_layer = 0
        self.show_details = True
        self.animation_speed = 1000
        self.is_animating = False
        self.activation_function = "sigmoid"

        # Setup UI in scrollable frame
        self.setup_ui()
        self.initialize_network()
        self.draw_network()

    def setup_ui(self):
        main_frame = ttk.Frame(self.scrollable_frame, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.forward_button = ttk.Button(button_frame, text="Forward Step", command=self.calculate_next_layer)
        self.forward_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_network)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.animate_button = ttk.Button(button_frame, text="Auto-Forward", command=self.start_animation)
        self.animate_button.pack(side=tk.LEFT, padx=5)

        speed_frame = ttk.Frame(button_frame)
        speed_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_combo = ttk.Combobox(speed_frame, values=["Slow", "Medium", "Fast"], width=10)
        self.speed_combo.current(1)
        self.speed_combo.pack(side=tk.LEFT, padx=5)
        self.speed_combo.bind("<<ComboboxSelected>>", self.update_speed)

        activation_frame = ttk.Frame(button_frame)
        activation_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(activation_frame, text="Activation:").pack(side=tk.LEFT)
        self.activation_combo = ttk.Combobox(activation_frame, values=["sigmoid", "relu", "tanh", "linear"], width=10)
        self.activation_combo.current(0)
        self.activation_combo.pack(side=tk.LEFT, padx=5)
        self.activation_combo.bind("<<ComboboxSelected>>", self.update_activation)

        details_frame = ttk.Frame(control_frame)
        details_frame.pack(fill=tk.X, pady=5)
        self.show_details_var = tk.BooleanVar(value=True)
        details_check = ttk.Checkbutton(details_frame, text="Show Details", variable=self.show_details_var, 
                                        command=self.toggle_details)
        details_check.pack(side=tk.LEFT)

        input_frame = ttk.LabelFrame(control_frame, text="Input Values", padding=5)
        input_frame.pack(fill=tk.X, pady=5)
        self.input_vars = []
        self.input_labels = []

        for i in range(len(self.inputs)):
            input_row = ttk.Frame(input_frame)
            input_row.pack(fill=tk.X, pady=2)
            ttk.Label(input_row, text=f"Input {i+1}:").pack(side=tk.LEFT, padx=5)
            input_var = tk.DoubleVar(value=self.inputs[i])
            self.input_vars.append(input_var)
            input_slider = ttk.Scale(input_row, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                     variable=input_var, command=lambda event, idx=i: self.update_input(idx))
            input_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            input_label = ttk.Label(input_row, text=f"{self.inputs[i]:.1f}", width=10)
            input_label.pack(side=tk.LEFT, padx=5)
            self.input_labels.append(input_label)

        self.canvas_frame = ttk.LabelFrame(main_frame, text="Neural Network", padding=10)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, background="white", width=1200, height=700)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        self.details_frame = ttk.LabelFrame(main_frame, text="Computation Details", padding=10)
        self.details_frame.pack(fill=tk.BOTH, pady=5)
        self.details_text = tk.Text(self.details_frame, height=6, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        details_scrollbar = ttk.Scrollbar(self.details_text, orient=tk.VERTICAL, command=self.details_text.yview)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)

        help_frame = ttk.LabelFrame(main_frame, text="How to Use", padding=10)
        help_frame.pack(fill=tk.X, pady=5)
        help_text = """
        1. Adjust the input values using the sliders
        2. Click \"Forward Step\" to propagate the data through one layer at a time
        3. Click \"Auto-Forward\" to automatically step through all layers
        4. Toggle \"Show Details\" to see computation formulas
        5. Try different activation functions to see how they transform the data

        Blue circles represent positive values, red represents negative. The intensity of the color indicates the magnitude of the value.
        Line thickness represents the weight magnitude.
        """
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT, wraplength=900)
        help_label.pack(fill=tk.X)
    def initialize_network(self):
        # Initialize random weights and biases for each layer
        self.weights = []
        self.biases = []
        self.activations = []
        
        # Input layer activations (start with the input values)
        self.activations.append(self.inputs.copy())
        
        # For each subsequent layer
        for i in range(1, len(self.layers)):
            # Weights for connections from previous layer to this layer
            layer_weights = []
            for j in range(self.layers[i-1]):
                neuron_weights = []
                for k in range(self.layers[i]):
                    # Random weight between -1 and 1
                    neuron_weights.append(round(np.random.uniform(-1, 1), 2))
                layer_weights.append(neuron_weights)
            self.weights.append(layer_weights)
            
            # Biases for this layer
            layer_biases = []
            for j in range(self.layers[i]):
                # Random bias between -1 and 1
                layer_biases.append(round(np.random.uniform(-1, 1), 2))
            self.biases.append(layer_biases)
            
            # Initialize activations for this layer to zeros
            self.activations.append([0] * self.layers[i])
    
    def update_input(self, index):
        # Update the input value and its display
        value = self.input_vars[index].get()
        self.inputs[index] = value
        self.input_labels[index].config(text=f"{value:.1f}")
        
        # Update the first layer of activations
        self.activations[0][index] = value
        
        # Reset network state
        self.reset_network(maintain_inputs=True)
    
    def update_speed(self, event=None):
        # Update the animation speed based on the dropdown selection
        selection = self.speed_combo.get()
        if selection == "Slow":
            self.animation_speed = 2000
        elif selection == "Medium":
            self.animation_speed = 1000
        elif selection == "Fast":
            self.animation_speed = 500
    
    def update_activation(self, event=None):
        # Update the activation function based on the dropdown selection
        self.activation_function = self.activation_combo.get()
        # Reset the network to see the effect of the new activation function
        self.reset_network()
    
    def toggle_details(self):
        # Toggle the visibility of details
        self.show_details = self.show_details_var.get()
        # Redraw the network to show/hide details
        self.draw_network()
    
    def apply_activation(self, x):
        # Apply the selected activation function to the input value
        if self.activation_function == "sigmoid":
            return 1 / (1 + math.exp(-x))
        elif self.activation_function == "relu":
            return max(0, x)
        elif self.activation_function == "tanh":
            return math.tanh(x)
        else:  # linear
            return x
    
    def calculate_next_layer(self):
        # Calculate the next layer's activations if not at the output layer
        if self.active_layer >= len(self.layers) - 1:
            return
        
        # For each neuron in the next layer
        for i in range(self.layers[self.active_layer + 1]):
            # Start with the bias
            sum_val = self.biases[self.active_layer][i]
            
            # Add weighted inputs from previous layer
            for j in range(self.layers[self.active_layer]):
                sum_val += self.activations[self.active_layer][j] * self.weights[self.active_layer][j][i]
            
            # Apply activation function and store the result
            self.activations[self.active_layer + 1][i] = round(self.apply_activation(sum_val), 3)
        
        # Move to the next layer
        self.active_layer += 1
        
        # Update the visualization
        self.draw_network()
        self.update_details()
        
        # Disable forward button if we've reached the output layer
        if self.active_layer >= len(self.layers) - 1:
            self.forward_button.config(state=tk.DISABLED)
    
    def reset_network(self, maintain_inputs=False):
        # Reset the network state to initial
        self.active_layer = 0
        self.forward_button.config(state=tk.NORMAL)
        
        # Reset activations but keep input values
        if not maintain_inputs:
            # Reset all activations to zeros except inputs
            for i in range(1, len(self.activations)):
                self.activations[i] = [0] * self.layers[i]
        else:
            # Only reset hidden and output layers
            for i in range(1, len(self.activations)):
                self.activations[i] = [0] * self.layers[i]
            
            # Update first layer with current input values
            for i in range(len(self.inputs)):
                self.activations[0][i] = self.inputs[i]
        
        # Stop any ongoing animation
        self.is_animating = False
        
        # Update the visualization
        self.draw_network()
        self.update_details()
    
    def start_animation(self):
        # Start the automatic forward propagation
        self.reset_network(maintain_inputs=True)
        self.is_animating = True
        self.animate_step()
    
    def animate_step(self):
        # Perform one step of the animation and schedule the next if needed
        if self.is_animating and self.active_layer < len(self.layers) - 1:
            self.calculate_next_layer()
            # Schedule the next step
            self.root.after(self.animation_speed, self.animate_step)
        else:
            self.is_animating = False
    
    def get_color(self, value):
        # Generate a color based on the activation value
        if value == 0:
            return "#e0e0e0"  # Light gray for zero
        
        # Calculate color intensity based on value magnitude
        intensity = min(abs(float(value)) * 2, 1)
        
        if float(value) > 0:
            # Blue for positive values
            color = self.rgb_to_hex(int(0), int(128 * intensity), int(255 * intensity))
        else:
            # Red for negative values
            color = self.rgb_to_hex(int(255 * intensity), int(77 * intensity), int(77 * intensity))
        
        return color
    
    def rgb_to_hex(self, r, g, b):
        # Convert RGB values to hex color code
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def get_line_width(self, weight):
        # Calculate line width based on weight value
        return max(min(abs(float(weight)) * 3, 5), 0.5)
    
    def get_line_color(self, weight):
        # Generate a color for the connection line based on weight value
        intensity = min(abs(float(weight)), 1)
        
        if float(weight) > 0:
            # Blue for positive weights
            color = self.rgb_to_hex(int(0), int(128 * intensity), int(255 * intensity))
        else:
            # Red for negative weights
            color = self.rgb_to_hex(int(255 * intensity), int(77 * intensity), int(77 * intensity))
        
        return color
    
    def draw_network(self):
        # Clear the canvas
        self.canvas.delete("all")
        
        # Configure the canvas size based on network dimensions
        canvas_width = 300 * len(self.layers)
        canvas_height = 150 * max(self.layers)
        self.canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))

        
        # Calculate size of neurons and positions
        neuron_radius = 20
        layer_x_spacing = 150
        neuron_y_spacing = 80
        
        # For each layer
        for layer_idx, layer_size in enumerate(self.layers):
            layer_x = 100 + layer_idx * layer_x_spacing
            
            # Draw layer label
            if layer_idx == 0:
                layer_name = "Input Layer"
            elif layer_idx == len(self.layers) - 1:
                layer_name = "Output Layer"
            else:
                layer_name = f"Hidden Layer {layer_idx}"
            
            self.canvas.create_text(layer_x, 30, text=layer_name, font=("Arial", 12, "bold"))
            
            # Mark active layer
            if layer_idx == self.active_layer:
                self.canvas.create_text(layer_x, 50, text="(Active)", fill="green", font=("Arial", 10, "bold"))
            
            # Draw each neuron in the layer
            for neuron_idx in range(layer_size):
                # Calculate neuron position
                y_offset = (canvas_height - (layer_size - 1) * neuron_y_spacing) / 2
                neuron_y = y_offset + neuron_idx * neuron_y_spacing
                
                # Get neuron activation value and color
                value = self.activations[layer_idx][neuron_idx] if layer_idx < len(self.activations) else 0
                color = self.get_color(value)
                
                # Draw neuron circle
                self.canvas.create_oval(
                    layer_x - neuron_radius, neuron_y - neuron_radius,
                    layer_x + neuron_radius, neuron_y + neuron_radius,
                    fill=color, outline="black", width=2
                )
                
                # Draw activation value
                self.canvas.create_text(layer_x, neuron_y, text=f"{value:.3f}" if isinstance(value, float) else value)
                
                # Draw connections to next layer
                if layer_idx < len(self.layers) - 1:
                    next_layer_x = layer_x + layer_x_spacing
                    next_layer_size = self.layers[layer_idx + 1]
                    
                    for next_neuron_idx in range(next_layer_size):
                        # Calculate next neuron position
                        next_y_offset = (canvas_height - (next_layer_size - 1) * neuron_y_spacing) / 2
                        next_neuron_y = next_y_offset + next_neuron_idx * neuron_y_spacing
                        
                        # Get weight and determine line properties
                        if layer_idx < len(self.weights):
                            weight = self.weights[layer_idx][neuron_idx][next_neuron_idx]
                            line_width = self.get_line_width(weight)
                            line_color = self.get_line_color(weight)
                            opacity = 1.0 if layer_idx < self.active_layer else 0.2
                            
                            # Draw connection line
                            line = self.canvas.create_line(
                                layer_x + neuron_radius, neuron_y,
                                next_layer_x - neuron_radius, next_neuron_y,
                                width=line_width, fill=line_color
                            )
                            
                            # Adjust opacity by changing the state
                            if opacity < 1.0:
                                self.canvas.itemconfig(line, state='disabled')
                
                # Draw neuron bias (for non-input layers)
                if layer_idx > 0 and self.show_details:
                    bias = self.biases[layer_idx - 1][neuron_idx]
                    
                    # Create a small circle for bias
                    bias_x = layer_x - neuron_radius * 0.8
                    bias_y = neuron_y - neuron_radius * 0.8
                    
                    self.canvas.create_oval(
                        bias_x - 8, bias_y - 8,
                        bias_x + 8, bias_y + 8,
                        fill="yellow", outline="orange"
                    )
                    
                    self.canvas.create_text(bias_x, bias_y, text="b", font=("Arial", 8))
    
    def update_details(self):
        # Update the computation details text area
        if not self.show_details or self.active_layer <= 0:
            self.details_text.delete(1.0, tk.END)
            return
        
        # Clear existing text
        self.details_text.delete(1.0, tk.END)
        
        # Add header
        self.details_text.insert(tk.END, f"Layer {self.active_layer} Computation Details:\n\n", "header")
        
        # For each neuron in the current layer
        for neuron_idx in range(self.layers[self.active_layer]):
            # Add neuron header
            self.details_text.insert(tk.END, f"Neuron {neuron_idx + 1}:\n", "neuron_header")
            
            # Add formula
            formula = f"  Value = {self.activation_function}("
            
            for prev_idx in range(self.layers[self.active_layer - 1]):
                if prev_idx > 0:
                    formula += " + "
                
                prev_value = self.activations[self.active_layer - 1][prev_idx]
                weight = self.weights[self.active_layer - 1][prev_idx][neuron_idx]
                
                formula += f"{prev_value} × {weight}"
            
            bias = self.biases[self.active_layer - 1][neuron_idx]
            formula += f" + {bias} (bias))\n"
            
            self.details_text.insert(tk.END, formula)
            
            # Calculate the weighted sum
            weighted_sum = 0
            for prev_idx in range(self.layers[self.active_layer - 1]):
                prev_value = self.activations[self.active_layer - 1][prev_idx]
                weight = self.weights[self.active_layer - 1][prev_idx][neuron_idx]
                weighted_sum += prev_value * weight
            
            # Add the bias
            weighted_sum += bias
            
            # Apply activation function
            result = self.activations[self.active_layer][neuron_idx]
            
            # Add numerical result
            numerical = f"  = {self.activation_function}({weighted_sum:.3f}) = {result}\n\n"
            self.details_text.insert(tk.END, numerical)
        
        # Configure tags
        self.details_text.tag_configure("header", font=("Arial", 12, "bold"))
        self.details_text.tag_configure("neuron_header", font=("Arial", 10, "bold"))

# Run the application
# This block ensures that the code below runs only when the script is executed directly,
# and not when it is imported as a module in another script.
if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkVisualization(root)
    root.mainloop()

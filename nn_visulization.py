import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Set random seed for reproducible results
np.random.seed(42)

# Network architecture
input_size = 2
hidden_size = 2
output_size = 1

# Initialize weights and biases
W1 = np.random.rand(hidden_size, input_size) * 2 - 1  # Hidden layer weights
b1 = np.random.rand(hidden_size, 1) * 2 - 1           # Hidden layer biases
W2 = np.random.rand(output_size, hidden_size) * 2 - 1 # Output layer weights
b2 = np.random.rand(output_size, 1) * 2 - 1           # Output layer biases

# Sample input
X = np.array([[0.5], [0.8]])

print("🧠 NEURAL NETWORK FORWARD PASS")
print("=" * 50)

# Forward pass calculations
print("\n📥 INPUT LAYER:")
print(f"x1 = {X[0,0]:.3f}")
print(f"x2 = {X[1,0]:.3f}")

print(f"\n⚖️ WEIGHTS & BIASES:")
print(f"W1 = \n{W1}")
print(f"b1 = \n{b1}")
print(f"W2 = \n{W2}")
print(f"b2 = \n{b2}")

# Hidden layer calculations
print(f"\n🔢 HIDDEN LAYER CALCULATIONS:")
z1 = np.dot(W1, X) + b1
print(f"z1 = W1 @ X + b1 = \n{z1}")

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

a1 = sigmoid(z1)
print(f"a1 = sigmoid(z1) = \n{a1}")

print(f"\nDetailed hidden layer calculations:")
print(f"h1: z = ({W1[0,0]:.3f} * {X[0,0]:.3f}) + ({W1[0,1]:.3f} * {X[1,0]:.3f}) + {b1[0,0]:.3f} = {z1[0,0]:.3f}")
print(f"    a = sigmoid({z1[0,0]:.3f}) = {a1[0,0]:.3f}")
print(f"h2: z = ({W1[1,0]:.3f} * {X[0,0]:.3f}) + ({W1[1,1]:.3f} * {X[1,0]:.3f}) + {b1[1,0]:.3f} = {z1[1,0]:.3f}")
print(f"    a = sigmoid({z1[1,0]:.3f}) = {a1[1,0]:.3f}")

# Output layer calculations
print(f"\n📤 OUTPUT LAYER CALCULATIONS:")
z2 = np.dot(W2, a1) + b2
a2 = sigmoid(z2)

print(f"z2 = W2 @ a1 + b2 = {z2[0,0]:.3f}")
print(f"output = sigmoid(z2) = sigmoid({z2[0,0]:.3f}) = {a2[0,0]:.3f}")

print(f"\nDetailed output calculation:")
print(f"z = ({W2[0,0]:.3f} * {a1[0,0]:.3f}) + ({W2[0,1]:.3f} * {a1[1,0]:.3f}) + {b2[0,0]:.3f} = {z2[0,0]:.3f}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Network Architecture Plot
def draw_network(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture & Forward Pass', fontsize=16, fontweight='bold')
    
    # Node positions
    input_pos = [(1, 6), (1, 2)]
    hidden_pos = [(5, 6), (5, 2)]
    output_pos = [(9, 4)]
    
    # Draw connections with weights
    colors = ['red', 'blue', 'green', 'orange']
    color_idx = 0
    
    # Input to Hidden connections
    for i, inp in enumerate(input_pos):
        for j, hid in enumerate(hidden_pos):
            ax.plot([inp[0], hid[0]], [inp[1], hid[1]], 
                   color=colors[color_idx % len(colors)], linewidth=2, alpha=0.7)
            # Weight labels
            mid_x, mid_y = (inp[0] + hid[0])/2, (inp[1] + hid[1])/2
            ax.text(mid_x, mid_y, f'{W1[j,i]:.2f}', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   fontsize=8, ha='center')
            color_idx += 1
    
    # Hidden to Output connections
    for i, hid in enumerate(hidden_pos):
        ax.plot([hid[0], output_pos[0][0]], [hid[1], output_pos[0][1]], 
               color=colors[color_idx % len(colors)], linewidth=2, alpha=0.7)
        # Weight labels
        mid_x = (hid[0] + output_pos[0][0])/2
        mid_y = (hid[1] + output_pos[0][1])/2
        ax.text(mid_x, mid_y, f'{W2[0,i]:.2f}', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
               fontsize=8, ha='center')
        color_idx += 1
    
    # Draw nodes
    # Input nodes
    for i, pos in enumerate(input_pos):
        circle = plt.Circle(pos, 0.4, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f'x{i+1}\n{X[i,0]:.3f}', ha='center', va='center', 
               fontweight='bold', fontsize=10)
    
    # Hidden nodes
    for i, pos in enumerate(hidden_pos):
        circle = plt.Circle(pos, 0.4, color='lightgreen', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f'h{i+1}\n{a1[i,0]:.3f}', ha='center', va='center', 
               fontweight='bold', fontsize=10)
        # Bias
        ax.text(pos[0], pos[1]-0.8, f'b={b1[i,0]:.2f}', ha='center', va='center', 
               fontsize=8, style='italic')
    
    # Output node
    circle = plt.Circle(output_pos[0], 0.4, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(output_pos[0][0], output_pos[0][1], f'y\n{a2[0,0]:.3f}', 
           ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(output_pos[0][0], output_pos[0][1]-0.8, f'b={b2[0,0]:.2f}', 
           ha='center', va='center', fontsize=8, style='italic')
    
    # Layer labels
    ax.text(1, 0.5, 'INPUT\nLAYER', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 0.5, 'HIDDEN\nLAYER', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(9, 0.5, 'OUTPUT\nLAYER', ha='center', va='center', fontsize=12, fontweight='bold')

# Activation Function Plot
def plot_activations(ax):
    x = np.linspace(-5, 5, 100)
    y_sigmoid = sigmoid(x)
    
    ax.plot(x, y_sigmoid, 'b-', linewidth=3, label='Sigmoid')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Input (z)', fontsize=12)
    ax.set_ylabel('Output', fontsize=12)
    ax.set_title('Sigmoid Activation Function', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    
    # Mark our specific points
    ax.scatter([z1[0,0], z1[1,0], z2[0,0]], 
              [a1[0,0], a1[1,0], a2[0,0]], 
              color='red', s=100, zorder=5)
    
    # Annotations
    ax.annotate(f'h1: ({z1[0,0]:.2f}, {a1[0,0]:.2f})', 
                xy=(z1[0,0], a1[0,0]), xytext=(z1[0,0]+1, a1[0,0]+0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.annotate(f'h2: ({z1[1,0]:.2f}, {a1[1,0]:.2f})', 
                xy=(z1[1,0], a1[1,0]), xytext=(z1[1,0]+1, a1[1,0]+0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.annotate(f'output: ({z2[0,0]:.2f}, {a2[0,0]:.2f})', 
                xy=(z2[0,0], a2[0,0]), xytext=(z2[0,0]+1, a2[0,0]-0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

# Draw both plots
draw_network(ax1)
plot_activations(ax2)

plt.tight_layout()
plt.show()

# Summary
print(f"\n📊 SUMMARY:")
print(f"Input: [{X[0,0]:.3f}, {X[1,0]:.3f}]")
print(f"Hidden Layer Output: [{a1[0,0]:.3f}, {a1[1,0]:.3f}]")
print(f"Final Output: {a2[0,0]:.3f}")
print(f"\n🎯 The network processed the input through:")
print(f"   1. Linear transformation (weights + bias)")
print(f"   2. Sigmoid activation")
print(f"   3. Final prediction: {a2[0,0]:.3f}")

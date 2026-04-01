import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim

# Target data: cosine function
x = np.linspace(-2*np.pi, 2*np.pi, 200)
y = np.cos(x)

X = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Simple neural network
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop with recording
epochs = 200
predictions = []
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    
    predictions.append(output.detach().numpy().flatten())
    losses.append(loss.item())

# --- Animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

# Left plot: cosine vs prediction
ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(-1.5, 1.5)
ax1.set_title("NN learning cosine")
line_target, = ax1.plot(x, y, 'r-', label="Target cos(x)")
line_pred, = ax1.plot([], [], 'b-', label="Prediction")
ax1.legend()

# Right plot: loss curve
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, max(losses))
ax2.set_title("Loss decreasing")
line_loss, = ax2.plot([], [], 'g-')

def init():
    line_pred.set_data([], [])
    line_loss.set_data([], [])
    return line_pred, line_loss

def update(frame):
    # Update prediction curve
    line_pred.set_data(x, predictions[frame])
    # Update loss curve
    line_loss.set_data(range(frame+1), losses[:frame+1])
    return line_pred, line_loss

ani = animation.FuncAnimation(
    fig, update, frames=epochs, init_func=init,
    blit=True, interval=100, repeat=False
)

# plt.show()
ani.save("nn_cosine_learning.gif", writer="pillow")


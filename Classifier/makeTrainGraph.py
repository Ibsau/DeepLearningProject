import matplotlib.pyplot as plt
import re

# Read from the uploaded file
file_path = '/mnt/data/TrainAccValAcc.txt'
with open(file_path, 'r') as file:
    raw_text = file.read()

# Use regex to extract the data
pattern = r"Epoch (\d+)/\d+\s+Train Loss: ([\d\.]+), Train Acc: ([\d\.]+)%\s+Val Loss: ([\d\.]+),\s+Val Acc: ([\d\.]+)%"
matches = re.findall(pattern, raw_text)

# Extract values
epochs = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for match in matches:
    epoch, train_loss, train_acc, val_loss, val_acc = match
    epochs.append(int(epoch))
    train_losses.append(float(train_loss))
    train_accs.append(float(train_acc))
    val_losses.append(float(val_loss))
    val_accs.append(float(val_acc))

# Create the plots
plt.figure(figsize=(14, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, val_accs, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


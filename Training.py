
# ========================== Training ===============================


base_path = '/lustre/scratch/mqsemi/train_eegs'
target_path = '/lustre/scratch/mqsemi/df.csv'
save_dir = '/home/mqsemi/results/cnn/'


df = pd.read_csv(target_path)
newdf = df[df['target'].isin(['LPD', 'Seizure'])]
# df['target'].value_counts()
ids = newdf['eeg_id'].astype(str).tolist()
# ids = df['eeg_id'].astype(str).tolist()

train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=42)

train_dataset = CNNDataset(train_ids, base_path, target_path, augment=False)
val_dataset = CNNDataset(val_ids, base_path, target_path, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# ========================== Training ===============================


def validate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Skip None batches
            if inputs is None or labels is None:
                continue

            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.long()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute validation metrics
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy


# Define the model, loss function, and optimizer
model = CNN_Transformer_Classifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

criterion = nn.CrossEntropyLoss()  # Loss function for classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from torch.optim.lr_scheduler import LambdaLR

epochs = 200


def lr_lambda(epoch):
    warmup_epochs = 15  # Number of warm-up epochs
    max_lr = 5e-3  # Maximum learning rate
    min_lr = 1e-6  # Minimum learning rate

    if epoch < warmup_epochs:
        # Linear warm-up
        return (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return min_lr / max_lr + (1 - min_lr / max_lr) * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress)))


# Define the scheduler
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

model.to(device)

from torch.cuda.amp import autocast, GradScaler

# Initialize the scaler before your training loop (just after model definition)
scaler = GradScaler()

best_val_accuracy = 0
best_epoch = 0
best_model_wts = None

# Early stopping
best_val_loss = float('inf')
patience = 50
counter = 0

training_loss_value = []
validate_loss_value = []
training_acc_value = []
validate_acc_value = []

# Initialize the scaler before your training loop (just after model definition)
scaler = GradScaler()

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar for training
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

    for inputs, labels in progress_bar:
        # Skip None batches
        if inputs is None or labels is None:
            continue

        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.long()

        # Zero the gradients
        optimizer.zero_grad()

        # Mixed Precision Training
        with autocast():
            # Forward pass
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, labels)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step with scaled gradients
        scaler.step(optimizer)
        scaler.update()

        # Update running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'accuracy': 100 * correct / total
        })

    # Compute training metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    training_loss_value.append(epoch_loss)
    training_acc_value.append(epoch_accuracy)

    # Validation
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    validate_loss_value.append(val_loss)
    validate_acc_value.append(val_accuracy)

    # Learning rate scheduling
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Print epoch statistics
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, ")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, ")

    # Check if this is the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        best_model_wts = model.state_dict()
        print(
            f"New Best Val Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}, Current Learning Rate: {current_lr:.6f}\n")
        torch.save(best_model_wts, f"{save_dir}/CNN_Transformer_Classifier.pth")

    # Early stopping
    # if val_loss < best_val_loss:
        # best_val_loss = val_loss  # Update the best validation loss
        # counter = 0  # Reset the counter
        # torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        counter += 1  # Increment the counter if validation loss does not improve

    # Check for early stopping
    # if counter >= patience:
        # print("Early stopping!")
        # break  # Stop training if patience is exceeded

print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}. Model saved.")

print("Training Loss:")
print(training_loss_value)
print()

print("Validate Loss:")
print(validate_loss_value)
print()

print("Training Accuracy:")
print(training_acc_value)
print()

print("Validate Accuracy:")
print(validate_acc_value)

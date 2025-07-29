import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from preprocess import load_data
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

class DigitDataset(Dataset):
    def __init__(self, data, labels, original_shape=(26, 40)):
        self.data = data
        self.labels = labels
        self.original_shape = original_shape
        
        # Create label mapping
        unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        
        print(f"Classes: {unique_labels}")
        print(f"Number of classes: {self.num_classes}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Unpack the packed bits back to binary array
        packed_data = self.data[idx]
        unpacked = np.unpackbits(packed_data)
        
        # Trim to original size (packbits pads to multiple of 8)
        total_pixels = np.prod(self.original_shape)
        binary_array = unpacked[:total_pixels].reshape(self.original_shape)
        
        # Convert to float tensor and add channel dimension
        image = torch.FloatTensor(binary_array).unsqueeze(0)  # Shape: (1, 26, 40)
        
        # Convert label to index
        label = torch.LongTensor([self.label_to_idx[self.labels[idx]]])
        
        return image, label.squeeze()

class DigitCNN(nn.Module):
    def __init__(self, num_classes):
        super(DigitCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 26x40 -> 26x40
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 26x40 -> 26x40
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 13x20 -> 13x20
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # After conv1 + pool: 26x40 -> 13x20
        # After conv2 + pool: 13x20 -> 6x10  
        # After conv3 + pool: 6x10 -> 3x5
        self.fc1 = nn.Linear(128 * 3 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.conv1(x)))  # (batch, 32, 13, 20)
        
        # Conv block 2
        x = self.pool(self.relu(self.conv2(x)))  # (batch, 64, 6, 10)
        
        # Conv block 3
        x = self.pool(self.relu(self.conv3(x)))  # (batch, 128, 3, 5)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128*3*5)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        scheduler.step()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert indices back to original labels
    pred_labels = [dataset.idx_to_label[idx] for idx in all_predictions]
    true_labels = [dataset.idx_to_label[idx] for idx in all_labels]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))
    
    return accuracy, pred_labels, true_labels

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Loading data...")
    train_data, val_data, test_data, train_labels, val_labels, test_labels = load_data(
        'Haas_speed_images', validation_size=0.15, test_size=0.15
    )
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create datasets
    train_dataset = DigitDataset(train_data, train_labels)
    val_dataset = DigitDataset(val_data, val_labels)
    test_dataset = DigitDataset(test_data, test_labels)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = DigitCNN(num_classes=train_dataset.num_classes)
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=50, learning_rate=0.001
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, pred_labels, true_labels = evaluate_model(model, test_loader, test_dataset)
    
    # Save model
    torch.save(model.state_dict(), 'digit_cnn_model.pth')
    print("\nModel saved as 'digit_cnn_model.pth'")
    
    # Save label mapping
    import json
    with open('label_mapping.json', 'w') as f:
        json.dump({
            'label_to_idx': train_dataset.label_to_idx,
            'idx_to_label': train_dataset.idx_to_label
        }, f)
    print("Label mapping saved as 'label_mapping.json'")

if __name__ == "__main__":
    main() 
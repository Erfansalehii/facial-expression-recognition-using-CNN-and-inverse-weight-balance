import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms

# Check GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This code requires a GPU.")
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# Load dataset
df = pd.read_csv(r"C:\Users\Salehi\Desktop\projects\Research Ai\fer2013.csv")

emotions = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

def visualize_images(df, n=12):
    indices = np.random.choice(df.shape[0], n, replace=False)
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(indices):
        pixels = np.array(df.loc[idx, 'pixels'].split(), dtype='float32').reshape(48, 48)
        plt.subplot(3, 4, i + 1)
        plt.imshow(pixels, cmap='gray')
        plt.title(emotions[df.loc[idx, 'emotion']])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_images(df)
train_df = df[df['Usage'] == 'Training'].reset_index(drop=True)
val_df = df[df['Usage'] == 'PublicTest'].reset_index(drop=True)
test_df = df[df['Usage'] == 'PrivateTest'].reset_index(drop=True)


# inverse-frequency class weights
emotion_counts = train_df['emotion'].value_counts().sort_index()
total_samples = emotion_counts.sum()
num_classes = len(emotion_counts)
class_weights = total_samples / (num_classes * emotion_counts)
class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(device)

print("Class Weights:", {emotions[i]: float(class_weights[i]) for i in range(num_classes)})


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

transform_eval = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
        image = np.expand_dims(pixels, axis=-1).astype(np.uint8)
        label = int(row['emotion'])
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(pixels).unsqueeze(0) / 255.0
        return image, label

train_loader = DataLoader(FERDataset(train_df, transform_train), batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(FERDataset(val_df, transform_eval), batch_size=32, shuffle=False, pin_memory=True)
test_loader = DataLoader(FERDataset(test_df, transform_eval), batch_size=32, shuffle=False, pin_memory=True)

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout_fc(x)
        return self.fc2(x)

model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 100

def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader.dataset), correct / len(loader.dataset), all_preds, all_labels

best_val_loss = float('inf')
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader)

    val_labels_np = np.array(val_labels)
    val_preds_np = np.array(val_preds)
    per_class_acc = {
        emotions[i]: f"{(val_preds_np[val_labels_np == i] == i).sum() / max((val_labels_np == i).sum(), 1):.3f}"
        for i in emotions
    }

    print(f"Epoch {epoch:03d}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
          f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
    print("Val Per-class Acc:", per_class_acc)
    
    test_loss, test_acc, _, _ = evaluate(model, test_loader)
    print(f"🔎 Epoch {epoch:03d} — Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

model.load_state_dict(torch.load("best_model.pth", map_location=device))

test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader)
print(f"\nTest Loss: {test_loss:.4f} — Test Accuracy: {test_acc:.4f}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(cm, index=emotions.values(), columns=emotions.values()), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

print("\nClassification Report (Test):")
print(classification_report(y_true, y_pred, target_names=list(emotions.values()), digits=3))

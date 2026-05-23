import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# ================= SETTINGS =================
MODEL_PATH = "models/brain_tumor_multiclass.pth"
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL (MATCH TRAINING) =================
model = models.resnet18(pretrained=False)

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 4)
)

# LOAD TRAINED WEIGHTS
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= FILE PICKER =================
root = tk.Tk()
root.withdraw() 

file_path = filedialog.askopenfilename(
    title="Select MRI Image",
    filetypes=[("Image files", "*.jpg *.png *.jpeg")]
)

if not file_path:
    print("No file selected!")
    exit()

# ================= LOAD IMAGE =================
image = Image.open(file_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# ================= PREDICTION =================
with torch.no_grad():
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

pred_class = class_names[predicted.item()]
conf = confidence.item() * 100

# ================= OUTPUT =================
print(f"\nPrediction: {pred_class}")
print(f"Confidence: {conf:.2f}%")

# ================= SHOW IMAGE =================
plt.imshow(image)
plt.title(f"{pred_class} ({conf:.2f}%)")
plt.axis("off")
plt.show()
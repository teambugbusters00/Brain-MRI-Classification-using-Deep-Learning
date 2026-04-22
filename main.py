
from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Recreate the exact architecture used in training (Cell peD2iIoTExgh)
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0).to(device)
num_ftrs = model.num_features

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 4)
).to(device)

model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, idx = torch.max(probs, 0)

    return {
        "prediction": labels[idx.item()],
        "confidence": float(conf),
        "all_probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))}
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

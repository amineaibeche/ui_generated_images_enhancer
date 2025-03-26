import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# the  resnet 50 model 
class CNNModel_resnet_50(nn.Module):
    def __init__(self):
        super(CNNModel_resnet_50, self).__init__()
        # Charger ResNet-50 pré-entraîné
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remplacer la dernière couche par une sortie unique
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)
    

# the  resnet 18 model 
class CNNModel_resnet_18(nn.Module):
    def __init__(self):
        super(CNNModel_resnet_18, self).__init__()
        # Charger ResNet-50 pré-entraîné
        self.model = models.resnet18(pretrained=True)  # Charger ResNet-18 pré-entraîné
        self.model.fc = nn.Linear(self.model.fc.in_features, 1) 

    def forward(self, x):
        return self.model(x)
    
# the  inferece  for  the resnet  50
def resnet_50_inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_save_path = "E:\\Amine\\PFE\\IQA\\IQA_MODELS\\CNNBaseLine2.pth"   
    model = CNNModel_resnet_50().to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Charger l'image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        predicted_score = model(image_tensor).item()
    del model
    return predicted_score


# the  inferece  for  the resnet  18
def resnet_18_inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_save_path = "E:\\Amine\\PFE\\IQA\\IQA_MODELS\\CNNBaseLine2_restnet18.pth"   
    model = CNNModel_resnet_18().to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Charger l'image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        predicted_score = model(image_tensor).item()
    del model
    return predicted_score


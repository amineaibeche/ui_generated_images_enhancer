import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer ,T5Tokenizer, T5EncoderModel
from torch.utils.data import Dataset, DataLoader ,random_split , Subset
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr



device = torch.device("cuda")
# 1. Modèle ResNet50 pour les caractéristiques d'image
image_model = models.resnet50(weights='IMAGENET1K_V1') 
image_model = nn.Sequential(*list(image_model.children())[:-1])  # Retirer la dernière couche de classification
image_model = image_model.to(device)

# 2. Modèle T5 pour encoder le texte (prompt)
tokenizer = T5Tokenizer.from_pretrained('t5-small')  # Utilisation de T5 tokenizer
text_model = T5EncoderModel.from_pretrained('t5-small')  # Utilisation de T5 model sans le paramètre 'weights'
text_model = text_model.to(device)

# without cross  attention model
class AlignmentPredictionModel_without_cross_attention(nn.Module):
    def __init__(self, image_model, text_model, common_embed_dim=512):
        super(AlignmentPredictionModel_without_cross_attention, self).__init__()
        self.image_model = image_model  # Modèle d'image
        self.text_model = text_model   # Modèle de texte

        # Couches entièrement connectées pour transformer les caractéristiques
        self.fc_image = nn.Linear(2048, common_embed_dim)  # Réduction des caractéristiques d'image
        self.fc_text = nn.Linear(512, common_embed_dim)    # Réduction des caractéristiques de texte

        # Couche combinée pour prédiction finale
        self.fc_combined = nn.Sequential(
            nn.Linear(common_embed_dim * 2, common_embed_dim),  # Fusion des caractéristiques
            nn.ReLU(),
            nn.Linear(common_embed_dim, 1)  # Prédiction finale (MOS)
        )

    def forward(self, image, text):
        # Extraire les caractéristiques de l'image
        image_features = self.image_model(image).view(image.size(0), -1)  # [B, 2048]
        image_features = self.fc_image(image_features)  # Réduction à [B, common_embed_dim]

        # Encoder le texte
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(image.device)
        text_features = self.text_model(**inputs).last_hidden_state.mean(dim=1)  # [B, 512]
        text_features = self.fc_text(text_features)  # Réduction à [B, common_embed_dim]

        # Combiner les caractéristiques d'image et de texte
        combined_features = torch.cat((image_features, text_features), dim=1)  # [B, common_embed_dim * 2]

        # Prédiction de la qualité d'alignement
        return self.fc_combined(combined_features)  # [B, 1]

    


# Module Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, image_embed_dim, text_embed_dim, common_embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.image_proj = nn.Linear(image_embed_dim, common_embed_dim)
        self.text_proj = nn.Linear(text_embed_dim, common_embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=common_embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, image_features, text_features):
        """
        Applique une attention croisée entre les caractéristiques de l'image et du texte.
        Args:
            image_features : Tensor de forme [B, common_embed_dim]
            text_features : Tensor de forme [B, common_embed_dim]
        Returns:
            Tensor de forme [B, common_embed_dim] (fusion des deux caractéristiques avec attention)
        """
        # Projection des caractéristiques de l'image et du texte dans un espace commun
        image_proj = self.image_proj(image_features).unsqueeze(1)  # [B, 1, common_embed_dim]
        text_proj = self.text_proj(text_features).unsqueeze(1)    # [B, 1, common_embed_dim]

        # Appliquer l'attention croisée : image s'intéresse au texte
        attended_img_features, _ = self.multihead_attn(image_proj, text_proj, text_proj)

        # Appliquer l'attention croisée : texte s'intéresse à l'image
        attended_text_features, _ = self.multihead_attn(text_proj, image_proj, image_proj)

        # Combiner les caractéristiques (par exemple, additionner ou concaténer)
        combined_features = attended_img_features.squeeze(1) + attended_text_features.squeeze(1)  # [B, common_embed_dim]
        return combined_features



# the  modle  using  the t5 small 
class AlignmentPredictionModel_t5small(nn.Module):
    def __init__(self, image_model, text_model, common_embed_dim=512, num_heads=8):
        super(AlignmentPredictionModel_t5small, self).__init__()
        self.image_model = image_model  # Modèle d'image
        self.text_model = text_model   # Modèle de texte
        self.cross_attention = CrossAttention(image_embed_dim=2048,  # Dimension des caractéristiques de l'image
                                              text_embed_dim=512,   # Dimension des caractéristiques du texte
                                              common_embed_dim=common_embed_dim, 
                                              num_heads=num_heads)
        self.fc = nn.Linear(common_embed_dim, 1)  # Prédiction finale de la qualité d'alignement (MOS)

    def forward(self, image, text):
        # Extraire les caractéristiques de l'image
        image_features = self.image_model(image).view(image.size(0), -1)  # [B, 2048] -> [B, common_embed_dim]

        # Encoder le texte (prompt)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(image.device)
        text_features = self.text_model(**inputs).last_hidden_state.mean(dim=1)  # [B, 512]

        # Appliquer l'attention croisée entre les caractéristiques de l'image et du texte
        combined_features = self.cross_attention(image_features, text_features)

        # Prédire la qualité d'alignement (MOS)
        return self.fc(combined_features)  # [B, 1]


# the  modle  using  the t5 small
class AlignmentPredictionModel_bert(nn.Module):
    def __init__(self, image_model, text_model, common_embed_dim=512, num_heads=8):
        super(AlignmentPredictionModel_bert, self).__init__()
        self.image_model = image_model  # Modèle d'image
        self.text_model = text_model   # Modèle de texte
        self.cross_attention = CrossAttention(image_embed_dim=common_embed_dim,  # Mettre à jour la dimension de l'image
                                              text_embed_dim=common_embed_dim,   # Mettre à jour la dimension du texte
                                              common_embed_dim=common_embed_dim, 
                                              num_heads=num_heads)
        self.fc = nn.Linear(common_embed_dim, 1)  # Prédiction finale de la qualité d'alignement (MOS)
        
        # Modification de la projection du texte pour correspondre à la sortie de BERT
        self.text_proj = nn.Linear(768, common_embed_dim)  # La sortie de BERT est 768, projeté dans l'espace commun
        
        # Projection des caractéristiques de l'image dans l'espace commun
        self.image_proj = nn.Linear(2048, common_embed_dim)  # Projection des caractéristiques de l'image

    def forward(self, image, text):
        # Extraire les caractéristiques de l'image et les projeter dans l'espace commun
        image_features = self.image_model(image).view(image.size(0), -1)  # [B, 2048] -> [B, common_embed_dim]
        image_proj = self.image_proj(image_features)  # [B, common_embed_dim]

        # Encoder le texte (prompt) et appliquer la projection dans l'espace commun
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(image.device)
        text_features = self.text_model(**inputs).last_hidden_state.mean(dim=1)  # [B, 768]
        text_proj = self.text_proj(text_features)  # [B, common_embed_dim]

        # Appliquer l'attention croisée entre les caractéristiques de l'image et du texte
        combined_features = self.cross_attention(image_proj, text_proj)

        # Prédire la qualité d'alignement (MOS)
        return self.fc(combined_features)  # [B, 1]



# the  infernce  without cross attention
def inference_without_cross_attention(image_path,  prompt ) : 
    # del tokenizer
    # del text_model
    # del image_model
    # 1. Modèle ResNet50 pour les caractéristiques d'image
    image_model = models.resnet50(weights='IMAGENET1K_V1') 
    image_model = nn.Sequential(*list(image_model.children())[:-1])  # Retirer la dernière couche de classification
    image_model = image_model.to(device)

    # 2. Modèle T5 pour encoder le texte (prompt)
    tokenizer = T5Tokenizer.from_pretrained('t5-small')  # Utilisation de T5 tokenizer
    text_model = T5EncoderModel.from_pretrained('t5-small')  # Utilisation de T5 model sans le paramètre 'weights'
    text_model = text_model.to(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Optionnel : Convertir les images en RGB si nécessaire
        transforms.Resize((224, 224)),  # Redimensionner les images pour ResNet
        transforms.ToTensor(),  # Convertir l'image en un tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
    ])
    model_path = "E:\\Amine\\PFE\\IQA\\IQA_MODELS\\alignment_prediction_model_without_cross_attention.pth"
    model = AlignmentPredictionModel_without_cross_attention(image_model,text_model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        predicted_score = model(image_tensor, prompt).squeeze().item()

    del model
    del tokenizer
    del text_model
    del image_model
    return predicted_score


# the  infernce  without cross attention
def inference_cross_attention_t5(image_path,  prompt ) :
    # del tokenizer
    # del text_model
    # del image_model 
    # 1. Modèle ResNet50 pour les caractéristiques d'image
    image_model = models.resnet50(weights='IMAGENET1K_V1') 
    image_model = nn.Sequential(*list(image_model.children())[:-1])  # Retirer la dernière couche de classification
    image_model = image_model.to(device)

    # 2. Modèle T5 pour encoder le texte (prompt)
    tokenizer = T5Tokenizer.from_pretrained('t5-small')  # Utilisation de T5 tokenizer
    text_model = T5EncoderModel.from_pretrained('t5-small')  # Utilisation de T5 model sans le paramètre 'weights'
    text_model = text_model.to(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Optionnel : Convertir les images en RGB si nécessaire
        transforms.Resize((224, 224)),  # Redimensionner les images pour ResNet
        transforms.ToTensor(),  # Convertir l'image en un tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
    ])
    model_path = "E:\\Amine\\PFE\\IQA\\IQA_MODELS\\alignment_prediction_model.pth"
    model = AlignmentPredictionModel_t5small(image_model,text_model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        predicted_score = model(image_tensor, prompt).squeeze().item()

    del model
    del tokenizer
    del text_model
    del image_model
    return predicted_score


# the  infernce  without cross attention
def inference_cross_attention_bert(image_path,  prompt ) :
    # del tokenizer
    # del text_model
    # del image_model
    # 1. Modèle ResNet50 pour les caractéristiques d'image
    image_model = models.resnet50(weights='IMAGENET1K_V2') 
    image_model = nn.Sequential(*list(image_model.children())[:-1])  # Retirer la dernière couche de classification
    image_model = image_model.to(device)

    # 2. Modèle T5 pour encoder le texte (prompt)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Tokenizer BERT
    text_model = BertModel.from_pretrained('bert-base-uncased')  # Modèle BERT sans la tête de classification
    text_model = text_model.to(device) 

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Optionnel : Convertir les images en RGB si nécessaire
        transforms.Resize((224, 224)),  # Redimensionner les images pour ResNet
        transforms.ToTensor(),  # Convertir l'image en un tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
    ])
    model_path = "E:\\Amine\\PFE\\IQA\\IQA_MODELS\\alignment_prediction_model_bert_resnetv2.pth"
    model = AlignmentPredictionModel_bert(image_model,text_model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        predicted_score = model(image_tensor, str(inputs)).squeeze().item()

    del model
    del tokenizer
    del text_model
    del image_model
    return predicted_score
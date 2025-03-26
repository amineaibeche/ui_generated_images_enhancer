# Generated Images Enhancer & Evaluator

Projet de traitement d'images utilisant l'IA pour améliorer et évaluer la qualité perceptive des images.

![Interface](./screenshot.png) *(Ajoutez une capture d'écran réelle)*

## Fonctionnalités
- **Amélioration d'images** : Utilise Stable Diffusion avec des paramètres personnalisables (prompt, étapes, force, etc.).
- **Évaluation de qualité** : 
  - Score MOS (ResNet-18/50)
  - Score d'alignement (AlignNet avec T5/BERT)
- **Interface dynamique** :
  - Thème clair/sombre
  - Animation de chargement avec dégradé
  - Boutons de téléchargement (1K, 2K, 4K)
- **Modèles personnalisables** : Choix du modèle perceptuel, d'alignement et d'amélioration

## Stack technique
- **Backend** : Django + Django REST Framework
- **Frontend** : React + Tailwind CSS
- **IA** :
  - PyTorch
  - Stable Diffusion
  - ResNet
  - Modèles T5/BERT pour l'alignement

## Installation

### Backend (Django)
```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Appliquer les migrations
python manage.py migrate

# Lancer le serveur
python manage.py runserver

# VANAD ML - Système de Prédiction de Temps d'Attente

## 📋 Aperçu du Projet

VANAD ML est un système de machine learning avancé conçu pour prédire les temps d'attente dans les systèmes de files d'attente. Ce projet implémente une approche multi-modèles combinant des algorithmes d'apprentissage automatique traditionnels et des réseaux de neurones profonds pour optimiser les prédictions.

## 🎯 Objectif Principal

Prédire avec précision le temps d'attente (`W`) dans un système de files d'attente en utilisant diverses caractéristiques temporelles et opérationnelles, avec pour métrique principale le **RRMSE (Relative Root Mean Square Error)**.

## 🏗️ Architecture du Système

### Modèles Implémentés

1. **Random Forest Regressor** - Modèle d'ensemble basé sur des arbres
2. **Gradient Boosting Regressor** - Boosting séquentiel 
3. **LightGBM** - Gradient boosting optimisé
4. **CatBoost** - Algorithme de boosting robuste
5. **AdvancedVANADNet** - Réseau de neurones profond personnalisé

### Réseau de Neurones Personnalisé

```python
class AdvancedVANADNet(nn.Module):
    - Architecture: [512, 256, 128] neurones
    - Normalisation par batch
    - Dropout adaptatif
    - Activation ReLU
    - Initialisation Kaiming
```

## 📊 Données et Préparation

### Fichiers de Données
- `vanad_training_ssj.csv` - Données d'entraînement
- `vanad_test_ssj.csv` - Données de test

### Variables Principales
- **Target**: `W` (temps d'attente)
- **Features de base**: `T`, `qT`, `l1`, `l2`, `l3`, `l4`, `t_hour`, `t_day_of_week`, `s`, `P_LES`, `P_Avg_LES`

### Preprocessing
- Filtrage des outliers (0 < W ≤ 7200)
- Suppression des valeurs extrêmes (Q1-Q99)
- Gestion des valeurs manquantes par la médiane

## 🔧 Feature Engineering

### Features Générées

#### 1. **Features de Files d'Attente**
```python
total_queue = l1 + l2 + l3 + l4
max_queue = max(l1, l2, l3, l4)
queue_std = std(l1, l2, l3, l4)
l1_ratio = l1 / total_queue
```

#### 2. **Features Temporelles**
```python
hour_sin = sin(2π * t_hour / 24)
hour_cos = cos(2π * t_hour / 24)
is_weekend = (t_day_of_week in [5, 6])
```

#### 3. **Features d'Interaction**
```python
queue_efficiency = qT / (total_queue + 1)
```

#### 4. **Transformations Non-linéaires**
```python
qT_log = log(qT + 1)
qT_sqrt = sqrt(qT)
T_log = log(T + 1)
T_sqrt = sqrt(T)
```

## 🚀 Entraînement et Optimisation

### Configuration Hardware
- **Support Multi-plateforme**: CPU, CUDA, Apple Silicon (MPS)
- **Gestion Mémoire**: Nettoyage automatique, batch adaptatif
- **Monitoring**: Utilisation RAM en temps réel

### Hyperparamètres Optimisés

#### Modèles ML
```python
RandomForest: n_estimators=300, max_depth=12
GradientBoosting: n_estimators=300, max_depth=10, learning_rate=0.1
LightGBM: n_estimators=300, max_depth=10, subsample=0.8
CatBoost: iterations=300, depth=10, learning_rate=0.1
```

#### Réseau de Neurones
```python
Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
Loss: MSE
Early Stopping: patience=20, min_delta=0.0001
Batch Size: Adaptatif (max 4096)
```

## 📈 Métrique d'Évaluation

### Métriques Calculées
- **RRMSE** (Principal): `RMSE / mean(y_true)`
- **R²**: Coefficient de détermination
- **RMSE**: Racine de l'erreur quadratique moyenne
- **MAE**: Erreur absolue moyenne
- **Accuracy ±10%**: Pourcentage de prédictions dans ±10% de la vraie valeur

### Baseline de Référence
- **Random Forest Baseline**: RRMSE = 0.8648
- **Objectif**: Améliorer cette performance de référence

## 🔬 Techniques Avancées

### 1. **Transformation Logarithmique**
```python
y_train_log = log(y_train + 1)
y_pred = exp(y_pred_log) - 1
```

### 2. **Normalisation Robuste**
- `StandardScaler` pour les réseaux de neurones
- `RobustScaler` pour les modèles ML (résistant aux outliers)

### 3. **Early Stopping**
- Prévention du surapprentissage
- Sauvegarde du meilleur état du modèle

### 4. **Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 💾 Sauvegarde et Persistance

### Modèles Sauvegardés
- Top 4 modèles par performance RRMSE
- Format: `model_rank_name_timestamp.pkl`
- Contenu: modèle, scaler, métriques, métadonnées

### Structure de Sauvegarde
```
saved_models/
├── model_1_LightGBM_20240109_143052.pkl
├── model_2_Neural_Network_20240109_143052.pkl
├── model_3_CatBoost_20240109_143052.pkl
└── model_4_RandomForest_20240109_143052.pkl
```

## 🎯 Résultats Typiques

### Classement des Modèles (Exemple)
```
1. LightGBM:
   RRMSE: 0.7842
   R²: 0.8756
   RMSE: 156.23
   MAE: 98.45
   Précision ±10%: 78.9%

2. Neural Network:
   RRMSE: 0.7901
   R²: 0.8721
   RMSE: 157.89
   MAE: 99.12
   Précision ±10%: 77.8%
```

## 🚀 Utilisation

### Prérequis
```bash
pip install pandas numpy torch scikit-learn lightgbm catboost joblib psutil
```

### Exécution
```bash
python vanad_ml.py
```

### Surveillance
- Monitoring automatique de la RAM
- Affichage des progrès d'entraînement
- Métriques en temps réel

## 🔍 Fonctionnalités Techniques

### Gestion Mémoire
- Nettoyage automatique avec `gc.collect()`
- Vidage cache GPU/MPS
- Batch processing pour les grandes données

### Parallélisation
- Utilisation maximale des cœurs CPU (`n_jobs=-1`)
- Support GPU complet (CUDA/MPS)

### Reproductibilité
- Seeds fixés: `torch.manual_seed(42)`, `np.random.seed(42)`
- Résultats reproductibles entre exécutions

## 📊 Monitoring et Logs

### Affichage en Temps Réel
```
🚀 Device: mps
💾 RAM disponible: 32.1 GB
📊 Utilisation de toutes les données: 45623 échantillons
🔄 LightGBM...
✅ LightGBM: RRMSE=0.7842, R²=0.8756
```

### Comparaison Performance
```
📈 Amélioration vs RF baseline (RRMSE 0.8648): +9.3%
```


## 🔧 Configuration et Personnalisation

### Paramètres Modifiables
```python
# Taille des couches du réseau
hidden_layers = [512, 256, 128]

# Taux de dropout
dropout_rate = 0.3

# Patience early stopping
patience = 20

# Nombre de modèles à sauvegarder
top_n = 4
```


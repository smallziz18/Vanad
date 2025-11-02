import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import catboost as cb
import joblib
import gc
import os
from datetime import datetime
import warnings
import psutil

warnings.filterwarnings('ignore')

# Configuration
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"🚀 Device: {device}")
print(f"💾 RAM disponible: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB")


class AdvancedVANADNet(nn.Module):
    """Réseau de neurones avancé avec attention"""

    def __init__(self, input_size, hidden_layers=[512, 256, 128], dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_layers):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate * (0.8 ** i))
            ])
            prev_size = hidden_size

        # Couche de sortie
        layers.extend([
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(prev_size // 2, 1)
        ])

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    """Early stopping simple"""

    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})
            return True
        return False


def create_features(df, feature_cols):
    """Création de features optimisée"""
    X = df[feature_cols].copy()

    # Features de queue
    queue_cols = [col for col in ['l1', 'l2', 'l3', 'l4'] if col in X.columns]
    if len(queue_cols) > 1:
        X['total_queue'] = X[queue_cols].sum(axis=1)
        X['max_queue'] = X[queue_cols].max(axis=1)
        X['queue_std'] = X[queue_cols].std(axis=1).fillna(0)
        X['l1_ratio'] = X['l1'] / (X['total_queue'] + 1e-8)

    # Features temporelles
    if 't_hour' in X.columns:
        X['hour_sin'] = np.sin(2 * np.pi * X['t_hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['t_hour'] / 24)
        X['is_weekend'] = (X['t_day_of_week'].isin([5, 6])).astype(int) if 't_day_of_week' in X.columns else 0

    # Features d'interaction
    if 'qT' in X.columns and 'total_queue' in X.columns:
        X['queue_efficiency'] = X['qT'] / (X['total_queue'] + 1)

    # Transformations non-linéaires
    for col in ['qT', 'T']:
        if col in X.columns:
            X[f'{col}_log'] = np.log1p(X[col])
            X[f'{col}_sqrt'] = np.sqrt(X[col])

    return X


def feature_engineering(train_df, test_df):
    """Feature engineering principal"""
    print("🔧 Feature engineering...")

    base_features = ['T', 'qT', 'l1', 'l2', 'l3', 'l4', 't_hour', 't_day_of_week', 's', 'P_LES', 'P_Avg_LES']
    available_features = [col for col in base_features if col in train_df.columns]

    X_train = create_features(train_df, available_features)
    X_test = create_features(test_df, available_features)

    y_train = train_df['target'].copy()
    y_test = test_df['target'].copy()

    print(f"✅ {X_train.shape[1]} features créées")
    return X_train, X_test, y_train, y_test


def get_models():
    """Modèles ML optimisés """
    print("🔧 Configuration des modèles ML...")

    return {
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            n_jobs=-1,
            random_state=42,
            max_features='sqrt'
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            subsample=0.8,
            learning_rate=0.1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.1
        ),
        'CatBoost': cb.CatBoostRegressor(
            iterations=300,
            depth=10,
            random_state=42,
            verbose=False,
            thread_count=-1,
            subsample=0.8,
            learning_rate=0.1
        )
    }


def train_ml_models(X_train, y_train, X_test, y_test):
    """Entraînement des modèles ML sans limitations"""
    print("🤖 Entraînement des modèles ML...")
    print(f"📊 Utilisation de toutes les données: {X_train.shape[0]} échantillons")

    # Transformation log
    y_train_log = np.log1p(y_train)

    # Normalisation
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"   🔄 {name}...")

        # Nettoyage mémoire avant chaque modèle
        gc.collect()

        try:
            # Entraînement
            model.fit(X_train_scaled, y_train_log)

            # Prédiction
            y_pred_log = model.predict(X_test_scaled)
            y_pred = np.expm1(y_pred_log)

            metrics = calculate_metrics(y_test, y_pred)
            results[name] = metrics
            trained_models[name] = {'model': model, 'scaler': scaler}

            print(f"   ✅ {name}: RRMSE={metrics['RRMSE']:.4f}, R²={metrics['R²']:.4f}")

        except Exception as e:
            print(f"   ❌ {name}: Erreur - {str(e)}")
            gc.collect()
            continue

    return results, trained_models


def train_neural_model(X_train, y_train, X_test, y_test):
    """Entraînement du réseau de neurones sans limitations"""
    print("🧠 Entraînement du réseau de neurones...")
    print(f"📊 Utilisation de toutes les données: {X_train.shape[0]} échantillons")

    # Transformation log
    y_train_log = np.log1p(y_train)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split validation
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train_scaled, y_train_log, test_size=0.15, random_state=42
    )

    # Modèle avec taille augmentée
    model = AdvancedVANADNet(
        X_train.shape[1],
        hidden_layers=[512, 256, 128],
        dropout_rate=0.3
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=20)

    # Batch size adaptatif selon la taille des données
    batch_size = min(4096, X_train.shape[0] // 100)
    print(f"   📦 Batch size: {batch_size}")

    # Tenseurs
    X_train_tensor = torch.FloatTensor(X_train_val).to(device)
    y_train_tensor = torch.FloatTensor(y_train_val.values).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Entraînement
    print("   🏃 Début de l'entraînement...")
    for epoch in range(300):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

                print(f"   📊 Epoch {epoch}: Train Loss={epoch_loss / len(train_loader):.6f}, Val Loss={val_loss:.6f}")

                if early_stopping(val_loss, model):
                    print(f"   ⏹️  Early stopping à l'epoch {epoch}")
                    break

    # Prédiction
    model.eval()
    with torch.no_grad():
        # Prédiction par batch pour gérer la mémoire
        y_pred_list = []
        test_batch_size = 20000

        for i in range(0, X_test_scaled.shape[0], test_batch_size):
            batch_end = min(i + test_batch_size, X_test_scaled.shape[0])
            X_batch = torch.FloatTensor(X_test_scaled[i:batch_end]).to(device)
            pred_batch = model(X_batch).cpu().numpy().flatten()
            y_pred_list.extend(pred_batch)

    y_pred_log = np.array(y_pred_list)
    y_pred = np.expm1(y_pred_log)
    metrics = calculate_metrics(y_test, y_pred)

    print(f"   ✅ Neural Network: RRMSE={metrics['RRMSE']:.4f}, R²={metrics['R²']:.4f}")

    # Nettoyage mémoire
    del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return {'Neural_Network': metrics}, {'Neural_Network': {'model': model, 'scaler': scaler}}


def calculate_metrics(y_true, y_pred):
    """Calcul des métriques"""
    y_true = np.maximum(y_true, 0.001)
    y_pred = np.maximum(y_pred, 0.001)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mean_waiting_time = np.mean(y_true)
    rrmse = rmse / mean_waiting_time

    relative_errors = np.abs((y_true - y_pred) / (y_true + 1e-7))
    accuracy_10 = np.mean(relative_errors < 0.1) * 100

    return {
        'RMSE': rmse,
        'RRMSE': rrmse,
        'MAE': mae,
        'R²': r2,
        'Accuracy_10%': accuracy_10
    }


def save_best_models(all_results, all_trained_models, top_n=4):
    """Sauvegarde des meilleurs modèles"""
    print(f"\n💾 Sauvegarde des {top_n} meilleurs modèles...")

    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['RRMSE'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (model_name, metrics) in enumerate(sorted_models[:top_n]):
        if model_name in all_trained_models:
            filename = f"{SAVE_DIR}/model_{i + 1}_{model_name}_{timestamp}.pkl"

            save_data = {
                'model': all_trained_models[model_name]['model'],
                'scaler': all_trained_models[model_name]['scaler'],
                'metrics': metrics,
                'model_name': model_name
            }

            joblib.dump(save_data, filename)
            print(f"   ✅ {model_name} (RRMSE: {metrics['RRMSE']:.4f}) → {filename}")


def main():
    """Fonction principale"""
    print("🚀 VANAD ML - VERSION SANS LIMITATIONS")
    print("=" * 50)

    # Chargement des données
    print("📖 Chargement des données...")
    train_df = pd.read_csv("vanad_training_ssj.csv").rename(columns={"W": "target"})
    test_df = pd.read_csv("vanad_test_ssj.csv").rename(columns={"W": "target"})



    Q1, Q99 = train_df['target'].quantile([0.01, 0.99])
    train_df = train_df[(train_df['target'] >= Q1) & (train_df['target'] <= Q99)]
    test_df = test_df[(test_df['target'] >= Q1) & (test_df['target'] <= Q99)]

    print(f"✅ Train: {train_df.shape}, Test: {test_df.shape}")

    # Feature engineering
    X_train, X_test, y_train, y_test = feature_engineering(train_df, test_df)

    # Nettoyage mémoire
    del train_df, test_df
    gc.collect()

    # Entraînement des modèles
    print(f"\n🚀 Début de l'entraînement avec {X_train.shape[0]} échantillons...")
    ml_results, ml_trained_models = train_ml_models(X_train, y_train, X_test, y_test)
    nn_results, nn_trained_models = train_neural_model(X_train, y_train, X_test, y_test)

    # Combinaison des résultats
    all_results = {**ml_results, **nn_results}
    all_trained_models = {**ml_trained_models, **nn_trained_models}

    # Affichage des résultats
    print("\n🏆 RÉSULTATS FINAUX:")
    print("=" * 60)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['RRMSE'])

    for i, (model_name, metrics) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {model_name}:")
        print(f"    RRMSE: {metrics['RRMSE']:.4f}")
        print(f"    R²: {metrics['R²']:.4f}")
        print(f"    RMSE: {metrics['RMSE']:.2f}")
        print(f"    MAE: {metrics['MAE']:.2f}")
        print(f"    Précision ±10%: {metrics['Accuracy_10%']:.1f}%")
        print("-" * 40)

    # Sauvegarde
    save_best_models(all_results, all_trained_models)

    # Comparaison avec baseline
    if sorted_results:
        best_rrmse = sorted_results[0][1]['RRMSE']
        baseline_rrmse = 0.8648
        improvement = ((baseline_rrmse - best_rrmse) / baseline_rrmse) * 100
        print(f"\n📈 Amélioration vs RF baseline (RRMSE {baseline_rrmse:.4f}): {improvement:+.1f}%")

    # Statistiques finales
    print(f"\n📊 Statistiques:")
    print(f"   • Nombre de modèles entraînés: {len(all_results)}")
    print(f"   • Meilleur modèle: {sorted_results[0][0]}")
    print(f"   • Meilleur RRMSE: {sorted_results[0][1]['RRMSE']:.4f}")

    # Nettoyage final
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return all_results, all_trained_models


if __name__ == "__main__":
    try:
        all_results, all_trained_models = main()
        print("\n✅ Entraînement terminé avec succès!")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Nettoyage final de la mémoire
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
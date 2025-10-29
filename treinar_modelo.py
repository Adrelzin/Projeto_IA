#.venv\Scripts\activate
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore') #Ignora os warnings de funções que não funcionariam ou precisam ser mudadas

def generate_data(n=500):
    """Gera dataset sintético balanceado"""
    np.random.seed(42) #Semente de aleatoriedade, o valor 42, cria valores constantes e que não mudam todas vez que o código roda 
    
    #Dicionário
    #radint(): Gera números inteiros aleatórios
    #random.choice(): Gera binários de 1 ou 0
    #n = Números de valores para gerar (Presente ali na função)
    #p: Probabilidades (inventadas)
    d = {
        'ages': np.random.randint(20, 85, n),
        'smoking': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'yellow_fingers': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'anxiety': np.random.choice([0, 1], n, p=[0.5, 0.5]),
        'alcohol': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'peer_pressure': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'chronic_disease': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'fatigue': np.random.choice([0, 1], n, p=[0.5, 0.5]),
        'allergy': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'wheezing': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'coughing': np.random.choice([0, 1], n, p=[0.5, 0.5]),
        'shortness_breath': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'swallowing_difficulty': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'chest_pain': np.random.choice([0, 1], n, p=[0.6, 0.4])
    }
    
    #normal: Simula variabilidade natural
    risk = ((d['ages'] > 60) * 0.3 + d['smoking'] * 0.4 + d['yellow_fingers'] * 0.2 +
            d['chronic_disease'] * 0.25 + d['coughing'] * 0.3 + d['shortness_breath'] * 0.35 +
            d['chest_pain'] * 0.3 + d['wheezing'] * 0.25 + d['fatigue'] * 0.15 +
            np.random.normal(0, 0.15, n))
    
    cancer = (risk > 1.2).astype(int)
    
    half = n // 2
    idx = np.concatenate([np.where(cancer == 0)[0][:half], np.where(cancer == 1)[0][:half]])
    
    return pd.DataFrame({
        'AGE': d['ages'][idx], 'SMOKING': d['smoking'][idx],
        'YELLOW_FINGERS': d['yellow_fingers'][idx], 'ANXIETY': d['anxiety'][idx],
        'ALCOHOL CONSUMING': d['alcohol'][idx], 'PEER_PRESSURE': d['peer_pressure'][idx],
        'CHRONIC DISEASE': d['chronic_disease'][idx], 'FATIGUE': d['fatigue'][idx],
        'ALLERGY': d['allergy'][idx], 'WHEEZING': d['wheezing'][idx],
        'COUGHING': d['coughing'][idx], 'SHORTNESS OF BREATH': d['shortness_breath'][idx],
        'SWALLOWING DIFFICULTY': d['swallowing_difficulty'][idx], 'CHEST PAIN': d['chest_pain'][idx],
        'LUNG_CANCER': cancer[idx]
    })

data = generate_data(500)
X = data.drop(['LUNG_CANCER'], axis=1)
y = data['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                                min_samples_leaf=3, max_features='sqrt', random_state=42,
                                n_jobs=-1, oob_score=True)
model.fit(X_train_bal, y_train_bal)

joblib.dump(model, 'modelo_random_forest.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')

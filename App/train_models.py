import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from pathlib import Path

# Path to data
DATA_PATH = Path('../data/final_processed_dataset_V2.csv')

def train_and_save_models():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    # 2. Features and Target
    X = df[['acousticness', 'danceability', 'energy', 'valence', 'tempo', 'loudness', 'speechiness', 'instrumentalness']]
    y = df['is_hit']

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Define individual models
    lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42))])
    knn = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))])
    rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    nb = Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())])

    models = {
        'Logistic Regression': lr,
        'K-Nearest Neighbors': knn,
        'Random Forest': rf,
        'Naive Bayes': nb
    }

    # 5. Define and add the Ensemble (Voting Classifier)
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('knn', knn),
            ('rf', rf),
            ('nb', nb)
        ],
        voting='soft'
    )
    models['Ensemble (All Models)'] = ensemble

    # 6. Train and Save
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        filename = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(pipeline, filename)
        print(f"Saved {filename}")

if __name__ == "__main__":
    train_and_save_models()

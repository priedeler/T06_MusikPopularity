# T06 MusikPopularity

Project analyzing the popularity of music tracks using Machine Learning. The goal is to predict whether a song will be a "hit" based on its audio features.

## Project Overview
This project uses the Spotify dataset to classify songs as hits or non-hits. We employ various classification techniques, with a primary focus on Naive Bayes and K-Nearest Neighbors (KNN).

## Project Structure
The repository is organized as follows:

- **`data/`**: Contains the processed dataset files.
  - `final_processed_dataset_V2.csv`: The primary dataset used for training, containing ~10,000 balanced samples.
- **`01_Data_Cleaning/`**:
  - `Data_Cleaning.ipynb`: Preprocessing and filtering logic. Filters songs released after 2000, removes non-music tracks (high speechiness), and balances the classes (50% hit / 50% non-hit).
- **`02_Models/`**:
  - **`Naive_Bayes/`**:
    - **`Naive_Bayes.ipynb`**: Comprehensive analysis of hit prediction using audio features.
      - **Gaussian NB**: Establishes a baseline for continuous data (~79.7% accuracy).
      - **Categorical NB**: Improves performance by discretizing audio features into "zones" using `KBinsDiscretizer`.
      - **Optimization**: Hyperparameter tuning for bin sizes (optimal at 9 bins) reached **80.59% accuracy**.
    - **`Lyrics_Analysis.ipynb`**: Natural Language Processing (NLP) focused on genre classification.
      - **Pipeline**: Lowercasing, regex cleaning (removing structure tags like `[Chorus]`), stop word removal, and lemmatization.
      - **Vectorization**: Employs **TF-IDF Vectorization** to weight unique lyrical themes.
      - **Results**: Demonstrates that lyrics are highly predictive of genre, with detailed evaluation via confusion matrices for the top 10 most frequent genres.


We implemented and compared three classifiers in a consistent pipeline setup.

- **kNN:** `StandardScaler` + `KNeighborsClassifier`, tuned with `GridSearchCV` (4-fold stratified CV) over `n_neighbors`, `weights`, and distance metric (`p`).
- **Logistic Regression:** `StandardScaler` + `LogisticRegression(max_iter=2000)` as a linear baseline.
- **Random Forest:** `RandomForestClassifier(n_estimators=200)` as a non-linear ensemble baseline.

All models were trained on the same train/test split and evaluated with Accuracy, Precision, Recall, F1, and ROC AUC to ensure a fair comparison.

- **`requirements.txt`**: Python dependencies required to run the notebooks.

---
*Note: The raw dataset `data/master_dataset_enriched.csv` is not included in the repository due to its size.*

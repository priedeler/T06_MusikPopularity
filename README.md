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
    - `Naive_Bayes.ipynb`: In-depth analysis of Naive Bayes variants. Includes Gaussian NB baseline and an optimized Categorical NB using hyperparameter tuning for bin/bracket sizes.
    - `Lyrics_Analysis.ipynb`: Exploration of using song lyrics for popularity prediction.
  - **`KNN/`**:
    - `data mining project.ipynb`: Comprehensive implementation of K-Nearest Neighbors, along with Logistic Regression and Random Forest comparisons.
- **`requirements.txt`**: Python dependencies required to run the notebooks.

## Key Methodology & Findings
### Data Processing
To ensure a high-quality model, the dataset was filtered to include only modern songs (Release Year >= 2000) and to exclude podcasts or audiobooks (Speechiness < 0.5). We defined a "hit" as a track with a popularity score >= 50 and balanced the dataset to prevent class bias.

### Model Optimization
Our research highlighted the importance of feature representation:
- **Gaussian Naive Bayes**: Provided a strong baseline by assuming normal distribution of features.
- **Categorical Naive Bayes**: By discretizing continuous features into "brackets," we could capture non-linear patterns.
- **Hyperparameter Tuning**: We benchmarked bin sizes from 2 to 20. The optimal configuration (9 bins) achieved an accuracy of **0.8059**, outperforming the Gaussian baseline (**0.7969**).

## Conclusion
The results suggest that song popularity is often better predicted by identifying specific "zones" of audio features (e.g., specific tempo ranges or loudness levels) rather than relying strictly on the exact continuous values.

---
*Note: The raw dataset `data/master_dataset_enriched.csv` is not included in the repository due to its size.*

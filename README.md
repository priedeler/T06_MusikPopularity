# T06 MusikPopularity

Project analyzing the popularity of music using Machine Learning.

## Project Structure
The repository is organized as follows:
- **`data/`**: Contains all dataset files (CSV, XLSX).
- **`01_Data_Cleaning/`**: Contains the data processing and filtering notebooks.
  - `Data_Cleaning.ipynb`: Preprocessing, balancing, and filtering of the raw dataset.
- **`02_Models/`**: Contains machine learning models and implementation.
  - `Naive_Bayes.ipynb`: Implementation and comparison of Gaussian, Categorical, and Multinomial Naive Bayes models.
  - `knn.py`: Placeholder for K-Nearest Neighbors implementation.
- **`requirements.txt`**: Python dependencies.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Navigate to the desired notebook:
   - For data cleaning: `01_Data_Cleaning/Data_Cleaning.ipynb`
   - For model analysis: `02_Models/Naive_Bayes.ipynb`

*Note: The raw dataset `data/master_dataset_enriched.csv` is not included in the repository due to its size (~178MB).*

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

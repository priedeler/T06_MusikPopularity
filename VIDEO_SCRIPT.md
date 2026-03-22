# Video Script: Predicting Hit Songs with Machine Learning

**Duration:** ~10 Minutes
**Goal:** Explain the project workflow, methodology, key findings, and future potential.

---

## 0. Introduction (0:00 - 1:00)
*   **Hook:** "Have you ever wondered if there's a mathematical formula for a hit song? Is it the beat, the lyrics, or just pure luck?"
*   **The Problem:** The music industry is flooded with millions of tracks. For artists and labels, predicting success is a high-stakes game.
*   **The Project:** "In this project, I used Machine Learning to analyze thousands of songs to see if we can predict a 'Hit' before it even climbs the charts."
*   **Visual:** Show a montage of Spotify interfaces and scrolling code.

## 1. The Data & Cleaning Phase (1:00 - 2:30)
*   **The Source:** We started with a massive master dataset of Spotify tracks, containing audio features like *Danceability*, *Energy*, and *Loudness*.
*   **The "Hit" Definition:** We defined a "Hit" as a song with a popularity score over 50.
*   **Cleaning Steps:**
    *   Filtered for modern songs (released after the year 2000).
    *   Removed non-musical content (podcasts, audiobooks) by filtering for low `speechiness`.
    *   **The Challenge:** Data Imbalance. There are way more "flops" than "hits."
*   **Solution:** I performed 50/50 balancing, sampling an equal number of hits and non-hits to ensure the models wouldn't be biased.
*   **Visual:** Show the `Data_Cleaning.ipynb` code and the resulting balanced distribution plot.

## 2. Audio Feature Analysis (2:30 - 4:30)
*   **The Models:** We tested three main contenders:
    1.  **k-Nearest Neighbors (kNN):** Looking for "neighbor" songs with similar vibes. (Accuracy: ~65%)
    2.  **Random Forest:** An ensemble of decision trees. (Accuracy: ~68%)
    3.  **Naive Bayes:** A probabilistic approach.
*   **The Breakthrough:** I discovered that standard Gaussian Naive Bayes assumed a "perfect bell curve" for features, which isn't always true. By switching to **Categorical Naive Bayes** and "binning" the data (grouping features into brackets), accuracy jumped to **81%**.
*   **Finding:** Danceability and Energy are the biggest predictors. Hits consistently have higher scores here.
*   **Visual:** Show the accuracy comparison bar chart from `Summary_Insights.ipynb`.

## 3. Advanced NLP: The Lyrics Factor (4:30 - 7:00)
*   **The Pivot:** "Audio features tell us how the song *sounds*, but not what it *says*."
*   **Methodology:** I brought in a specialized Lyrics Dataset. 
*   **The Tech:** 
    *   **TF-IDF:** To find the most important words.
    *   **SVD (Singular Value Decomposition):** To find "topics" or themes within the lyrics.
*   **The Result:** Combining Audio Features + Lyrical Topics pushed our accuracy to a staggering **89%**.
*   **Disclaimer:** Note that this was a different dataset, used specifically to identify *what* makes a hit semantically.
*   **Insight:** Lyrical repetition and relatable social themes were massive indicators of success.
*   **Visual:** Show the "Audio vs. Audio+Lyrics" comparison graph.

## 4. Key Takeaways: The "Perfect Hit" Formula (7:00 - 8:30)
*   **The Formula:**
    1.  **Vibe over Complexity:** High movement-oriented features (Danceability/Energy).
    2.  **Production Value:** Hits are consistently "Loud" but with controlled dynamic ranges.
    3.  **Lyrical Repetition:** Familiarity and rhythmic vocabulary clusters are key.
*   **The "Why":** These features align with how streaming algorithms and radio play prioritize content.
*   **Visual:** A simple table summarizing "Hit Ingredients."

## 5. Future Steps: AI & Backtesting (8:30 - 9:30)
*   **The MP3 Classifier:** My next goal is to build a tool that extracts these KPIs directly from raw `.mp3` files using libraries like `librosa`.
*   **The AI Lab:** We can use this to test AI-generated music. "Can we prompt an AI music generator, run the output through our model, and iterate until we have a mathematically optimized hit?"
*   **Backtesting:** Real-world validation by testing the model against next week's new releases.
*   **Visual:** Diagram of a feedback loop between an AI Music Generator and the Predictor Model.

## 6. Conclusion (9:30 - 10:00)
*   **Final Word:** "Machine Learning isn't replacing creativity, but it's giving us a powerful lens to understand why we love the music we do."
*   **Call to Action:** "Check out the full repository in the description. Thanks for watching!"
*   **Visual:** Credits and link to the GitHub repo.

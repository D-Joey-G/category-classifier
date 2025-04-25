# Initial Model Benchmarking Summary

## Objective
Classify quiz questions into one of 8 categories using supervised machine learning models, comparing multiple vectorizers, feature dimensions, and algorithms.

## Dataset
- Total samples: 36,700 
- Fields used: `Question`, `Answer`, `Category`
- Preprocessing steps:
  - Lowercased, removed punctuation, normalized whitespace
  - Combined `Question` and `Answer` into a `clean_text` field
  - Vectorized using `TfidfVectorizer` and `CountVectorizer` with varying `max_features`
  - Label-encoded `Category`

## Initial Experiment Setup
- **Vectorizers**: TF-IDF and CountVectorizer
- **Feature Dimensions**: 5000, 10000, 20000
- **Models**:
  - Logistic Regression
  - Random Forest
  - Naive Bayes
  - Linear SVM
  - XGBoost
- **Metrics Tracked**: Accuracy, F1-Score (weighted), Runtime (seconds)
- Benchmarked across all combinations and saved results for models with accuracy >= 81%

## Key Findings
- **Top model**: `LinearSVC` with TF-IDF (20k features): **83.3% accuracy** in under 2.1 seconds
- **TF-IDF** consistently outperforms CountVectorizer across all models and feature sizes
- Significant improvement in accuracy from 10k to 20k features, especially with linear models
- **Naive Bayes** performs reasonably well and is extremely fast
- **XGBoost** and **Random Forest** underperformed, especially on high-dimensional sparse vectors; also had the slowest runtimes

## Visuals
- Confusion matrices saved for each run
- CSV of all benchmark results: `model_benchmark_results.csv`

## Next Steps
- Test with **30,000+ features** to see if trend continues
- Use `GridSearchCV` or `Optuna` to **tune LinearSVC** (e.g., `C`, `loss`)
- Try **TruncatedSVD (LSA)** to reduce dimensionality and test performance drop
- Add **cross-validation** to better estimate generalization
- Later: experiment with **transformer embeddings** (e.g., `sentence-transformers`)

## Appendix
- All trained models (>= 81% accuracy) saved to `/models`
- Classification reports and confusion matrices saved in `/reports`
- Raw benchmark results CSV available at: `reports/model_benchmark_results.csv`

### Follow-Up: Higher-Dimensional TF-IDF Benchmarking (30k–50k)

We extended our feature space to 30k, 40k, and 50k dimensions using TF-IDF and CountVectorizer. 

- **Best result:** LinearSVC + TF-IDF (50k) achieved **84.5% accuracy**
- Performance increased slightly with each step
- Training time for LinearSVC remained fast (under 1.3s even at 50k)
- Logistic Regression showed similar patterns but with lower accuracy
- LinearSVM 30k outperforms all other models, including Logstic Regression 50k

Conclusion: We will continue to try higher feature counts before turning our focus towards tuning models and exploring dimensionality reduction.

## Follow-Up: Even-High-Dimensional TF-IDF Benchmarking (75k-full)

- **Best result:** LinearSVC + TF-IDF approximately **85.0% accuracy** with 200k, 250k and 300k
- Training time remained very fast (always under 2 seconds, even with full features)
- Performance peaked at 250k features

Conclusion: We have optimised our feature count at 250k for LinearSVM, and will now proceed to hyperparameter tuning with cross-validation.

## Follow-Up: SVM Tuning and Cross-Validation
Using `GridSearchCV` and 5-fold cross-validation, we tuned:
- `C`: [0.01, 0.1, 1, 10]
- `class_weight`: [None, "balanced"]

Results:
- Best parameters yielded slightly lower test accuracy (~84.85%) than original 250k peak
- However, results are now based on **cross-validated selection**, improving robustness and generalizability
- Confirms original model was well-chosen, and tuning reinforced its reliability

## Dimensionality Reduction Results — `6_dim_reduction.ipynb`

### Objective
Explore how much we can reduce the dimensionality of our TF-IDF vectors using `TruncatedSVD`, while preserving classification performance with `LinearSVC`.

---

### Setup
- Original TF-IDF: 200,000 features
- Reduction targets: 100, 300, 500, 1000 components
- Classifier: `LinearSVC`
- Evaluation: Test set accuracy and F1 score

---

### Performance vs Components

| Components | Accuracy | F1 Score |
|------------|----------|----------|
| 100        | 75.2%    | 74.7%    |
| 300        | 78.9%    | 78.6%    |
| 500        | 79.9%    | 79.7%    |
| 1000       | 81.1%    | 81.0%    |

- Accuracy improves steadily as we increase dimensionality
- Gap vs full TF-IDF model (~85.0%) remains, but narrows with 1000 components
- Strong potential for size/performance tradeoffs in deployment

---

### Top Words per Latent Component (SVD Interpretation)

Each component represents a latent **semantic axis** learned from TF-IDF patterns. Here's what some of them capture:

#### Component 0 — *City / Media Blend*
- High: `city`, `known`, `country`, `film`, `series`
- Captures both **geographical** and **entertainment** signals

#### Component 1 — *Capital Cities vs Media Terms*
- High: `city`, `capital`, `country`
- Low: `film`, `song`, `uk`, `number`
- A sharp divide between **location** and **pop culture**

#### Component 2 — *UK Music Terms*
- High: `uk`, `number`, `song`, `hit`, `band`, `single`
- Low: `film`, `series`
- Very **music-oriented**, especially UK chart terminology

#### Component 4 — *City/Country Polarity*
- High: `city`, `capital`
- Low: `country`, `used`, `word`
- Potentially distinguishes **urban vs. national** topics

#### Component 7 — *Sports Focus*
- High: `league`, `team`, `football`, `cup`
- Low: `game`, `gold`, `won`
- Very **sport-specific**

#### Component 8 — *Military & Government*
- High: `battle`, `war`, `minister`, `prime`, `british`
- Low: `film`, `known`, `word`
- Clear **historical/governmental** axis

---

### Conclusions
- `TruncatedSVD` components are interpretable and meaningful
- Possible to extract themes: music, geography, sports, politics, etc.
  - These themes map directly onto the category labels with which we are working
- The reduced models (~1000 components) retain decent performance and can be useful for:
  - Lightweight deployment
  - Exploratory analysis
  - Visualization

## Pause for analysis
- We performed analysis to see what sort of cases are giving our best LinearSVC model issues
- Two key findings:
  - 1) Many labels were incorrect in the ground truth, and these amounted for many error
    - We could go through and work out what percent of errors are of this sort
  - 2) Many categories have ambiguous boundaries, which accounts for many more errors
    - As suggested by the confusion matrix, Lifestyle and Entertainment, Music & Art, Geography & History are key culprits
- This suggests there may not be as much room for improvement from transformer-based models as the 85% accuracy initially suggests
  - Though this is the natural next step of the exercise in any case

---

## Data Cleaning
- In light of the above analysis, have decided to engage in some data cleaning.
- Added a Streamlit tool to efficiently go through the questions where model has highly confident incorrect predictions
- Relabels are saved in a corrected_labels.csv
  - Will re-incorporate into a new_cleaned_data set in future and re-run model to see if this impacts performance

## Next Steps
- Work on transformer-based embeddings to see if we can improve performance pass LinearSVC

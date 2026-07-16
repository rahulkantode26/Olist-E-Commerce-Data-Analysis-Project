# Olist-E-Commerce-Data-Analysis-Project
Welcome! This repo is built around the **Brazilian E-Commerce Public Dataset by Olist**. Olist is a Brazilian marketplace that connects small businesses to major sales channels. This dataset contains real (anonymized) orders made between 2016 and 2018.

# Intelligent Business Analytics & Forecasting System

An end-to-end machine learning mini-project built on the **Olist Brazilian E-Commerce
Public Dataset**, covering predictive modeling, customer segmentation, NLP-based review
analysis, and time-series sales forecasting.

## Dataset

[Olist](https://olist.com) is a Brazilian marketplace connecting small businesses to
major sales channels. The dataset covers **99,441 real, anonymized orders** placed
between September 2016 and October 2018, spread across 9 relational CSV files:

| File | Description |
|---|---|
| `olist_orders_dataset.csv` | One row per order — status, purchase/delivery timestamps |
| `olist_order_items_dataset.csv` | One row per item in an order — price, freight, product, seller |
| `olist_order_payments_dataset.csv` | One row per payment transaction — type, installments, value |
| `olist_order_reviews_dataset.csv` | One row per review — score, comment text |
| `olist_customers_dataset.csv` | Customer location; `customer_unique_id` identifies the real person |
| `olist_products_dataset.csv` | Product category, weight, and dimensions |
| `olist_sellers_dataset.csv` | Seller location |
| `olist_geolocation_dataset.csv` | Lat/long by zip code prefix |
| `product_category_name_translation.csv` | Portuguese → English category name lookup |

See the schema diagram in the notebook's introduction for how these tables connect
(`order_id`, `product_id`, `seller_id`, `customer_id`, and `zip_code_prefix`).

## Project Structure

```
├── olist_project.ipynb              # Full analysis — all 4 modules, run top to bottom
├── Model_Comparison_Report.docx     # 4-page summary report of all results
├── README.md                        # This file
└── data/                            # Place all 9 Olist CSVs here (or alongside the notebook — see Setup)
```

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy gensim statsmodels pmdarima prophet
```

The notebook downloads its own NLTK Portuguese stopwords and the spaCy `pt_core_news_sm`
model on first run (see Section 7.1). Place the 9 Olist CSV files in the same folder as
the notebook (or update `data_path` in Section 2 if you keep them elsewhere), then run
all cells in order.

## Modules & Models Used

### 1. Predictive Modeling (Regression + Classification)
- **Targets:** `delivery_time_days` (regression) and `is_late` (classification), both
  built from order/product/payment/location features known at order time.
- **Regression models:** Linear Regression, Ridge, Lasso, Random Forest Regressor
- **Classification models:** Logistic Regression, Decision Tree, Random Forest, KNN
- **Feature engineering:** missing-value imputation, one-hot encoding, standardization,
  `SelectKBest` feature selection

### 2. Customer Segmentation (Clustering)
- **Features:** RFM (Recency, Frequency, Monetary) + average review score + average
  delivery time, computed per unique customer (`customer_unique_id`)
- **Models:** KMeans (k=4, chosen via elbow/silhouette analysis), DBSCAN
- **Evaluation:** Silhouette score, PCA 2D visualization, written business interpretation

### 3. NLP Review Analysis
- **Task:** Binary sentiment classification (positive/negative) from `review_score`
- **Text pipeline:** lowercasing, punctuation removal, Portuguese stopword removal
  (NLTK), tokenization, TF-IDF vectorization
- **Model:** Logistic Regression
- **Extras:** optional Word2Vec word-similarity demo, spaCy-based Named Entity Recognition

### 4. Time Series Sales Forecasting
- **Target:** Total monthly sales revenue (Jan 2017 - Aug 2018, 20 clean months)
- **Models:** ARIMA, SARIMA, Prophet (via `pmdarima.auto_arima` for order selection)
- **Requirements covered:** stationarity check (ADF test), differencing, 12-month
  forward forecast, model comparison

### End-to-End Pipeline
Module 1's workflow was reformalized using `sklearn.Pipeline` + `ColumnTransformer` and
evaluated with 3-fold cross-validation, for a more production-realistic and robust
comparison than a single train/test split.

## Evaluation Results (Summary)

| Module | Best Model | Key Metric |
|---|---|---|
| Regression (delivery time) | Random Forest Regressor | R² = 0.275 (test), 0.268 (CV held-out) |
| Classification (is_late) | Random Forest (max_depth=10) | ROC-AUC = 0.744-0.745 |
| Clustering | KMeans (k=4) | Silhouette = 0.266, 4 interpretable segments |
| NLP Sentiment | Logistic Regression + TF-IDF | Accuracy = 90.4%, F1 = 0.929 |
| Time Series | ARIMA(1,1,0) | MAE = R$275,897 |

Full tables, confusion matrices, and cross-validated results are in the notebook and the
Model Comparison Report.

## Final Justification

Across independent modules, **delivery performance consistently emerges as the
dominant driver of customer experience** on this platform — it's the top feature in the
delivery-delay classifier, the clearest separator between the happiest and least-satisfied
customer segments, and the dominant theme in negative review text. This convergence
across three unrelated methods (supervised learning, clustering, and NLP) is the
project's strongest and most actionable finding.

Model selection favored **Random Forest** for both regression and classification tasks
(non-linear relationships in the data reward tree-based models), **KMeans** over DBSCAN
for actionable customer segments (with DBSCAN's noise points repurposed as an outlier
watchlist), **Logistic Regression + TF-IDF** for sentiment (strong accuracy without the
overhead of deep learning, appropriate for this dataset size), and **ARIMA** over SARIMA
and Prophet for forecasting (the dataset's 20-month history is too short to reliably
support the seasonal or trend-flexible components the other two models rely on) — a
deliberate example of choosing the simpler model when the data doesn't support a more
complex one, rather than defaulting to the most sophisticated option available.

## Known Limitations

- **~97% of customers ordered only once** — clustering separates mainly on spending/
  recency/experience rather than repeat-purchase frequency.
- **Only 20 months of clean sales history** — SARIMA's seasonal component and Prophet's
  yearly seasonality could not be reliably estimated; forecasts beyond ~3 months carry
  wide uncertainty.
- **`is_late` is imbalanced (~8% positive)** — accuracy alone is misleading here; recall
  and ROC-AUC were weighted more heavily in model selection.
- **Spam detection was scoped out** — the dataset has no spam label to train or validate
  against.

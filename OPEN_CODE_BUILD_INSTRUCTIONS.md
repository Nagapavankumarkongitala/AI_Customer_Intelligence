# OpenCode Build Instructions: AI Customer Intelligence Platform

> **Purpose:** Give this file to OpenCode as the project execution specification. OpenCode must build the project phase by phase, preserve one integrated codebase, test every phase, and never replace working implementations with placeholders.

---

## 1. Project Mission

Build a production-oriented **AI Customer Intelligence Platform** using **Python and Streamlit** for:

- Customer transaction-data upload and validation
- Data-quality assessment and cleaning
- Customer-level feature engineering
- RFM and LRFM customer analysis
- Demographic and geographic profiling
- Advanced customer segmentation
- Traditional-versus-AI segmentation comparison
- Time-aware churn prediction
- Explainable churn scoring
- Revenue-at-risk calculation
- Retention-strategy recommendations
- Executive dashboards and downloadable outputs

The application must use the following available source fields:

```text
customer_id
transaction_date
amount_spent
gender
age
city
state
country
region
```

The first three fields are required. The remaining fields are optional but must be used when available.

---

## 2. Non-Negotiable Execution Rules

OpenCode must follow these rules throughout the build.

1. Maintain **one integrated repository**. Do not create disconnected phase projects.
2. Complete phases sequentially. Do not start a later phase until the current phase passes its acceptance checks.
3. Write real implementations. Do not leave `pass`, placeholder functions, mock outputs, or unfinished TODO-only modules.
4. Preserve all working files from earlier phases.
5. Use modular services and analytics modules. Do not put all business logic in Streamlit page files.
6. Keep Streamlit pages focused on presentation, interaction, validation messages, and orchestration.
7. Add type hints and docstrings to public functions.
8. Use fixed random seeds for reproducibility.
9. Add unit tests for analytical and transformation functions.
10. Run formatting, linting, compilation, and tests at the end of every phase.
11. Never silently delete or alter customer records. Record every cleaning action in an audit report.
12. Do not define churn from current recency and then use the same current recency to predict it. Use separate observation and outcome windows.
13. Do not assume that more complex models are better. Compare models using technical, stability, interpretability, and business criteria.
14. Cluster numbers are arbitrary. Assign business labels only after profiling the clusters.
15. Treat demographic fields carefully. Use them for profiling, fairness review, and optional modeling only when justified.
16. Never expose raw customer identifiers in executive views unless a user explicitly drills down into an authorized detail view.
17. All errors shown in the UI must be actionable and written in business-readable language.
18. Every phase must update the README, the changelog, tests, and requirements when necessary.

---

## 3. Technology Stack

Use the following core stack:

```text
Python 3.11+
Streamlit
Pandas
NumPy
Plotly
Scikit-learn
SciPy
OpenPyXL
PyArrow
Joblib
PyYAML
Pydantic
Pytest
Ruff
Black
```

Add these packages only in the phases where required:

```text
hdbscan
lightgbm
xgboost
shap
tensorflow or torch
umap-learn
optuna
```

Prefer optional imports and helpful installation messages for heavy or platform-sensitive packages.

---

## 4. Target Repository Structure

Create and preserve the following structure:

```text
AI_Customer_Intelligence/
|
|-- app.py
|-- README.md
|-- CHANGELOG.md
|-- PROJECT_STATUS.md
|-- requirements.txt
|-- requirements-dev.txt
|-- pyproject.toml
|-- config.yaml
|-- .env.example
|-- .gitignore
|-- run_windows.ps1
|-- run_linux_mac.sh
|
|-- .streamlit/
|   `-- config.toml
|
|-- pages/
|   |-- 01_Data_Upload_and_Mapping.py
|   |-- 02_Data_Quality_and_Cleaning.py
|   |-- 03_RFM_LRFM_Analysis.py
|   |-- 04_Demographic_Profiling.py
|   |-- 05_Advanced_Segmentation.py
|   |-- 06_Segmentation_Comparison.py
|   |-- 07_Churn_Prediction.py
|   |-- 08_Retention_Strategies.py
|   `-- 09_Executive_Dashboard.py
|
|-- analytics/
|   |-- __init__.py
|   |-- customer_features.py
|   |-- rfm.py
|   |-- lrfm.py
|   |-- demographics.py
|   |-- customer_health.py
|   |-- revenue_at_risk.py
|   `-- statistical_tests.py
|
|-- services/
|   |-- __init__.py
|   |-- data_service.py
|   |-- column_detection.py
|   |-- profiling_service.py
|   |-- cleaning_service.py
|   |-- data_dictionary_service.py
|   |-- feature_service.py
|   |-- segmentation_service.py
|   |-- churn_service.py
|   |-- retention_service.py
|   |-- export_service.py
|   `-- state_service.py
|
|-- feature_engineering/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- behavioral_features.py
|   |-- categorical_features.py
|   |-- churn_features.py
|   `-- pipelines.py
|
|-- segmentation/
|   |-- __init__.py
|   |-- common.py
|   |-- kmeans_model.py
|   |-- gmm_model.py
|   |-- hierarchical_model.py
|   |-- hdbscan_model.py
|   |-- autoencoder_model.py
|   |-- ensemble_model.py
|   |-- evaluation.py
|   |-- stability.py
|   |-- segment_naming.py
|   `-- segment_profiling.py
|
|-- churn/
|   |-- __init__.py
|   |-- labeling.py
|   |-- dataset_builder.py
|   |-- baseline_model.py
|   |-- random_forest_model.py
|   |-- boosting_model.py
|   |-- evaluation.py
|   |-- calibration.py
|   |-- explainability.py
|   `-- risk_scoring.py
|
|-- retention_engine/
|   |-- __init__.py
|   |-- value_tiers.py
|   |-- priority_scoring.py
|   |-- strategy_mapper.py
|   |-- next_best_action.py
|   `-- campaign_export.py
|
|-- visualization/
|   |-- __init__.py
|   |-- theme.py
|   |-- quality_charts.py
|   |-- rfm_charts.py
|   |-- demographic_charts.py
|   |-- segmentation_charts.py
|   |-- churn_charts.py
|   |-- retention_charts.py
|   `-- executive_charts.py
|
|-- components/
|   |-- __init__.py
|   |-- layout.py
|   |-- metric_cards.py
|   |-- filters.py
|   |-- status_panel.py
|   |-- help_text.py
|   `-- download_buttons.py
|
|-- utils/
|   |-- __init__.py
|   |-- constants.py
|   |-- validators.py
|   |-- logging_config.py
|   |-- file_utils.py
|   |-- model_utils.py
|   `-- exceptions.py
|
|-- models/
|   |-- segmentation/
|   |-- churn/
|   `-- metadata/
|
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- features/
|   `-- sample/
|
|-- reports/
|   |-- csv/
|   |-- excel/
|   |-- figures/
|   `-- model_cards/
|
|-- logs/
|
`-- tests/
    |-- conftest.py
    |-- test_preprocessing.py
    |-- test_column_detection.py
    |-- test_cleaning.py
    |-- test_quality.py
    |-- test_customer_features.py
    |-- test_rfm.py
    |-- test_lrfm.py
    |-- test_demographics.py
    |-- test_segmentation.py
    |-- test_churn_labeling.py
    |-- test_churn_models.py
    |-- test_retention.py
    `-- test_exports.py
```

Place `.gitkeep` files in otherwise empty runtime directories.

---

## 5. Canonical Data Contract

The application must map uploaded source columns to this canonical schema:

```yaml
required:
  customer_id: string
  transaction_date: datetime
  amount_spent: numeric

optional:
  gender: category
  age: numeric
  city: category
  state: category
  country: category
  region: category
```

Rules:

- `customer_id` must not be blank in the modeling dataset.
- `transaction_date` must be parseable and must not be unexpectedly in the future.
- `amount_spent` must be numeric after cleaning.
- Negative values must not be converted automatically unless the user selects that treatment and understands that negatives may represent refunds.
- `age` must be configurable, with a default valid range of 0 to 120.
- Categorical values must be whitespace-normalized but not aggressively merged without documentation.

---

# PHASE 0: Repository Initialization

## Goal

Create a reproducible Python project and a working Streamlit shell before implementing analytical features.

## Tasks

1. Create the target directory structure.
2. Add `__init__.py` files to importable packages.
3. Add `.gitignore` entries for virtual environments, caches, logs, models, secrets, and runtime data.
4. Add `.env.example` without real credentials.
5. Add `pyproject.toml` for Black, Ruff, and Pytest configuration.
6. Add separate runtime and development requirements.
7. Add Windows and Linux/macOS startup scripts.
8. Add a Streamlit theme in `.streamlit/config.toml`.
9. Add centralized logging.
10. Create `PROJECT_STATUS.md` containing completed, in-progress, and pending phases.
11. Create a landing page in `app.py` with project purpose, current phase status, expected input fields, and navigation explanation.

## Commands

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Acceptance Gate

```powershell
python -m compileall .
ruff check .
black --check .
pytest -q
streamlit run app.py
```

The phase passes only if the application loads without import errors and the checks succeed.

---

# PHASE 1: Complete Data Foundation

## Goal

Create a reliable transaction-data ingestion, mapping, quality, cleaning, governance, and export layer.

## Page 1: Data Upload and Mapping

Implement:

- CSV, XLSX, XLS, and Parquet uploads
- CSV encoding fallback: UTF-8 BOM, UTF-8, CP1252, Latin-1
- Empty-file validation
- Unsupported-format validation
- Normalized column names
- Duplicate normalized-column suffixes
- File preview
- Row, column, memory, and duplicate counts
- Automated alias and fuzzy column detection
- Manual mapping controls
- Required-field status
- Duplicate source mapping prevention
- Mapping JSON download
- Validated dataset storage in Streamlit session state

Use aliases for likely variations such as:

```text
customer_id: customer_id, customerid, cust_id, client_id, member_id, user_id
transaction_date: transaction_date, purchase_date, order_date, invoice_date, txn_date, date
amount_spent: amount_spent, amount, sales, revenue, spend, transaction_amount, total_amount
gender: gender, sex
age: age, customer_age
city: city, town
state: state, province
country: country, nation
region: region, zone, territory
```

## Page 2: Data Quality and Cleaning

Calculate:

```text
Completeness = (1 - missing_cells / total_cells) * 100
Validity = (1 - invalid_records / evaluated_records) * 100
Uniqueness = (1 - duplicate_rows / total_rows) * 100
Overall quality = 0.40 * Completeness + 0.40 * Validity + 0.20 * Uniqueness
```

Check:

- Missing cells
- Missing customer IDs
- Exact duplicate rows
- Invalid dates
- Future transaction dates
- Invalid monetary values
- Negative amounts
- Invalid ages
- Monetary outliers
- Category inconsistencies

Support user-configurable treatments:

- Drop or retain duplicates
- Drop or retain missing customer IDs
- Drop or retain invalid dates as missing
- Drop, retain, fill zero, or fill median for missing amounts
- Keep, drop, set missing, or take absolute value for negative amounts
- Flag, cap, remove, or ignore IQR outliers
- Set missing, drop, or keep invalid ages

Generate an audit record containing:

```text
input row count
output row count
rows removed
selected cleaning rules
records affected by each rule
execution timestamp
source filename
column mapping
```

## Page 3: Data Dictionary

For every field, show:

- Source column name
- Canonical business field
- Description
- Required status
- Data type
- Non-null count
- Missing count and percentage
- Unique count
- Example values
- Minimum, maximum, mean, and median where applicable

## Phase 1 Tests

Test:

- Header normalization
- Duplicate normalized headers
- Currency parsing
- Date parsing
- Automatic column mapping
- Mapping validation
- Duplicates
- Missing identifiers
- Negative amount treatments
- Invalid age treatments
- IQR outlier treatments
- Quality score boundaries
- CSV and Excel exports

## Phase 1 Acceptance Gate

The phase passes only when a deliberately dirty sample file can be uploaded, mapped, cleaned, audited, previewed, and downloaded without code changes.

---

# PHASE 2: Customer Features and RFM/LRFM

## Goal

Transform cleaned transaction records into a reusable customer-level feature table and interpretable RFM/LRFM segments.

## Customer Feature Engineering

Use a snapshot date equal to one day after the last valid transaction unless the user selects another date.

Calculate:

```text
first_transaction_date
last_transaction_date
recency_days
frequency
monetary_value
relationship_length_days
average_order_value
active_months
transactions_per_active_month
average_interpurchase_days
interpurchase_std_days
recent_period_frequency
previous_period_frequency
frequency_change
recent_period_monetary
previous_period_monetary
monetary_change
```

Use customer-level demographic and geographic values selected through a documented rule. Default to the latest non-null observed value, not arbitrarily `first`, and record conflicts where a customer has multiple values.

## RFM Scoring

- Support 3, 4, and 5 quantile levels.
- Reverse recency scores.
- Score frequency and monetary value in ascending order.
- Use ranked values to handle ties.
- Add safe behavior for small datasets and low-cardinality features.
- Store raw metrics and score components separately.

```text
RFM score = 100 * R + 10 * F + M
LRFM score = 1000 * L + 100 * R + 10 * F + M
```

## Segment Naming

Define transparent, configurable rules for:

```text
Champions
Loyal Customers
Potential Loyalists
New Customers
Promising
Needs Attention
At Risk
Cannot Lose Them
Hibernating
Lost Customers
```

Do not label customers from the combined numeric score alone. Use score components and raw-value validation.

## RFM/LRFM Page

Include:

- KPI cards
- RFM and LRFM configuration
- Customer feature preview
- Segment distribution
- Revenue contribution by segment
- Recency-frequency scatter plot
- Frequency-monetary scatter plot
- RFM heatmap
- Segment profile table
- Customer drill-down
- Downloadable feature and segment files

## Phase 2 Tests

Test date boundaries, one-time customers, ties, missing optional demographics, all-equal values, small populations, and score directions.

## Phase 2 Acceptance Gate

Every cleaned transaction dataset with at least the three required fields must generate a customer-level table and valid RFM/LRFM output without manual code modifications.

---

# PHASE 3: Demographic and Geographic Profiling

## Goal

Explain who the customers are and how customer profiles differ across traditional segments.

## Implement

- Configurable age bands
- Average and median age
- Gender distribution
- Customer and revenue counts by city, state, country, and region
- Demographic profiles by RFM/LRFM segment
- Geographic concentration
- Missing demographic indicators
- Population-versus-segment comparisons

## Charts

- Age histogram
- Age-band bar chart
- Gender distribution
- Top cities and states
- Region and country treemap
- Revenue contribution by geography
- Stacked demographic composition by segment
- Optional map only when geographic values can be reliably geocoded

Do not add unreliable map coordinates or fabricate locations.

## Statistical Tests

Where suitable, provide:

- ANOVA, Welch ANOVA, or Kruskal-Wallis for continuous fields across segments
- Chi-square or Fisher exact test for categorical associations
- Effect sizes such as Cramer's V
- Multiple-comparison correction when many hypotheses are tested

## Acceptance Gate

The page must work even when some or all optional demographic fields are absent. Missing fields must produce helpful notices rather than crashes.

---

# PHASE 4: Advanced Segmentation

## Goal

Build reproducible, comparable, and interpretable unsupervised customer segmentation.

## Feature Pipeline

Default candidate features:

```text
recency_days
frequency
monetary_value
relationship_length_days
average_order_value
transactions_per_active_month
average_interpurchase_days
interpurchase_std_days
age
```

Implement:

- Missing-value imputation
- Optional `log1p` transformation for nonnegative skewed features
- Robust or standard scaling
- Correlation review
- Variance checks
- User-selectable features
- Saved fitted preprocessing pipeline

## Models

Implement separately:

1. K-Means
2. Gaussian Mixture Model
3. Agglomerative clustering
4. HDBSCAN
5. Autoencoder embeddings followed by clustering
6. Ensemble or consensus clustering

The autoencoder model must:

- Be optional
- Require a sensible minimum sample size
- Use early stopping
- Save training history
- Produce a latent feature table
- Fall back gracefully when the deep-learning package is unavailable

The ensemble model must use label alignment, co-association, or consensus logic. Never use a direct statistical mode of arbitrary cluster labels.

## Model Evaluation

Calculate:

- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz index
- Cluster counts and proportions
- Noise percentage where relevant
- Minimum cluster size
- Stability across multiple random seeds or bootstrap samples

## Visuals

- PCA 2D scatter
- Optional UMAP scatter
- 3D scatter when useful
- Cluster profile heatmap
- Radar chart used sparingly
- Segment size and revenue contribution
- Customer table with cluster labels

## Segment Naming

Generate names only after profiling clusters. Compare cluster metrics with population medians or percentile ranks. Keep cluster ID and business name as separate fields.

## Acceptance Gate

At least K-Means, GMM, and hierarchical clustering must run on a representative dataset. Optional HDBSCAN and autoencoder dependencies must fail gracefully with installation guidance.

---

# PHASE 5: Segmentation Comparison

## Goal

Compare interpretable rules with advanced segmentation fairly.

## Implement

- RFM/LRFM versus each AI model
- Model metric comparison
- Cluster-size balance
- Revenue distribution
- Customer-value separation
- Churn-rate association when churn labels become available
- Cross-tabulation
- Sankey mapping
- Segment profile variance
- Stability comparison
- Business interpretability checklist

Create a model recommendation summary containing:

```text
best statistical separation
best stability
best interpretability
best revenue differentiation
recommended operational model
recommendation rationale
```

Do not automatically choose the model with the highest Silhouette score.

## Acceptance Gate

The page must explain why a model is recommended and expose enough evidence for a human to disagree with the recommendation.

---

# PHASE 6: Leakage-Safe Churn Prediction

## Goal

Predict future inactivity using historical behavior without target leakage.

## Time-Aware Dataset Construction

Users must configure:

```text
observation start
cutoff date
outcome horizon in days
minimum customer history
```

Build predictors from transactions on or before the cutoff date. Define churn from transactions after the cutoff date and within the outcome window.

```text
churn = 1 if no transaction occurs in (cutoff, cutoff + horizon]
churn = 0 if at least one transaction occurs in that window
```

Validate that the outcome window exists in the source data.

## Models

Implement:

1. Logistic Regression baseline
2. Random Forest
3. HistGradientBoosting or equivalent Scikit-learn boosting model
4. LightGBM when installed
5. XGBoost when installed

Use preprocessing pipelines with:

- Numeric imputation
- Categorical imputation
- One-hot encoding
- Scaling where required
- Class weighting
- Time-based training, validation, and test splits

Apply sampling only to training data.

## Evaluation

Include:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Balanced accuracy
- Confusion matrix
- Log loss
- Brier score
- Calibration curve
- Lift and cumulative gain
- Threshold configuration

## Explainability

- Logistic Regression coefficients and odds ratios
- Tree feature importance
- Permutation importance
- SHAP for compatible models
- Global and customer-level explanations

## Outputs

```text
customer_id
churn_probability
risk_band
predicted_class
threshold
primary_risk_drivers
model_version
score_timestamp
```

## Acceptance Gate

The model must be tested on a chronological holdout. The page must display a leakage warning if the user tries to use post-cutoff information as a predictor.

---

# PHASE 7: Retention Intelligence

## Goal

Turn customer value and churn risk into auditable decision-support actions.

## Calculate

```text
customer_value_tier
expected_value_proxy
churn_probability
revenue_at_risk
retention_priority_score
recommended_action
recommended_channel
suggested_offer_type
action_reason
```

Use:

```text
Revenue at Risk = churn_probability * selected customer value
Priority = normalized value * churn_probability * optional strategic weight
```

Clearly label historical monetary value as historical value, not true future CLV.

## Strategy Matrix

At minimum support:

```text
High value + high risk -> personal outreach, loyalty recovery, VIP intervention
High value + low risk -> recognition, cross-sell, referral, early access
Medium value + high risk -> personalized reactivation and limited incentive
Low value + high risk -> automated low-cost re-engagement
Low value + low risk -> routine nurture and monitoring
```

The rules must be externally configurable in YAML or JSON.

## Retention Page

Include:

- Value-risk matrix
- Revenue at risk
- Priority customer count
- Strategy distribution
- Region and segment filters
- Ranked customer table
- Action explanation
- Campaign-ready export

## Acceptance Gate

Every recommended action must have an explicit reason and traceable rule. The application must not claim causal retention impact without experiment data.

---

# PHASE 8: Executive Dashboard and Production Readiness

## Goal

Deliver a polished management interface and prepare the repository for deployment.

## Executive Dashboard Layout

### Row 1: KPI cards

```text
Total customers
Total revenue
Average order value
Active customer rate
Predicted churn rate
Revenue at risk
Priority retention opportunities
```

### Row 2: Trends

- Revenue trend
- Customer activity trend
- New and returning customers

### Row 3: Segment portfolio

- Customer share by segment
- Revenue share by segment
- Segment performance table

### Row 4: Risk

- Risk-band distribution
- Revenue-at-risk chart
- High-value/high-risk customers

### Row 5: Demographics and geography

- Region/country contribution
- Age bands
- Gender and segment composition when available

### Row 6: Retention

- Recommended-action distribution
- Campaign-ready customer count
- Download controls

## Global Filters

```text
date range
country
region
state
city
gender
age band
RFM segment
AI segment
risk band
value tier
```

All dashboard cards and visualizations must update consistently from the selected filters.

## UI/UX Rules

- Use a restrained professional theme.
- Use blue for neutral information, green for positive states, amber for warnings, and red for high risk.
- Do not rely on color alone.
- Use business labels rather than raw variable names.
- Add tooltips and concise metric explanations.
- Hide advanced settings in expanders.
- Use skeleton or spinner states for long-running operations.
- Provide clear empty-state and error messages.
- Keep tables sortable and searchable.
- Limit initial charts to decision-relevant visuals.

## Production Improvements

- Centralized exception handling
- Structured logging
- Configuration validation
- Model metadata and model cards
- Persistent storage abstraction
- Role-based access design
- Privacy masking
- Export authorization hooks
- Dockerfile
- Health check
- CI workflow
- Test coverage report
- Deployment guide

## Final Acceptance Gate

Run:

```powershell
python -m compileall .
ruff check .
black --check .
pytest -q --cov=. --cov-report=term-missing
streamlit run app.py
```

Perform a complete manual workflow:

1. Upload a dirty sample dataset.
2. Correct automatic mappings.
3. Clean and export the transaction dataset.
4. Generate customer features.
5. Build RFM and LRFM segments.
6. View demographic profiles.
7. Build and compare AI segments.
8. Create a valid time-aware churn dataset.
9. Train and evaluate churn models.
10. Score customers and explain selected scores.
11. Generate retention strategies.
12. Filter the executive dashboard.
13. Export customer and management outputs.
14. Restart the application and confirm documented restart behavior.

The project is complete only when this workflow succeeds without editing source code.

---

## 6. Streamlit Session-State Contract

Use consistent session-state keys:

```python
raw_data
clean_data
column_mapping
cleaning_config
cleaning_audit
data_dictionary
customer_features
rfm_results
lrfm_results
demographic_profile
segmentation_runs
selected_segmentation_model
segmented_customers
churn_dataset
churn_model
churn_model_metadata
churn_scores
retention_results
global_filters
```

Create a `state_service.py` module to initialize, validate, reset, and report the status of these objects.

---

## 7. Configuration Requirements

Keep configurable business settings in `config.yaml`:

```yaml
app:
  name: AI Customer Intelligence Platform
  currency_symbol: "₹"
  random_seed: 42

quality:
  age_min: 0
  age_max: 120
  future_date_tolerance_days: 0
  outlier_iqr_multiplier: 1.5

rfm:
  quantiles: 5
  snapshot_offset_days: 1

segmentation:
  minimum_customers: 50
  cluster_range: [2, 10]
  default_features:
    - recency_days
    - frequency
    - monetary_value
    - relationship_length_days
    - average_order_value

churn:
  default_outcome_horizon_days: 90
  risk_bands:
    low: [0.0, 0.30]
    medium: [0.30, 0.60]
    high: [0.60, 0.80]
    critical: [0.80, 1.0]

retention:
  value_quantiles: 3
  strategic_weight_default: 1.0
```

Validate configuration at startup and provide clear errors for invalid values.

---

## 8. Sample Data Generator

Create a reproducible synthetic-data generator only for demonstration and automated testing.

Requirements:

- Clearly mark generated data as synthetic.
- Include all canonical fields.
- Generate repeated transactions per customer.
- Include realistic date ranges and spending distributions.
- Include controlled missing values, duplicates, invalid dates, negative refunds, and outliers.
- Include at least 1,000 customers for performance testing.
- Never mix generated sample data with a user's uploaded dataset without explicit action.

Output:

```text
data/sample/sample_customer_transactions.csv
```

---

## 9. Documentation Requirements

At the end of every phase, update:

### README.md

Include:

- Project purpose
- Implemented phases
- Input schema
- Installation
- Windows run instructions
- Linux/macOS run instructions
- Test commands
- Application workflow
- Known limitations
- Privacy notes

### PROJECT_STATUS.md

Include:

```text
Current phase
Completed features
Current test results
Known issues
Next phase
```

### CHANGELOG.md

Use dated entries and list added, changed, fixed, and removed items.

### Model cards

For every final segmentation and churn model, document:

- Model purpose
- Training data period
- Input features
- Preprocessing
- Hyperparameters
- Evaluation metrics
- Limitations
- Fairness considerations
- Model version and date

---

## 10. OpenCode Working Protocol

OpenCode must execute the work using this cycle for every phase:

```text
1. Inspect the current repository.
2. Read PROJECT_STATUS.md and CHANGELOG.md.
3. Confirm the exact phase and acceptance criteria.
4. Create or update the implementation plan.
5. Implement the smallest coherent module set.
6. Add or update tests.
7. Run compilation, linting, formatting, and tests.
8. Fix failures before continuing.
9. Run the Streamlit application and inspect the page behavior.
10. Update documentation and project status.
11. Commit or checkpoint the completed phase.
12. Produce a concise completion report.
```

Do not ask for confirmation between ordinary implementation steps. Stop and request clarification only when a missing business rule prevents a safe implementation.

---

## 11. Required Completion Report After Each Phase

OpenCode must report:

```markdown
## Phase Completion Report

### Phase
Name and number

### Files Added
- path

### Files Changed
- path

### Features Completed
- feature

### Test Results
- compilation
- lint
- format
- unit tests
- integration tests

### Manual Validation
- workflow tested

### Known Limitations
- limitation

### Next Phase
- next planned phase
```

---

## 12. First Command to Give OpenCode

Use the following instruction after placing this file in the repository root:

```text
Read OPEN_CODE_BUILD_INSTRUCTIONS.md completely. Inspect the existing repository and preserve all working code. Start with Phase 0 if repository configuration is incomplete; otherwise verify Phase 1 against every listed requirement and acceptance criterion. Implement missing items, add tests, run all quality checks, update README.md, CHANGELOG.md, and PROJECT_STATUS.md, and provide the required phase completion report. Do not start Phase 2 until Phase 1 passes its acceptance gate. Do not create placeholder modules.
```

After Phase 1 is complete, use:

```text
Continue with the next incomplete phase in OPEN_CODE_BUILD_INSTRUCTIONS.md. Preserve the integrated repository, implement every requirement for that phase, add tests, run all acceptance checks, update project documentation, and provide the phase completion report. Do not proceed to the following phase until the current phase passes.
```

---

## 13. Definition of Done

The project is done when:

- All eight development phases are implemented in one repository.
- Required pages work with the canonical data contract.
- There are no placeholders or unimplemented functions in production modules.
- The end-to-end workflow succeeds without editing code.
- Models are leakage-safe, evaluated, explainable, and versioned.
- Retention actions are traceable and do not overclaim causal impact.
- The executive dashboard is filterable, readable, and exportable.
- Automated tests pass.
- Linting and formatting checks pass.
- Setup and operating documentation are complete.
- The application can be installed and run from a clean environment using the documented commands.

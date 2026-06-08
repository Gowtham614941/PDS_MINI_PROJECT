# Smart Expense Analyzer

A Python-based **Smart Expense Analyzer** designed as a 2nd-semester project for Artificial Intelligence & Machine Learning (AIML) students. The project showcases basic programming principles (Object-Oriented Programming, file handling) integrated with foundational data science and machine learning concepts.

---

## 🌟 Features
1. **Interactive Dashboard**: CLI-based menu to perform analysis, predict spending, view visualizations, and record transactions.
2. **AI-Based Categorization**: Predicts the category (e.g., Food, Travel, Rent) of new expenses using a **Naive Bayes Classifier** trained on transaction descriptions.
3. **Statistical Anomaly Detection**: Uses **Z-Score** analysis (via NumPy) to identify if a transaction is an outlier compared to previous spending patterns.
4. **Predictive Budget Forecasting**: Fits a **Linear Regression** model on daily transaction totals to predict spending trends over the next 7 and 30 days.
5. **Data Visualization**: Generates rich Matplotlib visualizations (Pie charts for category breakdowns, Line graphs for spending history, and Bar charts for Budget vs. Actual comparisons).
6. **CSV Database**: Stores transactions locally in a structured CSV format, providing automatic simulation of seed data if empty.

---

## 📐 The ML & Analysis Concepts Explained

### 1. Expense Categorization: Naive Bayes & TF-IDF
* **What it does**: Takes a transaction description (e.g., `"ubereats dinner"`) and classifies it into a category (e.g., `Food`).
* **How it works**:
  * **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts text descriptions into numbers. It measures how important a word is to a document relative to a corpus of documents.
  * **Multinomial Naive Bayes (MNB)**: A classification algorithm based on Bayes' Theorem:
    $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
    It calculates the probability of each category given the words in the description and assigns the category with the highest probability.

### 2. Anomaly Detection: Z-Score
* **What it does**: Detects if you spent an unusually large amount on a transaction compared to your average spending in that category.
* **How it works**:
  * A **Z-Score** measures how many standard deviations ($\sigma$) a data point ($x$) is from the mean ($\mu$):
    $$Z = \frac{x - \mu}{\sigma}$$
  * If the calculated Z-score is greater than **2.0**, it means the spending is in the top ~5% of transactions (an outlier) and triggers an alert.

### 3. Forecasting: Simple Linear Regression
* **What it does**: Analyzes daily cumulative spending trends and forecasts future spending.
* **How it works**:
  * **Linear Regression** models the relationship between time (independent variable $X$, in days) and cumulative expense (dependent variable $Y$, in dollars/rupees):
    $$Y = mX + c$$
  * Where $m$ is the slope (rate of spending) and $c$ is the intercept. The model is trained using Least Squares to predict future spending bounds.

---

## 📂 Project Structure
```
d:/pds/
├── requirements.txt         # Package dependencies
├── README.md                # This documentation
├── main.py                  # Entrypoint / CLI Menu Loop
├── data/
│   └── expenses.csv         # Database containing expense history
└── src/
    ├── __init__.py          # Marks src as a Python package
    ├── utils.py             # CSV operations, budget management, seed data generator
    ├── parser.py            # User input reading and date/numeric validations
    ├── cli_ui.py            # CLI visual styles, menus, and grid table outputs
    ├── classifier.py        # TF-IDF + Naive Bayes text classification pipeline
    ├── analyzer.py          # Math calculations (Z-score & Linear Regression)
    └── visualizer.py        # Matplotlib visualization generator
```

---

## 🚀 How to Run the Project

### Prerequisites
Make sure Python 3.8+ is installed on your computer.

### Step 1: Install Dependencies
Open your terminal inside the project directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
Execute the main file:
```bash
python main.py
```

### Step 3: Interactive CLI
Follow the prompts on the terminal screen to:
* Add new expenses (watch the AI auto-fill the category!).
* View summary statistics.
* Run anomaly detection and forecasting.
* Generate and view charts.

# Smart Expense Analyzer (Multi-Platform Project)

A clean, feature-rich **Smart Expense Analyzer** designed as a 2nd-semester project for Artificial Intelligence & Machine Learning (AIML) students. The project comes in two versions:
1. **Interactive Web Dashboard** (HTML5 + CSS3 + Client-Side JavaScript + Chart.js)
2. **Interactive CLI Dashboard** (Self-contained Python Script using Pandas, NumPy, Scikit-learn, and Matplotlib)

Both versions implement the exact same math, machine learning, and statistical models.

---

## 📂 Project Structure
```
d:/pds/
├── index.html       # Web App: Main Single-Page HTML Layout
├── styles.css       # Web App: Premium Glassmorphic Stylesheet
├── app.js           # Web App: Client-Side JS (ML models, math, data grids, charts)
├── main.py          # Python CLI: Single-file interactive Python version
├── requirements.txt # Python CLI: Dependencies file
└── README.md        # This Documentation
```

---

## 💻 1. Interactive Web Application (Recommended)

### 🌟 Features
* **Premium Glassmorphic UI**: Sleek dark space-indigo design (`#080b11`) with animated transitions, tab navigation, hover micro-animations, and responsive layouts.
* **Client-Side Storage**: Leverages browser `LocalStorage` to persist transaction entries and category budgets across reloads.
* **Auto-Seeding**: Automatically populates 60 realistic transactions spanning the last 30 days if storage is empty, providing immediate data to train the models.
* **Dynamic Chart.js Visualizations**:
  - Interactive **Donut Chart** detailing category spending distribution.
  - Side-by-side **Horizontal Bar Chart** comparing budget limits versus actual totals.
  - **Regression Trend Curve** tracking actual daily spends and overlaying the forecasting line.

### ⚙️ How to Run
1. Navigate to the project folder `d:\pds`.
2. Double-click **[index.html](file:///d:/pds/index.html)** to open it directly in any modern web browser (no installation, web server, or Node.js required!).

---

## 🐍 2. Interactive CLI Application (Python)

### 🌟 Features
* **Formatted Console Grid Tables**: Custom-designed Unicode tables with colored indicator rows representing budget caps.
* **Real-Time Warning Banner**: Employs NumPy computations to audit and trigger Z-score warnings immediately upon logging anomalous amount values.
* **Visual Plots Output**: Draws category pie charts, budget comparison bars, and regression line fits using Matplotlib.

### ⚙️ How to Run
1. Open your terminal inside `d:\pds`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   python main.py
   ```

---

## 📐 The ML & Analysis Mathematics Explained

### 1. Expense Categorization: Naive Bayes & TF-IDF
* **Description**: Predictions auto-classify descriptions into categories (e.g. `"mcdonalds burger"` -> `Food`).
* **Mathematics**: Uses Bayes' Theorem to find the category $C$ that maximizes the posterior log probability given the words in the description:
  $$\log P(C) + \sum_{i=1}^{n} \log P(W_i | C)$$
  Where the conditional probabilities are smoothed using **Laplace Smoothing** to prevent zero frequency values:
  $$P(W_i | C) = \frac{\text{count}(W_i, C) + 1}{\text{total words in } C + \text{vocabulary size} + 1}$$

### 2. Outlier Detection: Z-Score Audit
* **Description**: Flags transaction amounts that deviate significantly from category spending norms.
* **Mathematics**: Calculates how many standard deviations ($\sigma$) a new amount $X$ is from the historical category mean ($\mu$):
  $$Z = \frac{X - \mu}{\sigma}$$
  If the calculated Z-score is greater than **2.0** (falling in the top ~5% tails), the system flags the transaction as an anomaly.

### 3. Forecasting: Least-Squares Linear Regression
* **Description**: Extrapolates daily cumulative spending to project expenses over the next 7 and 30 days.
* **Mathematics**: Fits a line of best fit $Y = mX + c$ through daily cumulative totals, calculating:
  - **Slope (m)** (daily spending accumulation rate):
    $$m = \frac{N\sum(XY) - \sum X \sum Y}{N\sum(X^2) - (\sum X)^2}$$
  - **Intercept (c)**:
    $$c = \frac{\sum Y - m\sum X}{N}$$
  - **R-Squared ($R^2$)** (Coefficient of determination evaluating fit quality):
    $$R^2 = 1 - \frac{\sum(Y_{\text{actual}} - Y_{\text{predicted}})^2}{\sum(Y_{\text{actual}} - Y_{\text{mean}})^2}$$

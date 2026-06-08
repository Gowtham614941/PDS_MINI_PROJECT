import os
import csv
import json
import random
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------
# 1. PATH CONFIGURATIONS & STORAGE HELPERS
# ----------------------------------------------------
# Store data inside a 'data' folder next to this main.py file
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
EXPENSES_FILE = os.path.join(DATA_DIR, 'expenses.csv')
BUDGETS_FILE = os.path.join(DATA_DIR, 'budgets.json')
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')

DEFAULT_BUDGETS = {
    "Food": 5000.0,
    "Travel": 3000.0,
    "Entertainment": 2500.0,
    "Utilities": 4000.0,
    "Shopping": 6000.0,
    "Others": 2000.0
}

def ensure_directories():
    """Ensure data and plot directories exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

def initialize_database():
    """Initializes CSV database. Generates seed data if empty so ML model works immediately."""
    ensure_directories()
    if os.path.exists(EXPENSES_FILE) and os.path.getsize(EXPENSES_FILE) > 0:
        return

    print("[System] Generating 30 days of sample data to train machine learning models...")
    categories_and_descriptions = {
        "Food": ["starbucks coffee", "mcdonalds burger", "grocery bill supermarket", "pizza dinner", "swiggy food order", "zomato dinner"],
        "Travel": ["uber ride home", "gas station petrol refill", "metro card recharge", "ola cab ride"],
        "Entertainment": ["netflix monthly subscription", "movie ticket cinema", "spotify premium music", "gaming arcade entry"],
        "Utilities": ["electricity utility bill", "wifi broadband internet", "water utility bill", "mobile recharge"],
        "Shopping": ["amazon t-shirt", "nike sneakers", "college textbooks", "electronics charger"],
        "Others": ["medical store medicines", "haircut salon", "xerox copy printing", "gift purchase"]
    }

    today = datetime.now()
    seed_records = []
    for _ in range(60):
        days_ago = random.randint(1, 30)
        txn_date = (today - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        category = random.choice(list(categories_and_descriptions.keys()))
        description = random.choice(categories_and_descriptions[category])
        
        if category == "Food":
            amount = round(random.uniform(50.0, 600.0), 2)
        elif category == "Travel":
            amount = round(random.uniform(100.0, 1200.0), 2)
        elif category == "Entertainment":
            amount = round(random.uniform(150.0, 1500.0), 2)
        elif category == "Utilities":
            amount = round(random.uniform(300.0, 2500.0), 2)
        elif category == "Shopping":
            amount = round(random.uniform(400.0, 4000.0), 2)
        else:
            amount = round(random.uniform(20.0, 500.0), 2)
            
        seed_records.append({
            "Date": txn_date,
            "Category": category,
            "Description": description,
            "Amount": amount
        })

    seed_records.sort(key=lambda x: x["Date"])
    with open(EXPENSES_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Category", "Description", "Amount"])
        writer.writeheader()
        for record in seed_records:
            writer.writerow(record)

def load_expenses():
    """Loads all expense records from CSV file."""
    ensure_directories()
    initialize_database()
    expenses = []
    with open(EXPENSES_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            expenses.append({
                "Date": row["Date"],
                "Category": row["Category"],
                "Description": row["Description"],
                "Amount": float(row["Amount"])
            })
    return expenses

def save_expense(date_str, category, description, amount):
    """Saves a single new expense record to CSV."""
    ensure_directories()
    initialize_database()
    with open(EXPENSES_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([date_str, category, description.strip(), round(amount, 2)])

def load_budgets():
    """Loads monthly budgets from JSON file."""
    ensure_directories()
    if not os.path.exists(BUDGETS_FILE):
        with open(BUDGETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_BUDGETS, f, indent=4)
        return DEFAULT_BUDGETS
    with open(BUDGETS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_budgets(budgets):
    """Saves monthly budgets to JSON file."""
    ensure_directories()
    with open(BUDGETS_FILE, 'w', encoding='utf-8') as f:
        json.dump(budgets, f, indent=4)


# ----------------------------------------------------
# 2. MACHINE LEARNING CATEGORY CLASSIFIER
# ----------------------------------------------------
class ExpenseCategorizer:
    """Uses TF-IDF to tokenize words and Multinomial Naive Bayes to classify categories."""
    def __init__(self):
        # TF-IDF converts text descriptions into columns of word scores.
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        # Naive Bayes learns the probability of word scores matching different categories.
        self.classifier = MultinomialNB(alpha=1.0)
        self.is_trained = False
        
    def train(self, expenses_list):
        if not expenses_list or len(expenses_list) < 5:
            return False
            
        df = pd.DataFrame(expenses_list)
        X_text = df['Description'].fillna('')
        y_labels = df['Category']
        
        # We need at least 2 distinct categories to train a classifier
        if len(y_labels.unique()) < 2:
            return False
            
        try:
            # Step 1: Learn vocabulary and convert text to numerical feature vectors
            X_vectors = self.vectorizer.fit_transform(X_text)
            # Step 2: Fit Naive Bayes classifier on the numerical arrays
            self.classifier.fit(X_vectors, y_labels)
            self.is_trained = True
            return True
        except Exception:
            return False

    def predict(self, description):
        """Auto-suggests category based on input description."""
        desc_clean = description.strip().lower()
        
        # Rule-Based matches for quick heuristic accuracy
        rules = {
            "coffee": "Food", "starbucks": "Food", "burger": "Food", "pizza": "Food", "zomato": "Food", "swiggy": "Food",
            "uber": "Travel", "ola": "Travel", "cab": "Travel", "metro": "Travel", "gas": "Travel", "petrol": "Travel",
            "netflix": "Entertainment", "spotify": "Entertainment", "movie": "Entertainment", "game": "Entertainment",
            "electricity": "Utilities", "wifi": "Utilities", "broadband": "Utilities", "recharge": "Utilities",
            "amazon": "Shopping", "nike": "Shopping", "shoes": "Shopping", "shirt": "Shopping",
            "medicine": "Others", "haircut": "Others", "xerox": "Others"
        }
        
        # Check rule match first
        for key, cat in rules.items():
            if key in desc_clean:
                return cat
                
        # If no rule matched, use Naive Bayes prediction
        if self.is_trained:
            try:
                input_vector = self.vectorizer.transform([description])
                prediction = self.classifier.predict(input_vector)
                return prediction[0]
            except Exception:
                pass
        return "Others"


# ----------------------------------------------------
# 3. STATISTICAL DATA ANALYZER & FORECASTING ENGINE
# ----------------------------------------------------
class ExpenseAnalyzer:
    """Performs Z-score anomaly checking and Linear Regression daily spending forecasting."""
    def __init__(self, expenses_list):
        self.reload_data(expenses_list)
            
    def reload_data(self, expenses_list):
        """Loads updated list of expenses into a Pandas DataFrame."""
        if expenses_list:
            self.df = pd.DataFrame(expenses_list)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        else:
            self.df = pd.DataFrame(columns=['Date', 'Category', 'Description', 'Amount'])

    def detect_category_anomaly(self, amount, category, z_threshold=2.0):
        """
        Z-Score Anomaly Formula: Z = (Amount - Average) / StdDev
        Checks if a transaction amount is unusually high (> 2.0 standard deviations from mean).
        """
        if self.df.empty:
            return False, 0.0, 0.0, 0.0

        category_df = self.df[self.df['Category'] == category]
        
        # We need at least 3 transactions to calculate a meaningful average and variance
        if len(category_df) < 3:
            return False, 0.0, float(amount), 0.0
            
        amounts = category_df['Amount'].values
        mean = np.mean(amounts)
        std_dev = np.std(amounts)
        
        if std_dev == 0:
            return False, 0.0, mean, std_dev
            
        z_score = (amount - mean) / std_dev
        is_anomaly = z_score > z_threshold
        
        return is_anomaly, z_score, mean, std_dev

    def get_all_anomalies(self):
        """Audit historical records and retrieve all outlier transactions."""
        anomalies = []
        if self.df.empty:
            return anomalies
            
        for category in self.df['Category'].unique():
            cat_df = self.df[self.df['Category'] == category]
            if len(cat_df) < 3:
                continue
                
            amounts = cat_df['Amount'].values
            mean = np.mean(amounts)
            std_dev = np.std(amounts)
            
            if std_dev == 0:
                continue
                
            for _, row in cat_df.iterrows():
                z_score = (row['Amount'] - mean) / std_dev
                if z_score > 2.0:
                    anomalies.append({
                        "Date": row['Date'].strftime('%Y-%m-%d'),
                        "Category": row['Category'],
                        "Description": row['Description'],
                        "Amount": row['Amount'],
                        "Z-Score": round(z_score, 2),
                        "Category-Mean": round(mean, 2)
                    })
        
        anomalies.sort(key=lambda x: x['Z-Score'], reverse=True)
        return anomalies

    def forecast_spending(self):
        """
        Linear Regression Formula: Y = m * X + c
        Uses time (Days elapsed, X) to fit and predict cumulative spending totals (Y).
        """
        if self.df.empty:
            return None
            
        daily_totals = self.df.groupby('Date')['Amount'].sum().reset_index()
        daily_totals = daily_totals.sort_values('Date')
        daily_totals['Cumulative'] = daily_totals['Amount'].cumsum()
        
        if len(daily_totals) < 3:
            return None
            
        first_date = daily_totals['Date'].min()
        daily_totals['Days_Elapsed'] = (daily_totals['Date'] - first_date).dt.days
        
        X = daily_totals[['Days_Elapsed']].values
        y = daily_totals['Cumulative'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        m = model.coef_[0]
        c = model.intercept_
        r_squared = model.score(X, y)
        
        last_day = X[-1][0]
        current_cumulative = y[-1]
        
        pred_7_cumulative = model.predict([[last_day + 7]])[0]
        pred_30_cumulative = model.predict([[last_day + 30]])[0]
        
        forecast_7_days = max(0.0, pred_7_cumulative - current_cumulative)
        forecast_30_days = max(0.0, pred_30_cumulative - current_cumulative)
        
        avg_daily_spend = self.df.groupby('Date')['Amount'].sum().mean()
        
        return {
            "Slope": round(m, 2),
            "Intercept": round(c, 2),
            "R-Squared": round(r_squared, 4),
            "Forecast-7Days": round(forecast_7_days, 2),
            "Forecast-30Days": round(forecast_30_days, 2),
            "Avg-Daily-Spending": round(avg_daily_spend, 2),
            "Total-Spent": round(current_cumulative, 2)
        }

    def get_budget_stats(self, budgets):
        """Compares actual category spending sums against configured limits."""
        stats = {}
        if self.df.empty:
            for cat, limit in budgets.items():
                stats[cat] = {"Spent": 0.0, "Limit": limit, "Percent": 0.0, "Remaining": limit}
            return stats
            
        spent_by_cat = self.df.groupby('Category')['Amount'].sum().to_dict()
        
        for category, limit in budgets.items():
            spent = spent_by_cat.get(category, 0.0)
            percent = (spent / limit) * 100 if limit > 0 else 0.0
            stats[category] = {
                "Spent": round(spent, 2),
                "Limit": round(limit, 2),
                "Percent": round(percent, 1),
                "Remaining": round(limit - spent, 2)
            }
        return stats


# ----------------------------------------------------
# 4. DATA PLOTS & GRAPHICS GENERATOR (MATPLOTLIB)
# ----------------------------------------------------
def generate_category_pie(expenses_list):
    """Generates a Donut/Pie Chart representing category-wise spending."""
    if not expenses_list:
        print("[Visualizer] No data available to plot.")
        return
        
    df = pd.DataFrame(expenses_list)
    category_totals = df.groupby('Category')['Amount'].sum()
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B', '#B8E986', '#BD10E0']
    
    wedges, texts, autotexts = ax.pie(
        category_totals, 
        labels=category_totals.index, 
        autopct='%1.1f%%',
        startangle=140, 
        colors=colors[:len(category_totals)],
        pctdistance=0.80,
        textprops=dict(color="black", weight="bold")
    )
    
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig.gca().add_artist(centre_circle)
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        
    ax.axis('equal')  
    plt.title("Category-Wise Spending Breakdown", fontsize=14, pad=20, weight='bold', color='#333333')
    plt.tight_layout()
    
    ensure_directories()
    plot_path = os.path.join(PLOTS_DIR, 'category_breakdown.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"[Visualizer] Saved Category Breakdown chart to: '{plot_path}'")
    plt.show()

def generate_budget_vs_actual(expenses_list, budgets):
    """Generates horizontal side-by-side bar chart of Budget vs Actual."""
    if not expenses_list:
        print("[Visualizer] No data available to plot.")
        return

    df = pd.DataFrame(expenses_list)
    actual_spending = df.groupby('Category')['Amount'].sum().to_dict()
    
    categories = list(budgets.keys())
    budget_limits = [budgets[cat] for cat in categories]
    actual_spent = [actual_spending.get(cat, 0.0) for cat in categories]
    
    y = np.arange(len(categories))
    width = 0.35  
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    rects1 = ax.barh(y - width/2, budget_limits, width, label='Budget Limit', color='#D0D9E0')
    
    bar_colors = []
    for cat in categories:
        spent = actual_spending.get(cat, 0.0)
        limit = budgets[cat]
        if spent > limit:
            bar_colors.append('#E94B3C')  
        else:
            bar_colors.append('#2ECC71')  
            
    rects2 = ax.barh(y + width/2, actual_spent, width, label='Actual Spent', color=bar_colors)
    
    ax.set_xlabel('Amount ($ / ₹)', fontsize=11, weight='bold', color='#333333')
    ax.set_title('Budget Limits vs. Actual Monthly Spending', fontsize=14, weight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=10, weight='bold')
    ax.legend(loc='lower right')
    
    def autolabel(rects):
        for rect in rects:
            width_val = rect.get_width()
            if width_val > 0:
                ax.annotate(f'{width_val:.0f}',
                            xy=(width_val, rect.get_y() + rect.get_height() / 2),
                            xytext=(5, 0),  
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9, color='#555555')
                            
    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    
    ensure_directories()
    plot_path = os.path.join(PLOTS_DIR, 'budget_vs_actual.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"[Visualizer] Saved Budget comparison chart to: '{plot_path}'")
    plt.show()

def generate_spending_trend_regression(expenses_list):
    """Scatter plots cumulative spending and fits a Linear Regression trend line."""
    if not expenses_list:
        print("[Visualizer] No data available to plot.")
        return
        
    df = pd.DataFrame(expenses_list)
    df['Date'] = pd.to_datetime(df['Date'])
    
    daily_spending = df.groupby('Date')['Amount'].sum().reset_index()
    daily_spending = daily_spending.sort_values('Date')
    daily_spending['Cumulative'] = daily_spending['Amount'].cumsum()
    
    if len(daily_spending) < 3:
        print("[Visualizer] Not enough data points (need at least 3 distinct days) to plot regression trend.")
        return
        
    start_date = daily_spending['Date'].min()
    daily_spending['Days'] = (daily_spending['Date'] - start_date).dt.days
    
    X = daily_spending[['Days']].values
    y = daily_spending['Cumulative'].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    ax.scatter(daily_spending['Date'], y, color='#3498DB', s=40, zorder=5, label='Actual Cumulative Spending')
    ax.plot(daily_spending['Date'], y_pred, color='#E74C3C', linestyle='--', linewidth=2, label='Linear Regression Trend Line')
    
    future_days = np.arange(X[-1][0], X[-1][0] + 8).reshape(-1, 1)
    future_dates = [start_date + pd.Timedelta(days=int(d[0])) for d in future_days]
    future_pred = model.predict(future_days)
    
    ax.plot(future_dates, future_pred, color='#F39C12', linestyle=':', linewidth=2, label='7-Day Future Projection')
    
    plt.title('Spending History & Linear Regression Forecasting', fontsize=14, weight='bold', pad=15)
    ax.set_ylabel('Cumulative Spent ($ / ₹)', fontsize=11, weight='bold')
    ax.set_xlabel('Timeline', fontsize=11, weight='bold')
    plt.xticks(rotation=45)
    ax.legend(loc='upper left')
    
    textstr = '\n'.join((
        f'Daily Accumulation Rate (Slope): {model.coef_[0]:.2f}/day',
        f'Model Accuracy (R²): {model.score(X, y):.4f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.70, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    plt.tight_layout()
    
    ensure_directories()
    plot_path = os.path.join(PLOTS_DIR, 'spending_trend_forecast.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"[Visualizer] Saved Spending Trend chart to: '{plot_path}'")
    plt.show()


# ----------------------------------------------------
# 5. VISUAL RENDERING & COLOR CONFIGURATIONS (CLI)
# ----------------------------------------------------
PRIMARY = "\033[38;5;69m"      # Sleek Royal Blue
SUCCESS = "\033[38;5;71m"      # Emerald Green
WARNING = "\033[38;5;215m"     # Muted Gold / Amber
ALERT = "\033[38;5;203m"       # Soft Crimson / Rose
CYAN = "\033[38;5;81m"         # Soft Ice Blue
MUTED = "\033[38;5;244m"       # Dim Slate Gray
TEXT = "\033[38;5;253m"        # Clear White
BOLD = "\033[1m"
RESET = "\033[0m"

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title, subtitle=None):
    width = 54
    print(f"\n{PRIMARY}┌" + "─" * (width - 2) + "┐")
    print(f"│ {BOLD}{title.center(width - 4)}{RESET}{PRIMARY} │")
    if subtitle:
        print(f"│ {MUTED}{subtitle.center(width - 4)}{PRIMARY} │")
    print("└" + "─" * (width - 2) + f"┘{RESET}")

def print_banner(text, color=PRIMARY):
    width = 54
    print(f"{color}┌" + "─" * (width - 2) + "┐")
    print(f"│ {TEXT}{text.center(width - 4)}{color} │")
    print("└" + "─" * (width - 2) + f"┘{RESET}")

def print_success(msg):
    print(f"{SUCCESS}✔ {msg}{RESET}")

def print_warning(msg):
    print(f"{WARNING}⚠ {msg}{RESET}")

def print_error(msg):
    print(f"{ALERT}✘ {msg}{RESET}")

def print_menu_item(index, label):
    """Prints a single menu line dynamically padded to fit a 50-character wide box."""
    padded_label = f"{label:<44}"
    print(f"  {PRIMARY}│{RESET}  {CYAN}[{index}]{RESET} {TEXT}{padded_label}{PRIMARY}│{RESET}")


# ----------------------------------------------------
# 6. USER INPUT READING & PARSING
# ----------------------------------------------------
def parse_date_input(prompt_text):
    """Prompts for and validates date input (YYYY-MM-DD). Defaults to Today if blank."""
    while True:
        date_input = input(prompt_text).strip()
        if not date_input:
            return datetime.now().strftime('%Y-%m-%d')
        try:
            datetime.strptime(date_input, '%Y-%m-%d')
            return date_input
        except ValueError:
            print_error("Invalid date format. Please use YYYY-MM-DD.")

def parse_description_input(prompt_text):
    """Prompts for non-empty text descriptions."""
    while True:
        desc = input(prompt_text).strip()
        if desc:
            return desc
        print_error("Description cannot be empty.")

def parse_float_input(prompt_text, min_val=0.0):
    """Prompts for positive decimal numerical numbers."""
    while True:
        try:
            val = float(input(prompt_text))
            if val < min_val:
                print_error(f"Value must be at least {min_val:.2f}.")
            else:
                return val
        except ValueError:
            print_error("Please enter a valid decimal number.")

def parse_category_selection(budgets):
    """Displays category choices and retrieves user category selection."""
    valid_categories = list(budgets.keys())
    print(f"\n{PRIMARY}Available Categories:{RESET}")
    for idx, cat in enumerate(valid_categories, 1):
        print(f"  {idx}. {cat}")
    
    while True:
        try:
            choice_idx = int(input(f"\nSelect category index (1-{len(valid_categories)}): "))
            if 1 <= choice_idx <= len(valid_categories):
                return valid_categories[choice_idx - 1]
            else:
                print_error("Index is out of range.")
        except ValueError:
            print_error("Please enter a valid number index.")


# ----------------------------------------------------
# 7. UNICODE DATA TABLES PRINTING
# ----------------------------------------------------
def print_budget_table(stats):
    """Draws a beautiful unicode stats performance grid."""
    print(f"{PRIMARY}┌──────────────────┬──────────────┬──────────────┬──────────────┬─────────┐")
    print(f"│ Category         │ Budget Limit │ Actual Spent │ Remaining    │ Usage % │")
    print(f"├──────────────────┼──────────────┼──────────────┼──────────────┼─────────┤{RESET}")
    
    for cat, data in stats.items():
        percent = data["Percent"]
        color = RESET
        if percent >= 100:
            color = ALERT
        elif percent >= 80:
            color = WARNING
        else:
            color = SUCCESS
        
        cat_str = f"{cat:<16}"
        limit_str = f"{data['Limit']:>12.2f}"
        spent_str = f"{color}{data['Spent']:>12.2f}{RESET}{PRIMARY}"
        rem_str = f"{data['Remaining']:>12.2f}"
        percent_str = f"{color}{percent:>7.1f}%{RESET}{PRIMARY}"
        
        print(f"{PRIMARY}│{RESET} {cat_str} {PRIMARY}│{RESET} {limit_str} {PRIMARY}│{RESET} {spent_str} {PRIMARY}│{RESET} {rem_str} {PRIMARY}│{RESET} {percent_str} {PRIMARY}│{RESET}")
    
    print(f"{PRIMARY}└──────────────────┴──────────────┴──────────────┴──────────────┴─────────┘{RESET}")

def print_anomaly_table(anomalies):
    """Draws a grid listing all statistical anomaly transactions."""
    print(f"{PRIMARY}┌────────────┬──────────────────┬──────────────────────────┬──────────┬─────────┬──────────┐")
    print(f"│ Date       │ Category         │ Description              │ Amount   │ Z-Score │ Mean     │")
    print(f"├────────────┼──────────────────┼──────────────────────────┼──────────┼─────────┼──────────┤{RESET}")
    
    for entry in anomalies:
        date_val = f"{entry['Date']:<10}"
        cat_val = f"{entry['Category']:<16}"
        desc_val = f"{entry['Description']:<24}"
        amt_val = f"{ALERT}{entry['Amount']:>8.2f}{RESET}{PRIMARY}"
        z_val = f"{ALERT}{BOLD}{entry['Z-Score']:>7.2f}{RESET}{PRIMARY}"
        mean_val = f"{entry['Category-Mean']:>8.2f}"
        
        print(f"{PRIMARY}│{RESET} {date_val} {PRIMARY}│{RESET} {cat_val} {PRIMARY}│{RESET} {desc_val} {PRIMARY}│{RESET} {amt_val} {PRIMARY}│{RESET} {z_val} {PRIMARY}│{RESET} {mean_val} {PRIMARY}│{RESET}")
    
    print(f"{PRIMARY}└────────────┴──────────────────┴──────────────────────────┴──────────┴─────────┴──────────┘{RESET}")

def print_forecast_box(forecast):
    """Draws a 50-width summary box containing Linear Regression forecasts."""
    print(f"  {PRIMARY}┌──────────────────────────────────────────────────┐{RESET}")
    print(f"  {PRIMARY}│{RESET}  {BOLD}Projected Spending Forecast:{RESET}                    {PRIMARY}│{RESET}")
    print(f"  {PRIMARY}├──────────────────────────────────────────────────┤{RESET}")
    print(f"  {PRIMARY}│{RESET}  Estimated 7-Day Net Spend:   {WARNING}{forecast['Forecast-7Days']:>14.2f}{RESET}     {PRIMARY}│{RESET}")
    print(f"  {PRIMARY}│{RESET}  Estimated 30-Day Net Spend:  {WARNING}{forecast['Forecast-30Days']:>14.2f}{RESET}     {PRIMARY}│{RESET}")
    print(f"  {PRIMARY}└──────────────────────────────────────────────────┘{RESET}")


# ----------------------------------------------------
# 8. MAIN INTERACTIVE CLI DASHBOARD
# ----------------------------------------------------
def main():
    initialize_database()
    expenses = load_expenses()
    
    print(f"{MUTED}[System] Training Expense Categorizer (TF-IDF + Naive Bayes)...{RESET}")
    categorizer = ExpenseCategorizer()
    categorizer.train(expenses)
    
    analyzer = ExpenseAnalyzer(expenses)
    budgets = load_budgets()
    
    clear_console()
    
    while True:
        print_header("SMART EXPENSE ANALYZER", "Personal Wealth Intelligence Engine")
        
        print(f"  {PRIMARY}┌──────────────────────────────────────────────────┐{RESET}")
        print_menu_item(1, "Add Transaction (Auto-Categorized via ML)")
        print_menu_item(2, "View Budget Performance & Analysis Summary")
        print_menu_item(3, "Audit Anomalous Expenditures (Z-Score)")
        print_menu_item(4, "Forecast Spending Projections (Regression)")
        print_menu_item(5, "Generate Interactive Visual Analytics")
        print_menu_item(6, "Configure Monthly Budget Limits")
        print_menu_item(7, "Shutdown Application")
        print(f"  {PRIMARY}└──────────────────────────────────────────────────┘{RESET}")
        
        choice = input(f"\n{BOLD}Select Option (1-7): {RESET}").strip()
        
        if choice == '1':
            print_header("ADD TRANSACTION RECORD")
            
            date_str = parse_date_input(f"{BOLD}Date (YYYY-MM-DD) [Default: Today]: {RESET}")
            description = parse_description_input(f"{BOLD}Description (e.g. 'Starbucks coffee'): {RESET}")
            
            # Auto-suggested Category using ML model
            predicted_cat = categorizer.predict(description)
            print(f"\n{CYAN}🤖 [Machine Learning Auto-Suggest]{RESET} Predicted Category: {BOLD}{predicted_cat}{RESET}")
            accept = input("Accept suggested category? (y/n) [Default: y]: ").strip().lower()
            
            category = predicted_cat
            if accept == 'n':
                category = parse_category_selection(budgets)

            amount = parse_float_input(f"{BOLD}Amount Spent: {RESET}", min_val=0.01)
            
            # Anomaly Audit Check (Z-score calculation)
            is_anomaly, z_score, mean, std_dev = analyzer.detect_category_anomaly(amount, category)
            if is_anomaly:
                print(f"\n{ALERT}┌" + "─" * 52 + "┐")
                print(f"│             ⚠️ Z-SCORE ANOMALY DETECTED             │")
                print(f"├" + "─" * 52 + "┤")
                print(f"│ Amount {amount:.2f} is abnormally high for {category:<12} │")
                print(f"│ Historical Mean: {mean:<8.2f} | Standard Deviation: {std_dev:<7.2f} │")
                print(f"│ Calculated Z-Score: {z_score:<5.2f} (Threshold > 2.0)            │")
                print(f"└" + "─" * 52 + f"┘{RESET}\n")
                
                proceed = input("Do you still want to log this transaction? (y/n) [Default: y]: ").strip().lower()
                if proceed == 'n':
                    print_warning("Transaction canceled.")
                    input(f"\n{MUTED}Press Enter to continue...{RESET}")
                    clear_console()
                    continue
            
            save_expense(date_str, category, description, amount)
            print_success(f"Expense of {amount:.2f} added to '{category}'.")
            
            # Re-load database and re-train model with new data point
            expenses = load_expenses()
            analyzer.reload_data(expenses)
            categorizer.train(expenses)
            
            input(f"\n{MUTED}Press Enter to continue...{RESET}")
            clear_console()
            
        elif choice == '2':
            print_header("BUDGET PERFORMANCE & SUMMARY")
            if not expenses:
                print_warning("No expenses recorded yet.")
                input(f"\n{MUTED}Press Enter to continue...{RESET}")
                clear_console()
                continue
                
            total_spent = sum(item["Amount"] for item in expenses)
            avg_txn = total_spent / len(expenses)
            
            print(f"  • Total Transactions : {BOLD}{len(expenses)}{RESET}")
            print(f"  • Cumulative Spent   : {BOLD}{total_spent:.2f}{RESET}")
            print(f"  • Mean Ticket Size   : {BOLD}{avg_txn:.2f}{RESET}\n")
            
            stats = analyzer.get_budget_stats(budgets)
            print_budget_table(stats)
            
            input(f"\n{MUTED}Press Enter to return to menu...{RESET}")
            clear_console()
            
        elif choice == '3':
            print_header("AUDIT ANOMALOUS EXPENDITURES (Z-SCORE)", "Z = (X - μ) / σ > 2.0 (Top ~5% Outliers)")
            
            anomalies = analyzer.get_all_anomalies()
            if not anomalies:
                print_success("No spending anomalies detected. Expenditures fall within uniform historical parameters.")
            else:
                print(f"{WARNING}⚠️ System flagged {len(anomalies)} anomalies exceeding Z-score threshold:{RESET}\n")
                print_anomaly_table(anomalies)
            
            input(f"\n{MUTED}Press Enter to return to menu...{RESET}")
            clear_console()
            
        elif choice == '4':
            print_header("SPENDING FORECAST ENGINE", "Daily Cumulative Least-Squares Fit (y = mx + c)")
            
            forecast = analyzer.forecast_spending()
            if forecast is None:
                print_error("Insufficient data. A minimum of 3 distinct spending days is required to calculate trends.")
            else:
                r2 = forecast['R-Squared']
                if r2 >= 0.9:
                    fit_desc = f"{SUCCESS}{BOLD}Highly Predictable (Steady Trend){RESET}"
                elif r2 >= 0.7:
                    fit_desc = f"{SUCCESS}Predictable (Structured Spending){RESET}"
                elif r2 >= 0.4:
                    fit_desc = f"{WARNING}Variable (Moderate Day-to-Day Fluctuation){RESET}"
                else:
                    fit_desc = f"{ALERT}Unpredictable (Highly Volatile / Random Spending){RESET}"
                
                print(f"  {BOLD}Regression Model Metrics:{RESET}")
                print(f"    • Growth Slope (m)  : {PRIMARY}{forecast['Slope']:.2f}{RESET} units accumulated per day")
                print(f"    • Base Offset (c)   : {forecast['Intercept']:.2f}")
                print(f"    • Fit R-Squared (R²): {CYAN}{r2:.4f}{RESET} ({fit_desc})")
                print(f"    • Daily Average Spend: {forecast['Avg-Daily-Spending']:.2f}\n")
                
                print_forecast_box(forecast)
                
            input(f"\n{MUTED}Press Enter to return to menu...{RESET}")
            clear_console()
            
        elif choice == '5':
            while True:
                print_header("VISUAL ANALYTICS INTERFACE")
                print(f"  {PRIMARY}┌──────────────────────────────────────────────────┐{RESET}")
                print_menu_item(1, "Render Category Distribution (Pie Chart)")
                print_menu_item(2, "Render Budget vs Actual Spending (Bar Chart)")
                print_menu_item(3, "Render Cumulative Trend & Forecast Curve")
                print_menu_item(4, "Return to Main Dashboard")
                print(f"  {PRIMARY}└──────────────────────────────────────────────────┘{RESET}")
                
                sub_choice = input(f"\n{BOLD}Select Plot (1-4): {RESET}").strip()
                if sub_choice == '1':
                    print(f"{MUTED}[Visualizer] Generating category pie chart...{RESET}")
                    generate_category_pie(expenses)
                elif sub_choice == '2':
                    print(f"{MUTED}[Visualizer] Generating budget vs actual...{RESET}")
                    generate_budget_vs_actual(expenses, budgets)
                elif sub_choice == '3':
                    print(f"{MUTED}[Visualizer] Generating trend regression curve...{RESET}")
                    generate_spending_trend_regression(expenses)
                elif sub_choice == '4':
                    break
                else:
                    print_error("Invalid selection. Try again.")
            clear_console()
                    
        elif choice == '6':
            while True:
                print_header("CONFIGURE MONTHLY BUDGETS")
                for cat, limit in budgets.items():
                    print(f"  • {cat:<16}: {BOLD}{limit:.2f}{RESET}")
                print(f"{PRIMARY}─" * 54 + f"{RESET}")
                
                cat_name = input("Enter Category Name to edit (or leave blank to go back): ").strip()
                if not cat_name:
                    break
                    
                # Search case-insensitively
                found_cat = None
                for cat in budgets.keys():
                    if cat.lower() == cat_name.lower():
                        found_cat = cat
                        break
                        
                if found_cat is None:
                    print_error(f"Category '{cat_name}' not found. Select a valid category name.")
                    continue
                    
                new_limit = parse_float_input(f"New monthly limit for {found_cat}: ", min_val=0.0)
                budgets[found_cat] = new_limit
                save_budgets(budgets)
                print_success(f"Updated budget for {found_cat} to {new_limit:.2f}.")
            clear_console()
                    
        elif choice == '7':
            print_banner("SHUTTING DOWN SYSTEM... GOODBYE!", SUCCESS)
            break
        else:
            print_error("Invalid input. Select an index between 1 and 7.")
            input(f"\n{MUTED}Press Enter to continue...{RESET}")
            clear_console()

if __name__ == '__main__':
    main()

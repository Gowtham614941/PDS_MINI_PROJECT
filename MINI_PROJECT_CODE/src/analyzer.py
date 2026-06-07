import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

class ExpenseAnalyzer:
    def __init__(self, expenses_list):
        self.expenses = expenses_list
        # Convert to Pandas DataFrame for easy operations
        if expenses_list:
            self.df = pd.DataFrame(expenses_list)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        else:
            self.df = pd.DataFrame(columns=['Date', 'Category', 'Description', 'Amount'])
            
    def reload_data(self, expenses_list):
        """Reloads fresh data into the analyzer."""
        self.expenses = expenses_list
        if expenses_list:
            self.df = pd.DataFrame(expenses_list)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        else:
            self.df = pd.DataFrame(columns=['Date', 'Category', 'Description', 'Amount'])

    # ----------------------------------------------------
    # 1. ANOMALY DETECTION (Z-SCORE METHOD)
    # ----------------------------------------------------
    def detect_category_anomaly(self, amount, category, z_threshold=2.0):
        """
        Calculates if a specific amount is an anomaly for a given category.
        Math formula: Z = (X - mean) / std_dev
        An expense is flagged as an anomaly if Z > z_threshold (default 2.0, meaning ~top 5% outlier).
        """
        if self.df.empty:
            return False, 0.0, 0.0, 0.0

        # Filter expenses for the specific category
        cat_expenses = self.df[self.df['Category'] == category]
        
        # Need at least 3 transactions to compute a meaningful standard deviation
        if len(cat_expenses) < 3:
            return False, 0.0, float(amount), 0.0
            
        amounts = cat_expenses['Amount'].values
        mean = np.mean(amounts)
        std_dev = np.std(amounts)
        
        if std_dev == 0:
            return False, 0.0, mean, std_dev
            
        # Compute the Z-Score
        z_score = (amount - mean) / std_dev
        is_anomaly = z_score > z_threshold
        
        return is_anomaly, z_score, mean, std_dev

    def get_all_anomalies(self, z_threshold=2.0):
        """
        Scans all historical transactions and flags those that are statistical anomalies
        within their respective categories.
        """
        anomalies = []
        if self.df.empty:
            return anomalies
            
        for category in self.df['Category'].unique():
            cat_df = self.df[self.df['Category'] == category]
            if len(cat_df) < 3:
                continue  # Skip categories with too few records
                
            amounts = cat_df['Amount'].values
            mean = np.mean(amounts)
            std_dev = np.std(amounts)
            
            if std_dev == 0:
                continue
                
            for idx, row in cat_df.iterrows():
                z_score = (row['Amount'] - mean) / std_dev
                if z_score > z_threshold:
                    anomalies.append({
                        "Date": row['Date'].strftime('%Y-%m-%d'),
                        "Category": row['Category'],
                        "Description": row['Description'],
                        "Amount": row['Amount'],
                        "Z-Score": round(z_score, 2),
                        "Category-Mean": round(mean, 2)
                    })
        
        # Sort anomalies by Z-Score descending
        anomalies.sort(key=lambda x: x['Z-Score'], reverse=True)
        return anomalies

    # ----------------------------------------------------
    # 2. FORECASTING SPENDING (LINEAR REGRESSION METHOD)
    # ----------------------------------------------------
    def forecast_spending(self):
        """
        Uses Simple Linear Regression to model daily cumulative spending over time 
        and forecast future expenses.
        
        Linear Equation: y = m * x + c
        Where:
          - x = Independent variable: Number of days from start
          - y = Dependent variable: Cumulative amount spent
          - m = Slope: Daily rate of spending accumulation
          - c = Intercept: Starting expense offset
        """
        if self.df.empty:
            return None
            
        # Group spending by Date and calculate daily totals
        daily_spending = self.df.groupby('Date')['Amount'].sum().reset_index()
        daily_spending = daily_spending.sort_values('Date')
        
        # Calculate daily cumulative spending
        daily_spending['Cumulative_Amount'] = daily_spending['Amount'].cumsum()
        
        # We need at least 3 distinct days to establish a trend line
        if len(daily_spending) < 3:
            return None
            
        # Prepare training data
        # Independent variable (X): Days elapsed since the start date
        start_date = daily_spending['Date'].min()
        daily_spending['Days'] = (daily_spending['Date'] - start_date).dt.days
        
        X = daily_spending[['Days']].values
        y = daily_spending['Cumulative_Amount'].values
        
        # Initialize and fit Linear Regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Slope (m) represents average daily spending increment
        m = model.coef_[0]
        # Intercept (c)
        c = model.intercept_
        
        # Calculate R-Squared (Coefficient of Determination) to measure model fit quality
        r_squared = model.score(X, y)
        
        # Forecast coordinates
        last_day = X[-1][0]
        
        # Forecast for next 7 and 30 days
        days_7_future = last_day + 7
        days_30_future = last_day + 30
        
        # Predict cumulative spends
        current_cumulative = y[-1]
        pred_cumulative_7 = model.predict([[days_7_future]])[0]
        pred_cumulative_30 = model.predict([[days_30_future]])[0]
        
        # Prevent predictions from decreasing if model slope is negative
        predicted_7_day_spend = max(0.0, pred_cumulative_7 - current_cumulative)
        predicted_30_day_spend = max(0.0, pred_cumulative_30 - current_cumulative)
        
        # Let's also compute basic stats for context
        avg_daily_spend = self.df.groupby('Date')['Amount'].sum().mean()
        
        return {
            "Slope": round(m, 2),
            "Intercept": round(c, 2),
            "R-Squared": round(r_squared, 4),
            "Forecast-7Days": round(predicted_7_day_spend, 2),
            "Forecast-30Days": round(predicted_30_day_spend, 2),
            "Avg-Daily-Spending": round(avg_daily_spend, 2),
            "Total-Spent": round(current_cumulative, 2)
        }

    # ----------------------------------------------------
    # 3. BUDGET STATISTICS
    # ----------------------------------------------------
    def get_budget_stats(self, budgets):
        """
        Compiles actual spending totals vs budget goals for all categories.
        """
        stats = {}
        if self.df.empty:
            for cat, limit in budgets.items():
                stats[cat] = {"Spent": 0.0, "Limit": limit, "Percent": 0.0, "Remaining": limit}
            return stats
            
        # Calculate sum of amount spent grouped by category
        spent_by_cat = self.df.groupby('Category')['Amount'].sum().to_dict()
        
        # Compare actual spending against budget thresholds
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

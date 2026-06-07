import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')

def ensure_plots_dir():
    """Ensure directory for saving charts exists."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

def generate_category_pie(expenses_list):
    """
    Generates a premium looking Donut/Pie Chart representing category-wise spending.
    """
    if not expenses_list:
        print("[Visualizer] No data available to plot.")
        return
        
    df = pd.DataFrame(expenses_list)
    category_totals = df.groupby('Category')['Amount'].sum()
    
    # Styling details
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # Modern, premium palette
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B', '#B8E986', '#BD10E0']
    
    # Plotting donut chart
    wedges, texts, autotexts = ax.pie(
        category_totals, 
        labels=category_totals.index, 
        autopct='%1.1f%%',
        startangle=140, 
        colors=colors[:len(category_totals)],
        pctdistance=0.80,
        textprops=dict(color="black", weight="bold")
    )
    
    # Create the center white circle (donut hole)
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Style text inside slices
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        
    ax.axis('equal')  
    plt.title("Category-Wise Spending Breakdown", fontsize=14, pad=20, weight='bold', color='#333333')
    plt.tight_layout()
    
    ensure_plots_dir()
    plot_path = os.path.join(PLOTS_DIR, 'category_breakdown.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"[Visualizer] Saved Category Breakdown chart to: '{plot_path}'")
    plt.show()

def generate_budget_vs_actual(expenses_list, budgets):
    """
    Generates a horizontal comparison bar chart comparing category budget limits vs actual spending.
    """
    if not expenses_list:
        print("[Visualizer] No data available to plot.")
        return

    df = pd.DataFrame(expenses_list)
    actual_spending = df.groupby('Category')['Amount'].sum().to_dict()
    
    categories = list(budgets.keys())
    budget_limits = [budgets[cat] for cat in categories]
    actual_spent = [actual_spending.get(cat, 0.0) for cat in categories]
    
    y = np.arange(len(categories))
    width = 0.35  # width of bar
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    
    # Premium colors: Teal for Budget limits, coral/rose for Actual spending
    rects1 = ax.barh(y - width/2, budget_limits, width, label='Budget Limit', color='#D0D9E0')
    
    # Use different colors for actual spent depending on whether it exceeded budget
    bar_colors = []
    for cat in categories:
        spent = actual_spending.get(cat, 0.0)
        limit = budgets[cat]
        if spent > limit:
            bar_colors.append('#E94B3C')  # Exceeded (Red)
        else:
            bar_colors.append('#2ECC71')  # Within (Green)
            
    rects2 = ax.barh(y + width/2, actual_spent, width, label='Actual Spent', color=bar_colors)
    
    ax.set_xlabel('Amount ($ / ₹)', fontsize=11, weight='bold', color='#333333')
    ax.set_title('Budget Limits vs. Actual Monthly Spending', fontsize=14, weight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=10, weight='bold')
    ax.legend(loc='lower right')
    
    # Add value labels to the ends of the bars
    def autolabel(rects):
        for rect in rects:
            width_val = rect.get_width()
            if width_val > 0:
                ax.annotate(f'{width_val:.0f}',
                            xy=(width_val, rect.get_y() + rect.get_height() / 2),
                            xytext=(5, 0),  # 5 points horizontal offset
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9, color='#555555')
                            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    ensure_plots_dir()
    plot_path = os.path.join(PLOTS_DIR, 'budget_vs_actual.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"[Visualizer] Saved Budget comparison chart to: '{plot_path}'")
    plt.show()

def generate_spending_trend_regression(expenses_list):
    """
    Generates a scatter plot of cumulative daily spending and overlays the Linear Regression trend line.
    """
    if not expenses_list:
        print("[Visualizer] No data available to plot.")
        return
        
    df = pd.DataFrame(expenses_list)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Aggregate spending by date
    daily_spending = df.groupby('Date')['Amount'].sum().reset_index()
    daily_spending = daily_spending.sort_values('Date')
    daily_spending['Cumulative'] = daily_spending['Amount'].cumsum()
    
    if len(daily_spending) < 3:
        print("[Visualizer] Not enough data points (need at least 3 distinct days) to plot regression trend.")
        return
        
    # Fit linear regression model
    start_date = daily_spending['Date'].min()
    daily_spending['Days'] = (daily_spending['Date'] - start_date).dt.days
    
    X = daily_spending[['Days']].values
    y = daily_spending['Cumulative'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict trend values for plot
    y_pred = model.predict(X)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    
    # Scatter plot of actual cumulative spend
    ax.scatter(daily_spending['Date'], y, color='#3498DB', s=40, zorder=5, label='Actual Cumulative Spending')
    # Line plot of regression line
    ax.plot(daily_spending['Date'], y_pred, color='#E74C3C', linestyle='--', linewidth=2, label='Linear Regression Trend Line')
    
    # Forecast future trend line (7 days ahead)
    future_days = np.arange(X[-1][0], X[-1][0] + 8).reshape(-1, 1)
    future_dates = [start_date + pd.Timedelta(days=int(d[0])) for d in future_days]
    future_pred = model.predict(future_days)
    
    ax.plot(future_dates, future_pred, color='#F39C12', linestyle=':', linewidth=2, label='7-Day Future Projection')
    
    plt.title('Spending History & Linear Regression Forecasting', fontsize=14, weight='bold', pad=15)
    ax.set_ylabel('Cumulative Spent ($ / ₹)', fontsize=11, weight='bold')
    ax.set_xlabel('Timeline', fontsize=11, weight='bold')
    plt.xticks(rotation=45)
    ax.legend(loc='upper left')
    
    # Display statistics details in box on plot
    textstr = '\n'.join((
        f'Daily Accumulation Rate (Slope): {model.coef_[0]:.2f}/day',
        f'Model Accuracy (R²): {model.score(X, y):.4f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.70, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    plt.tight_layout()
    
    ensure_plots_dir()
    plot_path = os.path.join(PLOTS_DIR, 'spending_trend_forecast.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"[Visualizer] Saved Spending Trend chart to: '{plot_path}'")
    plt.show()

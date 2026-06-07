import os
import sys
from datetime import datetime

# Add the current directory to path so src modules are accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import initialize_database, load_expenses, save_expense, load_budgets, save_budgets
from src.classifier import ExpenseCategorizer
from src.analyzer import ExpenseAnalyzer
from src.visualizer import generate_category_pie, generate_budget_vs_actual, generate_spending_trend_regression

# Modern 256-color premium palette (soft, restrained colors)
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
    # Box inner width is 50
    # Left padding "  " (2) + "[index] " (4) = 6 chars
    # Label is padded to 44 chars
    padded_label = f"{label:<44}"
    print(f"  {PRIMARY}│{RESET}  {CYAN}[{index}]{RESET} {TEXT}{padded_label}{PRIMARY}│{RESET}")

def validate_date(date_str):
    """Validates date format YYYY-MM-DD."""
    if not date_str:
        return datetime.now().strftime('%Y-%m-%d')
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        return None

def main():
    # Ensure database folder structure and seed data exists
    initialize_database()
    
    # Load expenses
    expenses = load_expenses()
    
    # Train machine learning categorizer model
    print(f"{MUTED}[System] Training Expense Categorizer (TF-IDF + Naive Bayes)...{RESET}")
    categorizer = ExpenseCategorizer()
    categorizer.train(expenses)
    
    # Initialize data analyzer
    analyzer = ExpenseAnalyzer(expenses)
    
    # Load budgets
    budgets = load_budgets()
    
    clear_console()
    
    while True:
        print_header("SMART EXPENSE ANALYZER", "Personal Wealth intelligence Engine")
        
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
            
            # Input date
            date_input = input(f"{BOLD}Date (YYYY-MM-DD) [Default: Today]: {RESET}").strip()
            date_str = validate_date(date_input)
            while date_str is None:
                print_error("Invalid date format. Use YYYY-MM-DD.")
                date_input = input(f"{BOLD}Date (YYYY-MM-DD): {RESET}").strip()
                date_str = validate_date(date_input)
                
            # Input description
            description = input(f"{BOLD}Description (e.g. 'Starbucks coffee'): {RESET}").strip()
            while not description:
                print_error("Description cannot be empty.")
                description = input(f"{BOLD}Description: {RESET}").strip()
                
            # ML classification prediction
            predicted_cat = categorizer.predict(description)
            print(f"\n{CYAN}🤖 [Machine Learning Auto-Suggest]{RESET} Predicted Category: {BOLD}{predicted_cat}{RESET}")
            accept = input("Accept suggested category? (y/n) [Default: y]: ").strip().lower()
            
            category = predicted_cat
            if accept == 'n':
                valid_categories = list(budgets.keys())
                print(f"\n{PRIMARY}Available Categories:{RESET}")
                for idx, cat in enumerate(valid_categories, 1):
                    print(f"  {idx}. {cat}")
                
                cat_idx = None
                while cat_idx is None:
                    try:
                        input_idx = int(input(f"\nSelect category index (1-{len(valid_categories)}): "))
                        if 1 <= input_idx <= len(valid_categories):
                            category = valid_categories[input_idx - 1]
                            cat_idx = input_idx
                        else:
                            print_error("Index out of range.")
                    except ValueError:
                        print_error("Enter a valid number.")

            # Input amount
            amount = None
            while amount is None:
                try:
                    amount_input = float(input(f"{BOLD}Amount Spent: {RESET}"))
                    if amount_input <= 0:
                        print_error("Amount must be a positive number.")
                    else:
                        amount = amount_input
                except ValueError:
                    print_error("Please enter a valid numeric value.")
            
            # ML Anomaly Detection (Z-Score check)
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
            
            # Save expense
            save_expense(date_str, category, description, amount)
            print_success(f"Expense of {amount:.2f} added to '{category}'.")
            
            # Reload data in analyzer and re-train model
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
            
            # Unicode-structured Table Header
            print(f"{PRIMARY}┌──────────────────┬──────────────┬──────────────┬──────────────┬─────────┐")
            print(f"│ Category         │ Budget Limit │ Actual Spent │ Remaining    │ Usage % │")
            print(f"├──────────────────┼──────────────┼──────────────┼──────────────┼─────────┤{RESET}")
            
            stats = analyzer.get_budget_stats(budgets)
            for cat, data in stats.items():
                percent = data["Percent"]
                color = RESET
                if percent >= 100:
                    color = ALERT
                elif percent >= 80:
                    color = WARNING
                else:
                    color = SUCCESS
                
                # Format variables to fit table spacing nicely
                cat_str = f"{cat:<16}"
                limit_str = f"{data['Limit']:>12.2f}"
                spent_str = f"{color}{data['Spent']:>12.2f}{RESET}{PRIMARY}"
                rem_str = f"{data['Remaining']:>12.2f}"
                percent_str = f"{color}{percent:>7.1f}%{RESET}{PRIMARY}"
                
                print(f"{PRIMARY}│{RESET} {cat_str} {PRIMARY}│{RESET} {limit_str} {PRIMARY}│{RESET} {spent_str} {PRIMARY}│{RESET} {rem_str} {PRIMARY}│{RESET} {percent_str} {PRIMARY}│{RESET}")
            
            print(f"{PRIMARY}└──────────────────┴──────────────┴──────────────┴──────────────┴─────────┘{RESET}")
            
            input(f"\n{MUTED}Press Enter to return to menu...{RESET}")
            clear_console()
            
        elif choice == '3':
            print_header("AUDIT ANOMALOUS EXPENDITURES (Z-SCORE)", "Z = (X - μ) / σ > 2.0 (Top ~5% Outliers)")
            
            anomalies = analyzer.get_all_anomalies(z_threshold=2.0)
            if not anomalies:
                print_success("No spending anomalies detected. Expenditures fall within uniform historical parameters.")
            else:
                print(f"{WARNING}⚠️ System flagged {len(anomalies)} anomalies exceeding Z-score threshold:{RESET}\n")
                
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
                
                print(f"  {PRIMARY}┌──────────────────────────────────────────────────┐{RESET}")
                print(f"  {PRIMARY}│{RESET}  {BOLD}Projected Spending Forecast:{RESET}                    {PRIMARY}│{RESET}")
                print(f"  {PRIMARY}├──────────────────────────────────────────────────┤{RESET}")
                print(f"  {PRIMARY}│{RESET}  Estimated 7-Day Net Spend:   {WARNING}{forecast['Forecast-7Days']:>14.2f}{RESET}     {PRIMARY}│{RESET}")
                print(f"  {PRIMARY}│{RESET}  Estimated 30-Day Net Spend:  {WARNING}{forecast['Forecast-30Days']:>14.2f}{RESET}     {PRIMARY}│{RESET}")
                print(f"  {PRIMARY}└──────────────────────────────────────────────────┘{RESET}")
                
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
                    
                try:
                    new_limit = float(input(f"New monthly limit for {found_cat}: "))
                    if new_limit < 0:
                        print_error("Budget limit must be a positive number.")
                    else:
                        budgets[found_cat] = new_limit
                        save_budgets(budgets)
                        print_success(f"Updated budget for {found_cat} to {new_limit:.2f}.")
                except ValueError:
                    print_error("Please enter a valid numeric value.")
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

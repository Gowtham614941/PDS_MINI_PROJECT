import os
import csv
import json
import random
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
EXPENSES_FILE = os.path.join(DATA_DIR, 'expenses.csv')
BUDGETS_FILE = os.path.join(DATA_DIR, 'budgets.json')

DEFAULT_BUDGETS = {
    "Food": 5000.0,
    "Travel": 3000.0,
    "Entertainment": 2500.0,
    "Utilities": 4000.0,
    "Shopping": 6000.0,
    "Others": 2000.0
}

def ensure_data_dir():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def initialize_database():
    """Initializes the CSV file. If it doesn't exist, generates seed data for training ML models."""
    ensure_data_dir()
    
    # Check if database exists and has data
    if os.path.exists(EXPENSES_FILE) and os.path.getsize(EXPENSES_FILE) > 0:
        return

    # Seed data generator for a realistic 30-day spending history
    print("[System] No expense database found. Generating 30 days of realistic seed data for ML training...")
    
    categories_and_descriptions = {
        "Food": [
            "starbucks coffee", "mcdonalds burger", "grocery bill supermarket", 
            "pizza delivery dinner", "subway sandwich lunch", "swiggy food order", 
            "zomato dinner", "cafe tea and snacks", "bakery bread and cake"
        ],
        "Travel": [
            "uber ride home", "gas station petrol refill", "metro card recharge", 
            "ola cab ride to college", "train ticket booking", "bus fare ticket", 
            "parking charges ticket"
        ],
        "Entertainment": [
            "netflix monthly subscription", "movie ticket cinema", "spotify premium music", 
            "gaming arcade entry fee", "concert show ticket", "bowling alley game"
        ],
        "Utilities": [
            "electricity utility bill", "wifi broadband internet bill", 
            "water utility bill", "mobile network recharge", "piped gas bill"
        ],
        "Shopping": [
            "amazon t-shirt clothes", "nike sneakers shoes", "college textbooks purchase", 
            "electronics phone adapter charger", "backpack travel bag", "watch buying mall"
        ],
        "Others": [
            "medical store medicines", "haircut salon charges", "xerox copy printing", 
            "gift card purchase", "charity donation"
        ]
    }

    # Generate dates across the last 30 days
    today = datetime.now()
    seed_records = []
    
    # We want to create around 50 to 60 realistic transactions
    for i in range(60):
        # Pick random days in the past 30 days
        days_ago = random.randint(1, 30)
        txn_date = (today - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        category = random.choice(list(categories_and_descriptions.keys()))
        description = random.choice(categories_and_descriptions[category])
        
        # Make amounts realistic for categories (approx INR/USD appropriate ratios)
        if category == "Food":
            amount = round(random.uniform(50.0, 600.0), 2)
        elif category == "Travel":
            amount = round(random.uniform(100.0, 1200.0), 2)
        elif category == "Entertainment":
            amount = round(random.uniform(150.0, 1500.0), 2)
        elif category == "Utilities":
            # Bills are usually higher and occur once or twice
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

    # Sort records by date ascending
    seed_records.sort(key=lambda x: x["Date"])

    # Write to CSV
    with open(EXPENSES_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Category", "Description", "Amount"])
        writer.writeheader()
        for record in seed_records:
            writer.writerow(record)
            
    print(f"[System] Successfully generated {len(seed_records)} seed expenses in '{EXPENSES_FILE}'.")

def load_expenses():
    """Loads all expenses from the CSV file as a list of dicts with typed values."""
    ensure_data_dir()
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
    """Saves a single expense record into the CSV file."""
    ensure_data_dir()
    initialize_database()
    
    # Save values
    with open(EXPENSES_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([date_str, category, description.strip(), round(amount, 2)])

def load_budgets():
    """Loads monthly budget limits from budgets.json. Creates default if missing."""
    ensure_data_dir()
    if not os.path.exists(BUDGETS_FILE):
        with open(BUDGETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_BUDGETS, f, indent=4)
        return DEFAULT_BUDGETS
    
    try:
        with open(BUDGETS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return DEFAULT_BUDGETS

def save_budgets(budgets):
    """Saves monthly budgets to budgets.json."""
    ensure_data_dir()
    with open(BUDGETS_FILE, 'w', encoding='utf-8') as f:
        json.dump(budgets, f, indent=4)

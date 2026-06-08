// ====================================================
// 1. DATA DATABASE & STORAGE STATE
// ====================================================
let expenses = [];
let budgets = {};

const DEFAULT_BUDGETS = {
  "Food": 5000.0,
  "Travel": 3000.0,
  "Entertainment": 2500.0,
  "Utilities": 4000.0,
  "Shopping": 6000.0,
  "Others": 2000.0
};

// Seed descriptions for initial mock database load
const SEED_DATA_POOL = {
  "Food": ["starbucks coffee", "mcdonalds burger", "grocery bill supermarket", "pizza dinner", "swiggy food order", "zomato dinner", "cafe snacks", "bakery bread"],
  "Travel": ["uber ride home", "gas station petrol refill", "metro card recharge", "ola cab ride", "bus ticket", "train booking"],
  "Entertainment": ["netflix monthly subscription", "movie ticket cinema", "spotify premium music", "gaming arcade entry", "bowling alley game"],
  "Utilities": ["electricity utility bill", "wifi broadband internet", "water utility bill", "mobile recharge", "phone bill"],
  "Shopping": ["amazon t-shirt", "nike sneakers", "college textbooks", "electronics charger", "backpack travel bag"],
  "Others": ["medical store medicines", "haircut salon", "xerox copy printing", "gift card purchase", "laundry charge"]
};

function initializeDatabase() {
  // Try loading budgets
  const storedBudgets = localStorage.getItem('budgets');
  if (storedBudgets) {
    budgets = JSON.parse(storedBudgets);
  } else {
    budgets = { ...DEFAULT_BUDGETS };
    localStorage.setItem('budgets', JSON.stringify(budgets));
  }

  // Try loading transactions
  const storedExpenses = localStorage.getItem('expenses');
  if (storedExpenses) {
    expenses = JSON.parse(storedExpenses);
  } else {
    // Generate 30 days of realistic spending history
    console.log("[System] Seeding LocalStorage with 60 simulated transactions...");
    const today = new Date();
    const seedRecords = [];
    const categories = Object.keys(SEED_DATA_POOL);
    
    for (let i = 0; i < 60; i++) {
      const daysAgo = Math.floor(Math.random() * 30) + 1;
      const date = new Date();
      date.setDate(today.getDate() - daysAgo);
      const dateStr = date.toISOString().split('T')[0];
      
      const category = categories[Math.floor(Math.random() * categories.length)];
      const descriptionsPool = SEED_DATA_POOL[category];
      const description = descriptionsPool[Math.floor(Math.random() * descriptionsPool.length)];
      
      let amount = 0;
      if (category === "Food") amount = parseFloat((Math.random() * 550 + 50).toFixed(2));
      else if (category === "Travel") amount = parseFloat((Math.random() * 1100 + 100).toFixed(2));
      else if (category === "Entertainment") amount = parseFloat((Math.random() * 1350 + 150).toFixed(2));
      else if (category === "Utilities") amount = parseFloat((Math.random() * 2200 + 300).toFixed(2));
      else if (category === "Shopping") amount = parseFloat((Math.random() * 3600 + 400).toFixed(2));
      else amount = parseFloat((Math.random() * 480 + 20).toFixed(2));
      
      seedRecords.append ? seedRecords.append({ Date: dateStr, Category: category, Description: description, Amount: amount }) : 
                            seedRecords.push({ Date: dateStr, Category: category, Description: description, Amount: amount });
    }
    
    // Sort chronological
    seedRecords.sort((a, b) => new Date(a.Date) - new Date(b.Date));
    expenses = seedRecords;
    localStorage.setItem('expenses', JSON.stringify(expenses));
  }
}

function saveTransaction(date, category, description, amount) {
  expenses.push({ Date: date, Category: category, Description: description, Amount: parseFloat(amount) });
  // Sort
  expenses.sort((a, b) => new Date(a.Date) - new Date(b.Date));
  localStorage.setItem('expenses', JSON.stringify(expenses));
}

function deleteTransaction(index) {
  expenses.splice(index, 1);
  localStorage.setItem('expenses', JSON.stringify(expenses));
}


// ====================================================
// 2. MACHINE LEARNING: NAIVE BAYES TEXT CLASSIFIER
// ====================================================
class NaiveBayesClassifier {
  constructor() {
    this.stopWords = new Set(["the", "is", "and", "to", "for", "a", "of", "in", "on", "at", "with", "by", "from", "up", "about", "out", "into", "over", "after"]);
    this.vocabulary = new Set();
    this.categoryCounts = {};    // P(C) prior calculation
    this.wordCounts = {};        // Word occurrence per category
    this.totalExpensesCount = 0;
  }

  tokenize(text) {
    return text.toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 2 && !this.stopWords.has(word));
  }

  train(expensesList) {
    if (!expensesList || expensesList.length < 5) return;
    
    // Reset state
    this.vocabulary.clear();
    this.categoryCounts = {};
    this.wordCounts = {};
    this.totalExpensesCount = expensesList.length;

    expensesList.forEach(item => {
      const category = item.Category;
      const tokens = this.tokenize(item.Description);

      // Increment category count
      this.categoryCounts[category] = (this.categoryCounts[category] || 0) + 1;

      if (!this.wordCounts[category]) {
        this.wordCounts[category] = { _total: 0 };
      }

      tokens.forEach(token => {
        this.vocabulary.add(token);
        this.wordCounts[category][token] = (this.wordCounts[category][token] || 0) + 1;
        this.wordCounts[category]._total += 1;
      });
    });
  }

  predict(description) {
    const descClean = description.trim().lower();
    
    // Heuristic rule-based check first (quick shortcuts)
    const rules = {
      "coffee": "Food", "starbucks": "Food", "burger": "Food", "pizza": "Food", "zomato": "Food", "swiggy": "Food", "grocery": "Food", "cafe": "Food",
      "uber": "Travel", "ola": "Travel", "cab": "Travel", "metro": "Travel", "gas": "Travel", "petrol": "Travel", "bus": "Travel",
      "netflix": "Entertainment", "spotify": "Entertainment", "movie": "Entertainment", "game": "Entertainment", "arcade": "Entertainment",
      "electricity": "Utilities", "wifi": "Utilities", "broadband": "Utilities", "recharge": "Utilities", "water bill": "Utilities",
      "amazon": "Shopping", "nike": "Shopping", "shoes": "Shopping", "shirt": "Shopping",
      "medicine": "Others", "haircut": "Others", "xerox": "Others", "gift": "Others"
    };

    const keys = Object.keys(rules);
    for (let i = 0; i < keys.length; i++) {
      if (descClean.includes(keys[i])) {
        return rules[keys[i]];
      }
    }

    if (this.totalExpensesCount === 0 || Object.keys(this.categoryCounts).length < 2) {
      return "Others";
    }

    const tokens = this.tokenize(description);
    if (tokens.length === 0) return "Others";

    let bestCategory = "Others";
    let highestProbability = -Infinity;

    // Iterate through all categories and compute log probability
    Object.keys(this.categoryCounts).forEach(category => {
      // Prior log probability log P(C)
      let logProb = Math.log(this.categoryCounts[category] / this.totalExpensesCount);
      
      const categoryWords = this.wordCounts[category] || { _total: 0 };
      const totalWords = categoryWords._total;
      const vocabSize = this.vocabulary.size;

      // Add conditional log probabilities log P(W | C) using Laplace smoothing
      tokens.forEach(token => {
        const count = categoryWords[token] || 0;
        // Laplace formula: (count + 1) / (total words in C + size of vocabulary)
        const condProb = (count + 1) / (totalWords + vocabSize + 1);
        logProb += Math.log(condProb);
      });

      if (logProb > highestProbability) {
        highestProbability = logProb;
        bestCategory = category;
      }
    });

    return bestCategory;
  }
}


// ====================================================
// 3. STATISTICAL CALCULATOR (Z-SCORE & REGRESSION)
// ====================================================
function calculateCategoryStats(category) {
  const categoryExpenses = expenses.filter(item => item.Category === category);
  
  if (categoryExpenses.length < 3) {
    return { mean: 0, stdDev: 0, count: categoryExpenses.length };
  }

  const amounts = categoryExpenses.map(item => item.Amount);
  const sum = amounts.reduce((a, b) => a + b, 0);
  const mean = sum / amounts.length;

  // Calculate Variance
  const sqDiffSum = amounts.reduce((accum, val) => accum + Math.pow(val - mean, 2), 0);
  const variance = sqDiffSum / amounts.length;
  const stdDev = Math.sqrt(variance);

  return { mean, stdDev, count: categoryExpenses.length };
}

function checkAnomaly(amount, category) {
  const stats = calculateCategoryStats(category);
  if (stats.count < 3 || stats.stdDev === 0) {
    return { isAnomaly: false, zScore: 0, mean: stats.mean, stdDev: stats.stdDev };
  }

  const zScore = (amount - stats.mean) / stats.stdDev;
  return {
    isAnomaly: zScore > 2.0,
    zScore: zScore,
    mean: stats.mean,
    stdDev: stats.stdDev
  };
}

function getHistoricalAnomalies() {
  const anomalies = [];
  const categories = Object.keys(budgets);

  categories.forEach(category => {
    const stats = calculateCategoryStats(category);
    if (stats.count < 3 || stats.stdDev === 0) return;

    const categoryExpenses = expenses.filter(item => item.Category === category);
    categoryExpenses.forEach(item => {
      const z = (item.Amount - stats.mean) / stats.stdDev;
      if (z > 2.0) {
        anomalies.push({
          Date: item.Date,
          Category: item.Category,
          Description: item.Description,
          Amount: item.Amount,
          Mean: stats.mean,
          ZScore: z
        });
      }
    });
  });

  return anomalies.sort((a, b) => b.ZScore - a.ZScore);
}

function runLinearRegressionForecast() {
  if (expenses.length === 0) return null;

  // Group spends by Date
  const dailyTotals = {};
  expenses.forEach(item => {
    dailyTotals[item.Date] = (dailyTotals[item.Date] || 0) + item.Amount;
  });

  // Sort dates
  const sortedDates = Object.keys(dailyTotals).sort((a, b) => new Date(a) - new Date(b));
  if (sortedDates.length < 3) return null;

  // Compute daily cumulative totals
  let cumulative = 0;
  const dataPoints = [];
  const startDay = new Date(sortedDates[0]);

  sortedDates.forEach(dateStr => {
    cumulative += dailyTotals[dateStr];
    const diffTime = Math.abs(new Date(dateStr) - startDay);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)); // X coordinate
    dataPoints.push({ x: diffDays, y: cumulative, dateStr: dateStr });
  });

  const N = dataPoints.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, sumYY = 0;

  dataPoints.forEach(point => {
    sumX += point.x;
    sumY += point.y;
    sumXY += point.x * point.y;
    sumXX += point.x * point.x;
    sumYY += point.y * point.y;
  });

  // Solve Slope (m) and Intercept (c)
  const denominator = (N * sumXX) - (sumX * sumX);
  if (denominator === 0) return null;

  const m = ((N * sumXY) - (sumX * sumY)) / denominator;
  const c = ((sumY - (m * sumX))) / N;

  // Compute R-Squared (fit accuracy)
  const yMean = sumY / N;
  let numR2 = 0; // residual sum of squares
  let denR2 = 0; // total sum of squares

  dataPoints.forEach(point => {
    const yPred = m * point.x + c;
    numR2 += Math.pow(point.y - yPred, 2);
    denR2 += Math.pow(point.y - yMean, 2);
  });
  const rSquared = denR2 === 0 ? 0 : 1 - (numR2 / denR2);

  // Projections
  const lastPoint = dataPoints[dataPoints.length - 1];
  const lastX = lastPoint.x;
  const currentCumulative = lastPoint.y;

  const pred7Cumulative = m * (lastX + 7) + c;
  const pred30Cumulative = m * (lastX + 30) + c;

  const forecast7Days = Math.max(0, pred7Cumulative - currentCumulative);
  const forecast30Days = Math.max(0, pred30Cumulative - currentCumulative);

  // Average Daily
  const totalDaysDiff = Math.max(1, Math.ceil(Math.abs(new Date(sortedDates[sortedDates.length - 1]) - startDay) / (1000 * 60 * 60 * 24)));
  const avgDaily = cumulative / totalDaysDiff;

  return {
    slope: m,
    intercept: c,
    rSquared: rSquared,
    forecast7: forecast7Days,
    forecast30: forecast30Days,
    avgDaily: avgDaily,
    totalSpent: cumulative,
    dataPoints: dataPoints
  };
}


// ====================================================
// 4. CHART RENDERING INTERFACE (CHART.JS)
// ====================================================
let chartDonut = null;
let chartComparison = null;
let chartTrend = null;

function renderCharts() {
  const categories = Object.keys(budgets);
  
  // Aggregate category spends
  const categoryTotals = {};
  categories.forEach(cat => categoryTotals[cat] = 0);
  expenses.forEach(item => {
    if (categoryTotals[item.Category] !== undefined) {
      categoryTotals[item.Category] += item.Amount;
    } else {
      categoryTotals[item.Category] = item.Amount;
    }
  });

  // Donut Chart
  const donutCtx = document.getElementById('chart-donut').getContext('2d');
  if (chartDonut) chartDonut.destroy();
  chartDonut = new Chart(donutCtx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(categoryTotals),
      datasets: [{
        data: Object.values(categoryTotals),
        backgroundColor: ['#4f46e5', '#10b981', '#fbbf24', '#f87171', '#a78bfa', '#ec4899'],
        borderWidth: 1,
        borderColor: 'rgba(8, 11, 17, 0.8)'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#9ca3af', font: { family: 'Inter', size: 12 } }
        }
      }
    }
  });

  // Budget vs Actual Comparison Chart
  const compCtx = document.getElementById('chart-comparison').getContext('2d');
  if (chartComparison) chartComparison.destroy();

  const actualData = categories.map(cat => categoryTotals[cat] || 0);
  const limitData = categories.map(cat => budgets[cat] || 0);
  // Bar colors depending on limit overrun
  const barColors = categories.map(cat => {
    return (categoryTotals[cat] || 0) > budgets[cat] ? 'rgba(239, 68, 68, 0.85)' : 'rgba(16, 185, 129, 0.85)';
  });

  chartComparison = new Chart(compCtx, {
    type: 'bar',
    data: {
      labels: categories,
      datasets: [
        {
          label: 'Monthly Limit',
          data: limitData,
          backgroundColor: 'rgba(255, 255, 255, 0.08)',
          borderColor: 'rgba(255, 255, 255, 0.2)',
          borderWidth: 1,
          borderRadius: 6
        },
        {
          label: 'Actual Spent',
          data: actualData,
          backgroundColor: barColors,
          borderRadius: 6
        }
      ]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#9ca3af' } },
        y: { grid: { display: false }, ticks: { color: '#f3f4f6', font: { weight: 'bold' } } }
      },
      plugins: {
        legend: { labels: { color: '#9ca3af' } }
      }
    }
  });

  // Trend Regression Chart
  const trendCtx = document.getElementById('chart-trend').getContext('2d');
  if (chartTrend) chartTrend.destroy();

  const forecast = runLinearRegressionForecast();
  if (!forecast) return;

  const datesLabels = forecast.dataPoints.map(p => p.dateStr);
  const actualTrendData = forecast.dataPoints.map(p => p.y);
  
  // Calculate predicted values for trend line fit
  const regressionFitData = forecast.dataPoints.map(p => {
    return forecast.slope * p.x + forecast.intercept;
  });

  // Projections 7 Days ahead
  const lastPoint = forecast.dataPoints[forecast.dataPoints.length - 1];
  const lastX = lastPoint.x;
  const lastDate = new Date(lastPoint.dateStr);

  const futureLabels = [...datesLabels];
  const futureRegressionData = [...regressionFitData];
  
  for (let i = 1; i <= 7; i++) {
    const futDate = new Date(lastDate);
    futDate.setDate(lastDate.getDate() + i);
    futureLabels.push(futDate.toISOString().split('T')[0]);
    futureRegressionData.push(forecast.slope * (lastX + i) + forecast.intercept);
  }

  chartTrend = new Chart(trendCtx, {
    type: 'line',
    data: {
      labels: futureLabels,
      datasets: [
        {
          label: 'Cumulative Expenditure',
          data: actualTrendData,
          borderColor: '#4f46e5',
          backgroundColor: 'rgba(79, 70, 229, 0.05)',
          fill: true,
          tension: 0.1,
          pointBackgroundColor: '#6366f1',
          pointRadius: 4,
          z: 10
        },
        {
          label: 'Linear Regression Trend',
          data: futureRegressionData,
          borderColor: '#ef4444',
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
          tension: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false }, ticks: { color: '#9ca3af', maxRotation: 45, minRotation: 45 } },
        y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#9ca3af' } }
      },
      plugins: {
        legend: { labels: { color: '#9ca3af' } }
      }
    }
  });
}


// ====================================================
// 5. APPLICATION DOM ORCHESTRATION & VIEWS
// ====================================================
const classifier = new NaiveBayesClassifier();

function updateStatsDashboard() {
  const totalSpent = expenses.reduce((sum, item) => sum + item.Amount, 0);
  const totalBudget = Object.values(budgets).reduce((sum, val) => sum + val, 0);
  const anomaliesCount = getHistoricalAnomalies().length;
  
  const forecast = runLinearRegressionForecast();

  // Update DOM elements
  document.getElementById('stat-total-spent').innerText = `$${totalSpent.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  document.getElementById('stat-transaction-count').innerText = `${expenses.length} logged transactions`;
  
  document.getElementById('stat-total-budget').innerText = `$${totalBudget.toLocaleString()}`;
  const usagePercent = totalBudget > 0 ? (totalSpent / totalBudget * 100).toFixed(0) : 0;
  document.getElementById('stat-budget-usage').innerText = `${usagePercent}% total limit usage`;

  document.getElementById('stat-outliers-count').innerText = anomaliesCount;

  if (forecast) {
    document.getElementById('stat-daily-rate').innerText = `$${forecast.slope.toFixed(2)}`;
    document.getElementById('stat-forecast-r2').innerText = `Fit quality R²: ${forecast.rSquared.toFixed(4)}`;
    
    // Projections page stats
    document.getElementById('forecast-slope').innerText = `$${forecast.slope.toFixed(2)}/day`;
    document.getElementById('forecast-intercept').innerText = `$${forecast.intercept.toFixed(2)}`;
    document.getElementById('forecast-r2-val').innerText = forecast.rSquared.toFixed(4);
    
    // Predictability description
    let fitText = "";
    if (forecast.rSquared >= 0.9) fitText = "Highly Predictable (Steady)";
    else if (forecast.rSquared >= 0.7) fitText = "Predictable (Consistent)";
    else if (forecast.rSquared >= 0.4) fitText = "Variable Spending Patterns";
    else fitText = "Volatile / Highly Irregular Spends";
    document.getElementById('forecast-fit-description').innerText = fitText;

    // Projected Future Spends Progress
    document.getElementById('forecast-7day').innerText = `$${forecast.forecast7.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
    document.getElementById('forecast-30day').innerText = `$${forecast.forecast30.toLocaleString(undefined, {minimumFractionDigits: 2})}`;

    // Fill bars proportion
    const maxBar = Math.max(forecast.forecast7, forecast.forecast30, 100);
    document.getElementById('forecast-7day-bar').style.width = `${(forecast.forecast7 / maxBar * 100)}%`;
    document.getElementById('forecast-30day-bar').style.width = `${(forecast.forecast30 / maxBar * 100)}%`;
  }
}

function renderLedgerTable(filterCategory = "ALL") {
  const tbody = document.getElementById('ledger-tbody');
  tbody.innerHTML = "";

  expenses.forEach((item, index) => {
    if (filterCategory !== "ALL" && item.Category !== filterCategory) return;

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${item.Date}</td>
      <td><span class="badge ${getCategoryBadgeClass(item.Category)}">${item.Category}</span></td>
      <td style="font-weight: 500;">${item.Description}</td>
      <td style="text-align: right; font-weight: bold;">$${item.Amount.toFixed(2)}</td>
      <td style="text-align: center;">
        <button class="badge badge-danger btn-delete-txn" data-index="${index}" style="border:none; cursor:pointer; padding: 6px 10px;">Delete</button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Attach delete events
  document.querySelectorAll('.btn-delete-txn').forEach(btn => {
    btn.addEventListener('click', function() {
      const index = parseInt(this.getAttribute('data-index'));
      if (confirm("Are you sure you want to delete this transaction?")) {
        deleteTransaction(index);
        refreshUI();
      }
    });
  });
}

function renderAnomalyTable() {
  const tbody = document.getElementById('anomaly-tbody');
  tbody.innerHTML = "";

  const anomalies = getHistoricalAnomalies();
  if (anomalies.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color: var(--text-muted); padding: 24px;">No spending anomalies flagged. Spending profiles are statistically uniform.</td></tr>`;
    return;
  }

  anomalies.forEach(item => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${item.Date}</td>
      <td><span class="badge ${getCategoryBadgeClass(item.Category)}">${item.Category}</span></td>
      <td>${item.Description}</td>
      <td style="text-align: right; color: var(--alert); font-weight: bold;">$${item.Amount.toFixed(2)}</td>
      <td style="text-align: right;">$${item.Mean.toFixed(2)}</td>
      <td style="text-align: center; color: var(--alert); font-weight: bold;">${item.ZScore.toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populateSettingsForm() {
  document.getElementById('budget-food').value = budgets.Food;
  document.getElementById('budget-travel').value = budgets.Travel;
  document.getElementById('budget-entertainment').value = budgets.Entertainment;
  document.getElementById('budget-utilities').value = budgets.Utilities;
  document.getElementById('budget-shopping').value = budgets.Shopping;
  document.getElementById('budget-others').value = budgets.Others;
}

function getCategoryBadgeClass(category) {
  switch(category) {
    case "Food": return "badge-primary";
    case "Travel": return "badge-success";
    case "Entertainment": return "badge-warning";
    case "Utilities": return "badge-danger";
    case "Shopping": return "badge-warning";
    default: return "badge-success";
  }
}

function refreshUI() {
  // Retrain Classifier
  classifier.train(expenses);
  // Reload Analyzer
  // Update UI Elements
  updateStatsDashboard();
  renderLedgerTable(document.getElementById('ledger-filter-category').value);
  renderAnomalyTable();
  populateSettingsForm();
  renderCharts();
}


// ====================================================
// 6. EVENT LISTENERS & INITS
// ====================================================
document.addEventListener('DOMContentLoaded', () => {
  initializeDatabase();
  
  // Set default date to today in input modal
  document.getElementById('input-date').value = new Date().toISOString().split('T')[0];

  // Tab Navigation Handling
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', function() {
      tabs.forEach(t => t.classList.remove('active'));
      this.classList.add('active');

      const targetTab = this.getAttribute('data-tab');
      document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
      });
      document.getElementById(targetTab).classList.add('active');
      
      // Re-trigger layout draws if dashboard active
      if (targetTab === 'tab-dashboard') {
        renderCharts();
      }
    });
  });

  // Modal Visibility Handling
  const modal = document.getElementById('modal-add-transaction');
  const openBtn = document.getElementById('btn-open-modal');
  const closeBtn = document.getElementById('btn-close-modal');
  const cancelBtn = document.getElementById('btn-cancel-modal');
  const suggestBox = document.getElementById('ai-suggest-box');
  const anomalyAlert = document.getElementById('anomaly-alert-box');

  function openModal() {
    modal.classList.add('active');
    document.getElementById('input-date').value = new Date().toISOString().split('T')[0];
    document.getElementById('form-add-expense').reset();
    suggestBox.style.display = "none";
    anomalyAlert.style.display = "none";
  }

  function closeModal() {
    modal.classList.remove('active');
  }

  openBtn.addEventListener('click', openModal);
  closeBtn.addEventListener('click', closeModal);
  cancelBtn.addEventListener('click', closeModal);

  // Close modal on click outside box
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });

  // Real-Time ML category Auto-Suggestions
  const descInput = document.getElementById('input-description');
  const predictedText = document.getElementById('ai-predicted-category');
  const categorySelect = document.getElementById('input-category');
  let acceptedAI = false;

  descInput.addEventListener('input', function() {
    const text = this.value;
    if (text.length > 2) {
      const pred = classifier.predict(text);
      predictedText.innerText = pred;
      suggestBox.style.display = "flex";
      
      if (!acceptedAI) {
        categorySelect.value = pred;
      }
    } else {
      suggestBox.style.display = "none";
    }
  });

  // Accept/Reject AI predictions
  document.getElementById('btn-accept-ai').addEventListener('click', () => {
    categorySelect.value = predictedText.innerText;
    suggestBox.style.display = "none";
    acceptedAI = true;
  });

  document.getElementById('btn-reject-ai').addEventListener('click', () => {
    suggestBox.style.display = "none";
    acceptedAI = false;
  });

  // Real-Time Z-Score Anomaly Warning
  const amountInput = document.getElementById('input-amount');
  const anomalyText = document.getElementById('anomaly-alert-text');

  amountInput.addEventListener('input', function() {
    const amount = parseFloat(this.value);
    const category = categorySelect.value;
    if (!isNaN(amount) && amount > 0) {
      const audit = checkAnomaly(amount, category);
      if (audit.isAnomaly) {
        anomalyText.innerText = `This amount is extremely high for category ${category}! Average: $${audit.mean.toFixed(2)}, Z-Score: ${audit.zScore.toFixed(2)}`;
        anomalyAlert.style.display = "block";
      } else {
        anomalyAlert.style.display = "none";
      }
    } else {
      anomalyAlert.style.display = "none";
    }
  });

  categorySelect.addEventListener('change', function() {
    const amount = parseFloat(amountInput.value);
    if (!isNaN(amount) && amount > 0) {
      const audit = checkAnomaly(amount, this.value);
      if (audit.isAnomaly) {
        anomalyText.innerText = `This amount is extremely high for category ${this.value}! Average: $${audit.mean.toFixed(2)}, Z-Score: ${audit.zScore.toFixed(2)}`;
        anomalyAlert.style.display = "block";
      } else {
        anomalyAlert.style.display = "none";
      }
    }
  });

  // Submit Expense Form
  document.getElementById('form-add-expense').addEventListener('submit', function(e) {
    e.preventDefault();
    const date = document.getElementById('input-date').value;
    const category = categorySelect.value;
    const description = descInput.value;
    const amount = parseFloat(amountInput.value);

    saveTransaction(date, category, description, amount);
    closeModal();
    refreshUI();
  });

  // Filter Ledger Log
  document.getElementById('ledger-filter-category').addEventListener('change', function() {
    renderLedgerTable(this.value);
  });

  // Submit Budget Settings Form
  document.getElementById('settings-budget-form').addEventListener('submit', function(e) {
    e.preventDefault();
    budgets.Food = parseFloat(document.getElementById('budget-food').value) || DEFAULT_BUDGETS.Food;
    budgets.Travel = parseFloat(document.getElementById('budget-travel').value) || DEFAULT_BUDGETS.Travel;
    budgets.Entertainment = parseFloat(document.getElementById('budget-entertainment').value) || DEFAULT_BUDGETS.Entertainment;
    budgets.Utilities = parseFloat(document.getElementById('budget-utilities').value) || DEFAULT_BUDGETS.Utilities;
    budgets.Shopping = parseFloat(document.getElementById('budget-shopping').value) || DEFAULT_BUDGETS.Shopping;
    budgets.Others = parseFloat(document.getElementById('budget-others').value) || DEFAULT_BUDGETS.Others;

    localStorage.setItem('budgets', JSON.stringify(budgets));
    alert("Category budgets updated successfully!");
    refreshUI();
  });

  // CSV Export Logic
  document.getElementById('btn-export-csv').addEventListener('click', () => {
    if (expenses.length === 0) {
      alert("No transaction records available to export.");
      return;
    }
    let csvContent = "Date,Category,Description,Amount\n";
    expenses.forEach(item => {
      const descEscaped = item.Description.replace(/"/g, '""');
      csvContent += `"${item.Date}","${item.Category}","${descEscaped}",${item.Amount}\n`;
    });
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "expenses_backup.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  // CSV Import Logic
  const importTrigger = document.getElementById('btn-trigger-import');
  const importInput = document.getElementById('input-import-csv');

  importTrigger.addEventListener('click', () => {
    importInput.click();
  });

  importInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(evt) {
      const text = evt.target.result;
      const lines = text.split(/\r\n|\n/);
      if (lines.length < 2) {
        alert("The selected CSV file is empty or invalid.");
        return;
      }

      const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
      const hasHeader = headers.includes("Date") && headers.includes("Category") && headers.includes("Description") && headers.includes("Amount");

      const newExpenses = [];
      const startIdx = hasHeader ? 1 : 0;

      for (let i = startIdx; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        const parts = [];
        let insideQuote = false;
        let entry = "";
        
        for (let j = 0; j < line.length; j++) {
          const char = line[j];
          if (char === '"') {
            insideQuote = !insideQuote;
          } else if (char === ',' && !insideQuote) {
            parts.push(entry.trim().replace(/^["']|["']$/g, ''));
            entry = "";
          } else {
            entry += char;
          }
        }
        parts.push(entry.trim().replace(/^["']|["']$/g, ''));

        if (parts.length < 4) continue;

        const date = parts[0];
        const category = parts[1];
        const description = parts[2];
        const amount = parseFloat(parts[3]);

        if (date && category && description && !isNaN(amount)) {
          newExpenses.push({ Date: date, Category: category, Description: description, Amount: amount });
        }
      }

      if (newExpenses.length > 0) {
        if (confirm(`Successfully parsed ${newExpenses.length} records. Do you want to overwrite your current dashboard data? (Click 'Cancel' to append instead)`)) {
          expenses = newExpenses;
        } else {
          expenses = [...expenses, ...newExpenses];
        }
        
        expenses.sort((a, b) => new Date(a.Date) - new Date(b.Date));
        localStorage.setItem('expenses', JSON.stringify(expenses));
        refreshUI();
        alert(`Loaded transactions successfully!`);
      } else {
        alert("Could not parse any valid transaction records from the CSV file.");
      }
      importInput.value = "";
    };
    reader.readAsText(file);
  });

  // Initialize UI on first run
  refreshUI();
});

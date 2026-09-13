# AegisPay

## AI-Powered Financial Safety & Payment Planning Agent

AegisPay is an AI-powered financial decision-support agent developed for the **HackerRank Orchestrate — Buy or Wait?** hackathon.

Instead of checking only a user's current balance, AegisPay reconstructs the user's financial state, forecasts future cash flow, evaluates available payment options, and determines the safest way to handle a requested expense.

For a request such as:

> "Can I afford this purchase?"

AegisPay evaluates whether the user should pay now, use a payment plan, pay partially, wait until a safer date, or avoid the purchase.

The core financial calculations are deterministic to ensure that recommendations are reproducible, verifiable, and safe.

---

## Features

- Financial state reconstruction from structured financial data
- Historical and future transaction analysis
- Recurring income and expense detection
- Pending and scheduled obligation handling
- Minimum-balance protection
- 90-day cash-flow forecasting
- Payment-plan feasibility checking
- Full-payment planning
- Partial-payment planning
- Installment-plan evaluation
- Wait-until-affordable recommendations
- Fixed-date currency conversion using provided exchange rates
- Deterministic candidate generation and ranking
- Deterministic financial safety validation
- Structured Gemini AI agent integration
- Exact `output.csv` generation
- Processing of the complete request dataset

---

## System Architecture

```text
                         USER REQUEST
                              |
                              v
                    +-------------------+
                    |    Data Loader    |
                    +-------------------+
                              |
                              v
                 +-------------------------+
                 | Financial State         |
                 | Reconstruction          |
                 +-------------------------+
                              |
                              v
                 +-------------------------+
                 | 90-Day Financial        |
                 | Forecast & Safety       |
                 | Analysis                |
                 +-------------------------+
                              |
                              v
                 +-------------------------+
                 | Payment Plan Generator  |
                 |                         |
                 | - Full Payment          |
                 | - Partial Payment       |
                 | - Installments         |
                 | - Wait                  |
                 +-------------------------+
                              |
                              v
                 +-------------------------+
                 | Deterministic           |
                 | Validation               |
                 +-------------------------+
                              |
                              v
                         output.csv
```

AegisPay separates **AI-assisted reasoning** from critical financial computation.

Money calculations, dates, forecasting, payment constraints, and safety validation are handled deterministically in Python.

---

## Project Structure

```text
AegisPay/
│
├── code/
│   ├── main.py
│   ├── data_loader.py
│   ├── financial_state.py
│   ├── forecasting.py
│   ├── planner.py
│   └── llm_agent.py
│
├── evaluation/
│   └── usage_report.md
│
├── dataset/
│   ├── requests.csv
│   ├── financial_profiles.csv
│   ├── financial_events.csv
│   ├── request_payment_options.csv
│   ├── exchange_rates.csv
│   ├── messages.csv
│   ├── images.csv
│   └── media/
│       └── images/
│
├── output.csv
├── README.md
├── requirements.txt
├── AGENTS.md
└── log.txt
```

The challenge dataset is used locally during execution and is not required inside the solution code package.

---

# Core Components

## 1. Data Loader

**File:** `code/data_loader.py`

The data loader reads and normalizes the provided CSV datasets into structured Python objects.

It handles:

- Financial profiles
- Financial events
- User requests
- Payment options
- Messages
- Image references
- Exchange rates

The loader preserves important attributes such as:

- Dates
- Amounts
- Currencies
- Transaction status
- Categories
- Payment preferences
- Minimum balance requirements
- Financial priorities

---

## 2. Financial State Reconstruction

**File:** `code/financial_state.py`

For every request, AegisPay builds a request-specific financial state.

The system distinguishes between:

- Historical settled transactions
- Future transactions
- Recurring income
- Recurring expenses
- Pending obligations
- Scheduled obligations
- Cancelled transactions
- Failed transactions
- Unrealized investment values
- Relevant contextual messages
- Relevant image references

Recurring transactions are detected using historical transaction patterns rather than assuming that every repeated transaction is recurring.

Currency conversion uses the dated exchange rates supplied with the dataset.

---

## 3. 90-Day Financial Forecast

**File:** `code/forecasting.py`

AegisPay forecasts the user's financial position over the next 90 days.

For each day, the system maintains:

```text
Opening Balance
       +
Credits
       -
Debits
       =
Closing Balance
```

The forecast incorporates relevant future financial events and recurring transactions while avoiding double-counting where applicable.

A payment plan is considered safe only if the user's projected balance remains above the required minimum balance.

---

## 4. Payment Planner

**File:** `code/planner.py`

The planner generates and evaluates possible payment strategies.

Supported strategies include:

```text
Full Payment
Partial Payment
Installments
Wait
Not Recommended
```

Each candidate plan is checked against the financial forecast before it can be selected.

### Full Payment

The system checks whether the complete requested amount can safely be paid on the request date.

### Partial Payment

When partial payment is permitted, AegisPay can split the purchase into:

```text
Safe payment today
+
Remaining amount on the earliest safe date
```

### Installments

The system evaluates installment options supplied for the specific request.

The original payment schedule and payment amounts are preserved rather than being invented by the model.

### Wait

If the full amount is not safely payable today but becomes affordable later, AegisPay identifies the earliest safe payment date.

### Not Recommended

If no valid payment strategy satisfies the financial constraints, the system recommends against the purchase.

---

## 5. Financial Safety Validation

AegisPay does not consider a purchase safe simply because the user has enough money today.

For every payment candidate, planned payments are applied cumulatively against the projected daily balances.

Conceptually:

```text
Projected Balance
        -
Cumulative Planned Payments
        =
Adjusted Balance
```

If the adjusted balance falls below the user's required minimum balance on any forecast day, the candidate is rejected.

This creates a deterministic safety boundary around the final recommendation.

---

# Decision Model

For every request, AegisPay generates the required output fields:

| Field | Description |
|---|---|
| `request_id` | Unique request identifier |
| `amount_safe_to_pay` | Maximum amount safely payable today |
| `affordability_status` | Overall affordability classification |
| `recommended_payment_method` | Recommended payment strategy |
| `payment_plan` | Chronological payment dates and amounts |
| `earliest_date_for_full_payment` | Earliest date the complete amount is safely payable |
| `spending_changes_needed` | Required spending adjustments |
| `decision_explanation` | Concise explanation for the decision |

---

## Affordability Statuses

```text
affordable_now
affordable_with_plan
affordable_later
not_affordable
```

---

## Recommended Payment Methods

```text
full_payment
partial_payment
installments
wait
not_recommended
```

---

# Safe Payment Amount

`amount_safe_to_pay` is not calculated simply as:

```text
Current Balance - Minimum Balance
```

Instead, AegisPay considers the user's future financial obligations.

The calculation is based on:

```text
Current Financial Position
          +
Expected Safe Credits
          -
Future Obligations
          -
Required Minimum Balance
          =
Safe Payment Capacity
```

The final value is constrained to:

```text
0 <= amount_safe_to_pay <= requested_amount
```

---

# Financial Event Handling

AegisPay applies conservative rules when processing uncertain financial information.

### Pending Debits

Pending and scheduled debits are treated conservatively when determining available spending capacity.

### Pending Credits

Uncertain future credits are not treated as immediately available spending money.

### Failed and Cancelled Transactions

Failed and cancelled transactions are excluded from the effective financial state.

### Unrealized Investments

Unrealized investment valuations are not treated as available cash.

### Recurring Transactions

Recurring income and expenses are detected from repeated historical patterns using transaction characteristics such as:

- Category
- Description
- Amount
- Timing
- Historical frequency

---

# Candidate Ranking

After invalid candidates are removed, AegisPay ranks the remaining valid plans deterministically.

The ranking prioritizes:

1. Completing the request by the required deadline
2. Avoiding spending changes
3. Minimizing total amount paid
4. Starting earlier
5. Using fewer payments
6. Using the lowest payment option ID as a deterministic tie-breaker

This ensures consistent results when the same input data is processed repeatedly.

---

# AI Integration

**File:** `code/llm_agent.py`

AegisPay includes a structured Gemini-based AI agent using the Google GenAI SDK.

The AI component uses structured output through Pydantic models.

Instead of allowing the LLM to freely invent financial calculations or payment schedules, the model operates over structured financial facts and candidate plans generated by the deterministic system.

The architecture is:

```text
Financial Data
      |
      v
Deterministic Financial Analysis
      |
      v
Valid Candidate Plans
      |
      v
Gemini Structured Reasoning
      |
      v
Candidate Selection / Explanation
      |
      v
Deterministic Validation
      |
      v
Final Output
```

This approach reduces the risk of hallucinated monetary values and keeps financial safety checks outside the language model.

The final full-dataset prediction run used the deterministic planning pipeline for reproducibility.

---

# Why a Hybrid AI Architecture?

Financial decisions require both contextual reasoning and numerical reliability.

An unrestricted LLM is not ideal for:

- Monetary arithmetic
- Date calculations
- Cash-flow forecasting
- Constraint checking
- Payment schedule validation

Therefore, AegisPay follows a hybrid approach.

### Deterministic Layer

Responsible for:

- Money calculations
- Date calculations
- Financial state reconstruction
- Forecasting
- Payment constraints
- Minimum-balance protection
- Candidate validation
- Candidate ranking
- Output validation

### AI Layer

Responsible for:

- Structured reasoning
- Context interpretation
- Candidate selection
- Natural-language decision explanations

This gives the system both **AI reasoning capability** and **deterministic financial safety**.

---

# Running the Project

## Requirements

- Python 3.10+
- Google GenAI SDK
- Pydantic
- python-dotenv

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

For the Gemini component, configure:

```text
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.8-flash
```

Do not commit API keys or other secrets to the repository.

---

## Run the Agent

From the project root:

```bash
python code/main.py
```

The program reads the challenge dataset and generates:

```text
output.csv
```

in the project root.

---

# Output

The generated file follows the required schema and column order:

```text
request_id
amount_safe_to_pay
affordability_status
recommended_payment_method
payment_plan
earliest_date_for_full_payment
spending_changes_needed
decision_explanation
```

The final full-dataset execution processes:

```text
250 requests
```

and produces one output row per request.

---

# Validation

Before producing the final result, AegisPay performs deterministic validation of the planning pipeline.

Validation includes:

- Processing all requests
- Required output schema
- Correct column order
- One output row per request
- Safe payment amount bounds
- Payment-plan feasibility
- Forecast safety
- Valid payment schedules
- Deterministic planner execution

The model and token usage information is documented in:

```text
evaluation/usage_report.md
```

---

# Technology Stack

- Python
- Google Gemini
- Google GenAI SDK
- Pydantic
- CSV Data Processing
- Financial State Reconstruction
- 90-Day Forecasting
- Deterministic Rule-Based Planning

---

# Design Principles

## 1. Safety First

A payment should not be considered affordable merely because it is possible today.

Future obligations and minimum balance requirements must also be respected.

## 2. Deterministic Financial Logic

Critical financial calculations are performed programmatically rather than delegated entirely to an LLM.

## 3. Evidence-Based Decisions

The system uses the supplied financial data and contextual evidence instead of hardcoded request-specific answers.

## 4. Reproducibility

The same financial inputs should produce consistent planning results.

## 5. AI With Guardrails

The LLM operates within a constrained decision framework instead of having unrestricted control over financial calculations.

---

# Future Improvements

Potential extensions include:

- Stronger OCR/VLM extraction from financial images
- More advanced message interpretation
- Automated optimization of flexible spending changes
- Improved recurring-transaction detection
- Interactive financial forecast visualization
- Explainable decision traces
- Automated field-level evaluation dashboards
- Production database integration
- Real-time transaction ingestion
- Secure API deployment
- Multi-model reasoning and verification

---

# Hackathon

Built for the:

**HackerRank Orchestrate — September 2026**

### Challenge

**Buy or Wait?**

### Objective

Build an AI-powered financial agent capable of analyzing a user's financial context and determining the safest way to handle a requested purchase.

---

# Conclusion

AegisPay combines financial data processing, forecasting, deterministic constraint validation, payment planning, and structured AI reasoning into a single financial safety agent.

The central design principle is:

> **AI should assist financial reasoning, while deterministic safety constraints control financial decisions.**

This allows AegisPay to move beyond a simple "Can I afford this?" balance check toward a more complete **financially safe payment-planning decision**.
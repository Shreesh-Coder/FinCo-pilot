# FinCo-pilot: Financial Data Analysis Chatbot

FinCo-pilot is a Python-based conversational agent designed to help you analyze your State Bank of India (SBI) financial statements. It processes raw `.xls` statement files, categorizes transactions, and provides a chatbot interface powered by LangChain and OpenAI to query, summarize, visualize, and manage your financial data.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

---

## Features

* **SBI Statement Parsing:** Reads and extracts transaction data from `.xls` files downloaded from SBI Net Banking.
* **Data Processing & Enrichment:** Cleans, formats, and adds useful columns like `Amount`, `Year`, `Month`, and `DayOfWeek`.
* **Categorization:** Automatically categorizes transactions based on keywords defined in a JSON mapping file. Supports dynamic updating via chatbot commands.
* **Duplicate Removal:** Identifies and removes duplicate transactions at startup.
* **Conversational Interface:** Interact using natural language to ask questions, perform analysis, and generate visualizations.
* **Data Visualization:** Create plots (e.g., spending trends) directly from the chatbot.
* **Data Saving:** Save processed data or analysis output to files.
* **Account Details Display:** Retrieve parsed account holder information on demand.

---

## Prerequisites

* **Python 3.8+**
* **SBI Bank Statements:** Download your transaction history as `.xls` files.
* **OpenAI API Key:** Obtain from [OpenAI](https://platform.openai.com/).

---

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Place SBI statements**

   * Put all `.xls` files into: `financial_analyst/statements/`

4. **Configure OpenAI API Key**

   * Create a file named `.env` in the project root:

     ```dotenv
     OPENAI_API_KEY='your_api_key_here'
     ```

5. **Provide parsing script**

   * Ensure `financial_analyst/parse_sbi_statement.py` exists and correctly parses your statement format.

---

## Usage

1. **Activate virtual environment** (if not already active).

2. **Run the chatbot**

   ```bash
   python financial_analyst/accountant_chatbot.py
   ```

3. **Interact** using natural language:

   * *Summarize spending by category for 2024.*
   * *What was my total income in March 2023?*
   * *Plot my monthly spending trend.*
   * *Update the category mapping for 'Groceries' with keywords \['reliance fresh','more store'].*
   * *Save the processed data to report.csv.*
   * *Show me my account details.*

4. **Commands**

   | Command    | Description                         |
   | ---------- | ----------------------------------- |
   | `/help`    | Display help menu                   |
   | `/clear`   | Clear conversation history          |
   | `/restart` | Reload data and restart the chatbot |
   | `/quit`    | Exit the chatbot                    |

---

## File Structure

```
.
├── .env                          # Environment variables
├── plots/                        # Saved plot files
├── financial_analyst/
│   ├── accountant_chatbot.py     # Main chatbot script
│   ├── category_mapping.json     # Transaction categorization map
│   ├── parse_sbi_statement.py    # SBI parser (provide your own)
│   ├── processed_data/
│   │   ├── financials.csv        # Processed data cache
│   │   └── account_details.json  # Parsed account info
│   └── statements/               # Raw `.xls` statements
└── README.md                     # Project documentation
```

---

## Customization

* **Category Mapping**: Edit `financial_analyst/category_mapping.json` or use the `/restart` command after updating.
* **Parsing Logic**: Modify `parse_sbi_statement.py` to match your statement format.

---

## Troubleshooting

* **Missing API Key**: Ensure `.env` contains a valid `OPENAI_API_KEY`.
* **Parsing Errors**: Adjust `parse_sbi_statement.py` to handle your `.xls` structure.
* **No Plots Saved**: Verify plotting code uses `plt.savefig()` with a path under `./plots/`.
* **Tool Errors**: Check error messages in the chatbot and correct column names or code logic.

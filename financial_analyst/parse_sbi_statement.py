import pandas as pd
import re

def parse_sbi_statement(file_path: str):
    """
    Parses a tab-separated SBI bank statement text file. (Version 5 - Final)

    This version is designed for the format where account details are in a
    key-value section, followed by a tab-separated transaction table. It includes
    robust cleaning for column names and data types.

    Args:
        file_path: The path to the bank statement .txt or .xls (as text) file.

    Returns:
        A tuple containing:
        - account_details (dict): A dictionary of the account holder's information.
        - transactions_df (pd.DataFrame): A DataFrame of the parsed transactions.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

    # --- 1. Find the start of the transaction table header ---
    header_line_index = -1
    for i, line in enumerate(lines):
        if "Txn Date\tValue Date\tDescription" in line:
            header_line_index = i
            break

    if header_line_index == -1:
        print(f"Error: Could not find the transaction table header in '{file_path}'.")
        return None, pd.DataFrame()

    # --- 2. Parse Account Details ---
    account_details = {}
    header_lines = lines[:header_line_index]
    for line in header_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key == 'Address' and 'Address' in account_details:
                account_details['Address'] += ", " + value
            else:
                account_details[key] = value
        elif 'Address' in account_details and line.strip():
             account_details['Address'] += ", " + line.strip()

    # --- 3. Parse Transactions using pandas.read_csv ---
    try:
        transactions_df = pd.read_csv(
            file_path,
            sep='\t',
            skiprows=header_line_index,
            header=0,
            engine='python',
            skipfooter=1,
            on_bad_lines='warn'
        )
    except Exception as e:
        print(f"Pandas failed to parse the transaction table in '{file_path}'. Error: {e}")
        return account_details, pd.DataFrame()

    # --- 4. Clean the resulting DataFrame ---

    # Drop any extra "Unnamed" columns that might appear
    unnamed_cols = [col for col in transactions_df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        transactions_df = transactions_df.drop(columns=unnamed_cols)

    # Standardize column names to remove leading/trailing spaces
    transactions_df.columns = [str(col).strip() for col in transactions_df.columns]

    # Robust cleaning for numeric columns
    for col in ['Debit', 'Credit', 'Balance']:
        if col in transactions_df.columns:
            transactions_df[col] = pd.to_numeric(transactions_df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # Clean and convert date columns
    for col in ['Txn Date', 'Value Date']:
        if col in transactions_df.columns:
            transactions_df[col] = pd.to_datetime(transactions_df[col], format='%d %b %Y', errors='coerce')

    # Drop any rows that were not parsed correctly, identified by a null Txn Date
    transactions_df.dropna(subset=['Txn Date'], inplace=True)

    return account_details, transactions_df

# --- HOW TO USE THIS SCRIPT ---
if __name__ == '__main__':
    file_path = "financial_analyst/statements/sbi_shreesh_13_jun_28_jun_25.xls"

    details, df = parse_sbi_statement(file_path)

    if df is not None:
        print("--- Account Details ---")
        if details:
            for key, value in details.items():
                print(f"{key:<20}: {value}")
        
        print("\n--- Parsed Transactions ---")
        if not df.empty:
            print(df.to_string())
        else:
            print("The transaction DataFrame is empty.")

        print("\n--- Data Types of Columns ---")
        df.info()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import glob
import re
import sys
import subprocess
import json
from typing import List, Dict, Any

# Load .env file at the very top
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool, Tool
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field

# Local parser import
# Create a dummy function if the real one isn't available, for testing purposes
try:
    from parse_sbi_statement import parse_sbi_statement
except ImportError:
    print("⚠️  'parse_sbi_statement' not found. Using a dummy parser function.")
    def parse_sbi_statement(file_path):
        # This is a placeholder. Replace with your actual parser.
        data = {
            'Txn Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'Value Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'Description': ['UPI/FOOD/ZOMATO', 'UPI/SHOP/AMAZON', 'SALARY CREDIT'],
            'Ref No./Cheque No.': ['123', '456', '789'],
            'Debit': [15.50, 120.00, 0.0],
            'Credit': [0.0, 0.0, 2500.00],
            'Balance': [1000.0, 880.0, 3380.0]
        }
        # Return a dictionary for account details in the dummy function
        dummy_details = {
            'Account Holder Name': 'Dummy User',
            'Account Number': 'XXXXXX1234',
            'Bank Name': 'Dummy Bank'
        }
        return dummy_details, pd.DataFrame(data)

# --- 1. System State (Holds the in-memory DataFrame) ---
class SystemState:
    def __init__(self):
        self.dataframe_path = os.path.abspath("financial_analyst/processed_data/financials.csv")
        self.df = None # Will hold the DataFrame in memory
        self.last_query_result = ""
        self.account_details = {} # Add a field to store account details

state = SystemState()

# --- 2. Data Processing and Tools ---

def process_data(statements_dir: str = os.path.abspath("financial_analyst/statements/")) -> bool:
    """Processes raw .xls statement files and loads the result into the state."""
    print("🚀 Processing raw statement files...")

    processed_file_path = state.dataframe_path
    statements_dir_abs = os.path.abspath(statements_dir)
    xls_files = glob.glob(os.path.join(statements_dir_abs, '*.xls'))
    account_details_path = os.path.abspath("financial_analyst/processed_data/account_details.json")

    needs_reprocessing = True # Assume reprocessing is needed by default

    # Check if processed file exists and is up to date
    if os.path.exists(processed_file_path):
        processed_mtime = os.path.getmtime(processed_file_path)
        is_up_to_date = True
        for f in xls_files:
            if os.path.getmtime(f) > processed_mtime:
                is_up_to_date = False
                break

        if is_up_to_date:
            # Attempt to load from cache
            print(f"✅ Processed data file '{os.path.basename(processed_file_path)}' is up to date. Attempting to load from cache.")
            try:
                state.df = pd.read_csv(processed_file_path)
                # Ensure date columns are in datetime format after loading from CSV
                state.df['Txn Date'] = pd.to_datetime(state.df['Txn Date'])
                state.df['Value Date'] = pd.to_datetime(state.df['Value Date'])
                state.data_loaded = True

                # Attempt to load account details from JSON
                state.account_details = {} # Initialize before loading
                if os.path.exists(account_details_path):
                    try:
                        with open(account_details_path, 'r') as f:
                            loaded_details = json.load(f)
                            if isinstance(loaded_details, dict) and loaded_details: # Ensure loaded data is a non-empty dictionary
                                state.account_details = loaded_details
                                print("✅ Loaded account details from JSON cache.")
                            else:
                                print(f"❌ Account details JSON cache is empty or invalid at {account_details_path}.")
                    except Exception as e:
                        print(f"❌ Error loading account details from JSON cache: {e}.")

                # If both CSV and account details loaded successfully, no reprocessing needed
                if state.df is not None and not state.df.empty and state.account_details:
                     needs_reprocessing = False
                     print("✅ Data and account details loaded successfully from cache.")
                     return True # Successfully loaded from cache
                else:
                    print("⚠️ Cache loading incomplete or account details missing. Proceeding to full reprocessing.")

            except Exception as e:
                print(f"❌ Error loading from cache: {e}. Proceeding to full reprocessing.")

    # If processed file doesn't exist or is outdated, or cache loading failed, perform full reprocessing
    if needs_reprocessing:
        print("🔄 Processed data is missing, outdated, or cache load failed. Performing full reprocessing.")

        if not xls_files:
            print(f"⚠️ No .xls files found in '{statements_dir_abs}'. Using dummy data for demonstration.")
            account_details, master_df = parse_sbi_statement(None) # Use dummy data if no files
            if isinstance(account_details, dict): # Ensure dummy details are a dictionary
                 state.account_details = account_details
        else:
            all_dfs = []
            all_account_details = {} # Initialize a dictionary to collect details from all files
            for i, f in enumerate(xls_files):
                try:
                    details, df = parse_sbi_statement(f)
                    # print(f"DEBUG: Parsed file {os.path.basename(f)}. Details type: {type(details)}, DataFrame empty: {df.empty if df is not None else 'N/A'}") # Added debug print
                    if df is not None and not df.empty:
                        all_dfs.append(df)
                        if isinstance(details, dict): # Check if details is a dictionary
                            all_account_details.update(details) # Merge details from this file
                except Exception as e:
                    print(f"❌ Error parsing {f}: {e}")

            if not all_dfs:
                print("❌ No data could be parsed from the statement files.")
                return False
            master_df = pd.concat(all_dfs, ignore_index=True)
            if isinstance(all_account_details, dict): # Store the merged unique details if it's a dictionary
                state.account_details = all_account_details
                print(f"✅ Account details updated from parsed files: {state.account_details}") # Confirmation print

        # Data Enrichment
        master_df['Debit'] = pd.to_numeric(master_df['Debit'], errors='coerce').fillna(0)
        master_df['Credit'] = pd.to_numeric(master_df['Credit'], errors='coerce').fillna(0)
        master_df['Amount'] = master_df['Credit'] - master_df['Debit']
        master_df['Value Date'] = pd.to_datetime(master_df['Value Date'])
        master_df['Year'] = master_df['Value Date'].dt.year
        master_df['Month'] = master_df['Value Date'].dt.month_name()
        master_df['DayOfWeek'] = master_df['Value Date'].dt.day_name()

        # Categorization
        mapping_path = os.path.abspath("financial_analyst/category_mapping.json")
        mapping = {}
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"⚠️ Category mapping file not found or invalid JSON at {mapping_path}. Using default mapping.")
            mapping = {
                'Transfers': ['TRANSFER', 'UPI', 'NEFT', 'RTGS', 'IMPS'],
                'Food & Dining': ['SWIGGY', 'ZOMATO', 'DOMINO', 'ISTHARA', 'RESTAURANT', 'FOOD'],
                'Utilities': ['ELECT', 'BROADBAND', 'BILL', 'JIO', 'TEL', 'PAYTM', 'RECHARGE', 'GAS'],
                'Transportation': ['IRCTC', 'OLA', 'UBER', 'FUEL', 'AUTO', 'METRO'],
                'Shopping': ['AMAZON', 'FLIPKART', 'MALL', 'MYNTRA', 'SHOP', 'AJIO'],
                'Groceries': ['DMART', 'GROCERY', 'BIGBASKET', 'SUPERMARKET'],
                'Cash Withdrawal': ['ATM', 'CASH WITHDRAWAL'],
                'Bank Charges': ['CHARGES', 'FEE', 'MAB', 'MIN BAL'],
                'Investments & Returns': ['INTEREST', 'DIVIDEND', 'MUTUAL FUND', 'ZERODHA'],
                'Rent': ['RENT'],
                'Income': ['SALARY']
            }

        def categorize(desc):
            d = str(desc).upper()
            return next((cat for cat, kws in mapping.items() if isinstance(cat, str) and any(kw in d for kw in kws)), 'Others')
        master_df['Category'] = master_df['Description'].apply(categorize)
        master_df = master_df.sort_values(by='Value Date')

        os.makedirs(os.path.dirname(state.dataframe_path), exist_ok=True)
        master_df.to_csv(state.dataframe_path, index=False)
        # Save account details to a separate JSON file only if state.account_details is not empty
        if state.account_details:
            account_details_path = os.path.abspath("financial_analyst/processed_data/account_details.json")
            try:
                with open(account_details_path, 'w') as f:
                    json.dump(state.account_details, f, indent=4)
                print("✅ Account details saved to JSON.")
            except Exception as e:
                print(f"❌ Error saving account details to JSON: {e}")

        state.df = master_df # Load into memory
        print(f"✅ Data processed and saved. DataFrame is now loaded into memory.")
        state.data_loaded = True
        return True # Full reprocessing completed

    return False # Should not be reached if logic is correct, but as a fallback

def inspect_data(input: str = "") -> str:
    """Provides a concise summary of the master financial DataFrame."""
    if state.df is None: return "Error: The DataFrame is not loaded. Please restart the application."
    df = state.df
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    columns = df.columns.tolist()
    date_range = f"From {df['Value Date'].min().strftime('%Y-%m-%d')} to {df['Value Date'].max().strftime('%Y-%m-%d')}"
    categories = df['Category'].unique().tolist()
    return (f"**Data Inspector Report**\n\n**Columns:**\n{columns}\n\n**Data Types & Non-Nulls:**\n{info_str}\n"
            f"**Date Range:**\n{date_range}\n\n**Available Categories:**\n{categories}")

class PythonCodeExecutorInput(BaseModel):
    code: str = Field(description="The Python code to execute for data analysis. It must operate on a DataFrame named 'df' and use `print()` to output the result.")

class PythonCodeExecutorTool(BaseTool):
    name: str = "PythonCodeExecutor"
    description: str = "Use for questions requiring calculations, filtering, or displaying data via `print()`. The code should produce output that can be captured by `print()`."
    args_schema: type[BaseModel] = PythonCodeExecutorInput

    def _run(self, code: str):
        if state.df is None: return "Error: The DataFrame is not loaded. Please restart the application."
        try:
            df = state.df.copy()
            local_namespace = {'df': df, 'pd': pd}
            buffer = io.StringIO()
            sys.stdout = buffer
            exec(code, {}, local_namespace)
            sys.stdout = sys.__stdout__
            result = buffer.getvalue()
            state.last_query_result = result
            return f"Code executed successfully. Output:\n{result}"
        except Exception as e:
            sys.stdout = sys.__stdout__
            return f"Error executing Python code: {e}. Please check the code for syntax errors or incorrect column names. Available columns are: {state.df.columns.tolist()}"

class PlottingToolInput(BaseModel):
    code: str = Field(description="The Python code to execute for generating a plot. It must use matplotlib or seaborn and save the plot to the './plots/' directory. It must not use print().")

class PlottingTool(BaseTool):
    name: str = "PlottingTool"
    description: str = "Use this tool ONLY for creating visualizations like a 'plot', 'chart', or 'graph'. The code must be a self-contained script that includes all data preparation and saves the figure to the './plots/' directory."
    args_schema: type[BaseModel] = PlottingToolInput

    def _run(self, code: str):
        if state.df is None:
            return "Error: The DataFrame is not loaded. Please restart the application."
        try:
            df = state.df.copy()
            plots_dir = './plots'
            os.makedirs(plots_dir, exist_ok=True)

            # Get a snapshot of files before execution
            files_before = set(os.listdir(plots_dir))

            # Always set plots_dir in the local namespace as './plots'
            local_namespace = {'df': df, 'plt': plt, 'sns': sns, 'pd': pd, 'os': os}
            exec(code, local_namespace)

            # Check for a new file
            files_after = set(os.listdir(plots_dir))
            new_files = files_after - files_before

            if not new_files:
                return "Error: Plotting code executed, but no new plot file was saved in the './plots/' directory. Ensure your code calls `plt.savefig()`."
            
            new_plot_name = new_files.pop()
            new_plot_path = os.path.join(plots_dir, new_plot_name)
            
            # Automatically open the plot file
            try:
                if sys.platform == "win32":
                    os.startfile(new_plot_path)
                else:
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.call([opener, new_plot_path])
                return f"Plot saved successfully to {os.path.normpath(new_plot_path)} and opened for you."
            except Exception as open_error:
                return f"Plot saved successfully to {os.path.normpath(new_plot_path)}, but could not be opened automatically: {open_error}"

        except Exception as e:
            return f"Error executing plotting code: {e}. Available columns are: {state.df.columns.tolist()}"

class UpdateCategoryMappingInput(BaseModel):
    category: str = Field(description="The name of the category to add or update.")
    keywords: List[str] = Field(description="A list of keywords associated with this category.")

class UpdateCategoryMappingTool(BaseTool):
    name: str = "UpdateCategoryMappingTool"
    description: str = "Use this tool to add new categories or update existing ones with a list of keywords. This helps the system categorize transactions more accurately."
    args_schema: type[BaseModel] = UpdateCategoryMappingInput

    def _run(self, category: str, keywords: List[str]):
        mapping_path = os.path.abspath("financial_analyst/category_mapping.json")
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
        except FileNotFoundError:
            mapping = {}
        except json.JSONDecodeError:
            return f"Error reading {mapping_path}. Please check the file format."

        mapping[category] = [k.upper() for k in keywords] # Store keywords in uppercase

        try:
            with open(mapping_path, 'w') as f:
                json.dump(mapping, f, indent=4)
        except Exception as e:
            return f"Error writing to {mapping_path}: {e}"

        # Re-categorize the DataFrame in memory if it exists
        if state.df is not None:
            def categorize(desc):
                d = str(desc).upper()
                # Ensure mapping keys are strings
                return next((cat for cat, kws in mapping.items() if isinstance(cat, str) and any(kw in d for kw in kws)), 'Others')
            state.df['Category'] = state.df['Description'].apply(categorize)
            print("DataFrame re-categorized with updated mapping.")

        return f"Category '{category}' updated successfully with keywords: {keywords}."

class SaveDataInput(BaseModel):
    filePath: str = Field(description="The absolute path including filename and extension (e.g., .csv, .md, .txt) where the data should be saved.")
    contentType: str = Field(description="The type of content to save. Use 'dataframe' to save the main processed financial data table (as CSV). Use 'last_output' to save the output from the last PythonCodeExecutor run (as text).")

class SaveDataTool(BaseTool):
    name: str = "SaveDataTool"
    description: str = "Use this tool to save data to a file. Can save the main processed DataFrame to CSV or the output of the last PythonCodeExecutor run to a text file."
    args_schema: type[BaseModel] = SaveDataInput

    def _run(self, filePath: str, contentType: str = 'dataframe'):
        try:
            if contentType == 'dataframe':
                if state.df is None:
                    return "Error: The DataFrame is not loaded. Cannot save."
                if not filePath.lower().endswith('.csv'):
                    filePath += '.csv'
                    print(f"Warning: Appending .csv extension to the file path: {filePath}")
                state.df.to_csv(filePath, index=False)
                return f"DataFrame successfully saved to {os.path.normpath(filePath)}"
            elif contentType == 'last_output':
                if not state.last_query_result:
                    return "Error: No output available from the last PythonCodeExecutor run to save."
                # Determine appropriate extension if none provided
                if not any(filePath.lower().endswith(ext) for ext in ['.md', '.txt', '.log', '.json', '.csv']):
                     filePath += '.txt'
                     print(f"Warning: Appending .txt extension to the file path: {filePath}")
                with open(filePath, 'w') as f:
                    f.write(state.last_query_result)
                return f"Last tool output successfully saved to {os.path.normpath(filePath)}"
            else:
                return f"Error: Invalid contentType '{contentType}'. Must be 'dataframe' or 'last_output'."
        except Exception as e:
            return f"Error saving data to file {os.path.normpath(filePath)}: {e}"

class RemoveDuplicatesInput(BaseModel):
    # No specific input needed for this tool as it operates on the loaded DataFrame
    pass

class RemoveDuplicatesTool(BaseTool):
    name: str = "RemoveDuplicatesTool"
    description: str = "Use this tool to remove duplicate transactions from the loaded data based on the 'Description' column. It keeps the first occurrence of each unique description."
    args_schema: type[BaseModel] = RemoveDuplicatesInput

    def _run(self, *args, **kwargs):
        if state.df is None:
            return "Error: The DataFrame is not loaded. Cannot remove duplicates."
        
        initial_rows = len(state.df)
        
        # Remove duplicates based on 'Description', keeping the first occurrence
        state.df.drop_duplicates(subset=['Description'], keep='first', inplace=True)
        
        rows_after = len(state.df)
        duplicates_removed = initial_rows - rows_after
        
        return f"Successfully removed {duplicates_removed} duplicate transactions based on Description. The DataFrame now has {rows_after} rows."

class EmptyInput(BaseModel):
    # Use for tools that do not require any input arguments.
    pass

class DisplayAccountDetailsTool(BaseTool):
    name: str = "DisplayAccountDetailsTool"
    description: str = "Use this tool to display the account details that were parsed from the bank statements."
    args_schema: type[BaseModel] = EmptyInput # Use the empty input model

    def _run(self, *args, **kwargs):
        if not state.account_details:
            return "Account details are not available."

        details_str = "Here are the details associated with your account:\n\n"
        for key, value in state.account_details.items():
            details_str += f"- **{key}:** {value}\n"
        return details_str

# --- 3. The Supervisor Agent (with refined prompt) ---

def create_financial_copilot():
    """Initializes and returns the main financial co-pilot agent."""
    print("🧠 Initializing FinCo-pilot...")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Make sure it's in your .env file.")

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    tools = [
        Tool(name="DataInspector", func=inspect_data, description="Provides a concise summary of the master financial DataFrame. Use this tool first to understand the data's schema, categories, and date range."),
        PythonCodeExecutorTool(),
        PlottingTool(),
        UpdateCategoryMappingTool(),
        SaveDataTool(),
        RemoveDuplicatesTool(),
        DisplayAccountDetailsTool() # Add the new tool
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f'''You are "FinCo-pilot," a world-class financial analyst assistant. Your task is to help a user by intelligently using tools to write and execute Python code.

        You have access to the user's account details in the `state.account_details` dictionary. If the key 'Account Holder Name' exists in this dictionary, you can use the value to greet the user by name in your initial response.

        **Core Workflow:**
        1.  **Inspect First:** On the first turn, ALWAYS call `DataInspector` to understand the data schema.
        2.  **Generate Code:** Based on the user's request and the data schema, generate Python code for either the `PythonCodeExecutor` or `PlottingTool`.
        3.  **Self-Correction Loop:
            - If a tool returns an error, DO NOT stop. Analyze the error message provided by the tool.
            - Re-think your code, identify the mistake (e.g., wrong column name, syntax error, logical flaw), generate corrected code, and call the tool again.
            - Be persistent. If one approach fails, try a different one. For example, if a complex plot fails, try a simpler one.
            - Repeat this process up to 5 times. If you still fail, apologize and explain the final error.
        4.  **Synthesize and Respond:
            - After a successful tool use, summarize the findings for the user in a friendly, conversational way.
            - **Do not show the code by default.** Only provide the Python code block if the user explicitly asks for "the code", "script", etc.

        **CRITICAL Tool Rules:**
        - **Tool Isolation:** Each tool call is independent. Variables or data from one call (e.g., in `PythonCodeExecutor`) are NOT available in the next call.
        - **`PythonCodeExecutor`:** Use for questions requiring calculations, filtering, or displaying data via `print()`. The code should produce output that can be captured by `print()`.
        - **`PlottingTool`:** Use for *any* request involving a plot, chart, or graph. The code sent to `PlottingTool` MUST be self-contained, including all data preparation (filtering, grouping, etc.) and the `plt.savefig()` command. Do NOT use `print()` in this tool.
        - **DO NOT** call `PythonCodeExecutor` to find data and then call `PlottingTool` in a separate step to plot it. All data preparation for a plot MUST happen inside the single code block passed to `PlottingTool`.
        - **`UpdateCategoryMappingTool`:** Use this tool when the user asks to add or update categories for transaction categorization. Provide the category name and a list of keywords.
        - **`SaveDataTool`:** Use this tool when the user explicitly asks to save data to a file. Specify the `filePath` and `contentType` ('dataframe' or 'last_output').
        - **`RemoveDuplicatesTool`:** Use this tool when the user asks to remove duplicate transactions from the data based on the 'Description' column.
        - **`DisplayAccountDetailsTool`:** **CRITICAL:** Use this tool *whenever* the user asks for their account details or information about their account holder details.

        **Financial Logic Rules:**
        - Spending, purchases, or debits are in the `Debit` column. The largest debit is the most expensive purchase.
        - Income, credits, or deposits are in the `Credit` column.

        **Technical Rules:**
        - Pay strict attention to column names from `DataInspector`. They are case-sensitive.
        - For dates, use 'Value Date' or 'Txn Date'. Never invent a column named 'Date'.
        '''),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=ConversationBufferWindowMemory(k=20, memory_key="chat_history", return_messages=True),
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10 # Increased iterations to allow for more complex tasks including categorization updates
    )
    
    print("🤖 FinCo-pilot is ready!")
    return agent_executor

# --- 4. Main Conversational Loop ---
if __name__ == "__main__":
    if not process_data():
        print("Could not start the agent as data processing failed.")
        exit()

    copilot = create_financial_copilot()

    # Automatically run the RemoveDuplicatesTool at startup
    print("🧹 Automatically removing duplicate transactions...")
    try:
        duplicate_removal_response = copilot.invoke({"input": "Remove duplicate transactions based on description."})
        print(f"✅ Duplicate removal result: {duplicate_removal_response['output']}")
        # pass
    except Exception as e:
        print(f"❌ Error during automatic duplicate removal: {e}")

    print("\n\n--- 🤖 Welcome to the Financial Data Co-pilot ---")
    print("I can answer questions, perform analysis, and create plots about your financial data.")
    print("Type '/quit' or '/exit' to end the session.\n")

    while True:
        try:
            query = input("You: ")
            if not query.strip():
                continue

            query_lower = query.lower()
            invoke_agent = True # Assume we will invoke the agent

            if query_lower in ['/quit', '/exit']:
                print("🤖 Goodbye! Have a great day."); break
            elif query_lower == '/restart':
                print("🔄 Restarting session...")
                if not process_data():
                    print("Could not restart the agent as data processing failed.")
                    invoke_agent = False # Don't invoke agent if restart failed
                else:
                    copilot = create_financial_copilot()
                    print("✅ Session restarted.")
                    invoke_agent = False # Don't invoke agent after successful restart
            elif query_lower == '/clear':
                print("🧹 Clearing conversation history...")
                copilot.memory.clear()
                print("✅ Conversation history cleared.")
                invoke_agent = False # Don't invoke agent after clearing history
            elif query_lower == '/help':
                print("\n--- FinCo-pilot Help ---")
                print("You can ask me questions about your financial data, like:")
                print("- 'Summarize my spending by category.'")
                print("- 'Show me my total income for 2024.'")
                print("- 'Plot my monthly spending trend.'")
                print("- 'Update the category mapping for 'Shopping' with keywords ['mall', 'store'].'")
                print("- 'Save the processed data to my_data.csv'.")
                print("\nAvailable commands:")
                print("- '/quit' or '/exit': End the session.")
                print("- '/restart': Reload data and re-initialize the agent.")
                print("- '/clear': Clear the current conversation history.")
                print("- '/help': Display this help message.")
                print("------------------------\n")
                invoke_agent = False # Don't invoke agent after showing help

            if invoke_agent:
                # If none of the commands matched and invoke_agent is still True, invoke the agent
                response = copilot.invoke({"input": query})
                print(f"\nFinCo-pilot: {response['output']}\n")

        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
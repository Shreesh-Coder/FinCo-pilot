import os
import glob
import pandas as pd

# 1. Configure your paths:
input_dir  = r'C:\Users\gupta\Downloads\sbi\xls'      # folder with your .xls files
output_dir = r'C:\Users\gupta\Downloads\sbi\csvs'     # where to drop the .csvs

# 2. Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# 3. Process every .xls (and .xlsx, if any) in that folder
for ext in ('*.xls', '*.xlsx'):
    pattern = os.path.join(input_dir, ext)
    for fp in glob.glob(pattern):
        fname = os.path.splitext(os.path.basename(fp))[0]
        out_csv = os.path.join(output_dir, f'{fname}.csv')

        df = None
        # Try fixed-width parsing first (good for space-aligned text “xls” dumps)
        try:
            df = pd.read_fwf(fp)
            print(f'✔️ read_fwf succeeded for {fp}')
        except Exception as e_fwf:
            print(f'⚠️ read_fwf failed for {fp}: {e_fwf}')
            # Fallback: read as whitespace-delimited
            try:
                df = pd.read_csv(fp, delim_whitespace=True, engine='python')
                print(f'✔️ whitespace CSV read succeeded for {fp}')
            except Exception as e_ws:
                print(f'❌ both methods failed for {fp}: {e_ws}')
                continue

        # 4. Write out the cleaned DataFrame to CSV
        try:
            df.to_csv(out_csv, index=False)
            print(f'✔️ wrote {out_csv}')
        except Exception as e_out:
            print(f'❌ failed to write {out_csv}: {e_out}')

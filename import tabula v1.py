import tabula

# Read PDF into a list of DataFrame
dfs = tabula.read_pdf(r'C:\Users\robert\Documents\GitHub\python_code_portfolio\2023-annual-report.pdf', pages='11')

# Now you can treat dfs as a list of DataFrames and process them accordingly
for df in dfs:
    print(df)
import pandas as pd

def calculate_rent_change(rent, transportation, income):
    data = {
        'expenses': ['Rent','Groceries','Utilities','Transportation','Student Loans','Misc'],
        'amount':[rent, 500, 150, transportation, 400, 400],
    
    }

    df = pd.DataFrame(data)

    total_expenses = df['amount'].sum()

    profit = income - total_expenses

    return df, profit

df, profit = calculate_rent_change(2156,454,5232)


    
print(df)

print('income:5232','Profit:', profit)

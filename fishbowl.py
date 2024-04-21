import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file (replace 'your_file.xlsx' with the actual file path)
df = pd.read_excel(r'C:\Users\robert\downloads\Untitled spreadsheet(1).xlsx')

# Extract the 'age' and 'title' columns
age_data = df['age']


# Create histograms
plt.figure(figsize=(10, 6))

# Age histogram
plt.subplot(2, 1, 1)
plt.hist(age_data, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age when salary hit 100k')



plt.tight_layout()  # Adjust subplot spacing
plt.show()

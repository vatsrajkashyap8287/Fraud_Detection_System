import pandas as pd

df = pd.read_csv('test_results.csv')

print("Pehle columns:", df.columns.tolist())
print("Rows:", len(df))

# TransactionID add karo
df.insert(0, 'TransactionID', range(3000000, 3000000 + len(df)))

# Save karo
df.to_csv('test_results.csv', index=False)

print("\nFix ho gaya! Ab columns:", df.columns.tolist())
print(df.head(3))
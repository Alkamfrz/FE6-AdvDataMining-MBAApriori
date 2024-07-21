# Import the pandas library
import pandas as pd

# Import the Apriori algorithm
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Convert transactions to a one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder

# Sample transactions
transactions = [
    ['egg', 'bread'],
    ['juice', 'egg', 'butter'],
    ['juice', 'egg', 'bread'],
    ['juice', 'bread'],
    ['juice', 'egg'],
    ['juice', 'bread', 'butter'],
    ['juice', 'egg', 'butter'],
    ['bread', 'butter'],
    ['juice', 'bread'],
    ['egg', 'butter'],
    ['juice', 'egg', 'butter']
]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Set precision for floating point numbers
pd.options.display.float_format = '{:,.2f}'.format

# Sort rules by confidence and lift for better insights
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

# Select and rename columns for better readability
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
rules.columns = ['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift']

# Reset index for a cleaner look
rules.reset_index(drop=True, inplace=True)

# Display the formatted rules
print(rules)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
# Generate Sample Sales Dataset
data = {
    'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor'], size=100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], size=100),
    'Sales': np.random.randint(50, 500, size=100),
    'Profit': np.random.uniform(10, 200, size=100)
}
df = pd.DataFrame(data)

# Save dataset as CSV (for external loading)
df.to_csv('sales_data.csv', index=False)

# Load dataset
try:
    df = pd.read_csv('sales_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset file not found.")

# Display first few rows
print("\nFirst few rows:")
print(df.head())

# Check data types and missing values
print("\nData Types and Missing Values:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values (if any)
df.fillna(method='ffill', inplace=True)

# Task 2: Basic Data Analysis
# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by 'Product' and calculate mean sales
print("\nAverage Sales by Product:")
print(df.groupby('Product')['Sales'].mean())

# Task 3: Data Visualization
# Line chart for sales trend
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Sales'], marker='o', linestyle='-')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Bar chart for average sales by product
plt.figure(figsize=(8, 5))
sns.barplot(x='Product', y='Sales', data=df, estimator=np.mean)
plt.title('Average Sales by Product')
plt.xlabel('Product')
plt.ylabel('Average Sales')
plt.show()

# Histogram for sales distribution
plt.figure(figsize=(8, 5))
plt.hist(df['Sales'], bins=10, color='skyblue', edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Scatter plot for sales vs profit
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Sales', y='Profit', data=df)
plt.title('Sales vs Profit')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.show()

print("\nAnalysis completed successfully!")

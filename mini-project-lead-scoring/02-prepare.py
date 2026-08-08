import numpy as np
import pandas as pd

# 1. Load
df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')

# 2. Drop duration (data leakage)
# TODO: one line — df.drop(...)
df.drop(columns = ['duration'], inplace=True)

# 3. Fix pdays sentinel
# TODO: create was_contacted_before, create pdays_actual, drop original pdays
df['was_contacted_before'] = (df['pdays'] < 999).astype(int)
df['pdays_actual'] = df['pdays'].where(df['pdays'] < 999, 0)
df = df.drop(columns=['pdays'])



# 4. Encode target: y from 'yes'/'no' to 1/0
# TODO: one line
df['y'] = df['y'].map({'yes': 1, 'no': 0})


# 5. One-hot encode categoricals
# TODO: pd.get_dummies(...)
df = pd.get_dummies(df, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'], drop_first=True).astype(int)

# 6. Feature scale numerics (z-score)
# TODO: for each numeric column: (col - mean) / std
numeric_columns = ['age', 'campaign', 'previous', 'pdays_actual',
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                   'euribor3m', 'nr.employed']

for col in numeric_columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# 7. Train/test split (80/20)
# TODO: shuffle, then split 
# Separate features (X) from target (y)
y = df['y'].values
X = df.drop(columns=['y']).values

# Shuffle — same permutation for X and y so rows stay matched
np.random.seed(42)
indices = np.random.permutation(len(df))
X, y = X[indices], y[indices]

# Split at 80%
split = int(0.8 * len(df))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Verify
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")


np.savez('data/prepared.npz',
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test)
print("Saved prepared arrays to data/prepared.npz")
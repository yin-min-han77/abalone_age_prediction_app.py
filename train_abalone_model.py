import pandas as pd
import pickle
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from imblearn.over_sampling import RandomOverSampler

print("Fetching data...")
abalone = fetch_ucirepo(id=1)
X = abalone.data.features
y = abalone.data.targets

df = pd.concat([X, y], axis = 1)

ordinal = OrdinalEncoder()
df[['Sex']] = ordinal.fit_transform(df[["Sex"]])

df['Age_Group'] = df['Rings'].apply(lambda x: 'Older' if x > 12 else 'Young')

X = df.drop(columns = ['Age_Group'])
y = df['Age_Group']

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_sample, y_sample = ros.fit_resample(X, y)
X_sample.shape, y_sample.shape

X = X_sample.drop(columns= 'Rings')
y = X_sample['Rings']

# #applying Standard Scaler to X
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# #X_scaled to X dataframe
# X = pd.DataFrame(X_scaled, columns=X.columns)

# pca = PCA(n_components=2)
# data = pca.fit_transform(X)
# X[['PC1', 'PC2']] = data 

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42, shuffle= True)

from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, y_train)

y_pred = linear.predict(X_test)

# --- EVALUATION ---
print("Evaluating model...")
y_pred = linear.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

filename = 'abalone_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump((linear, ordinal), file)

print(f"Model and Encoder saved successfully as {filename}")

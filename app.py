import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn import metrics
from joblib import dump

# Load dataset
df = pd.read_csv(r'C:\Users\aishw\OneDrive\Documents\price_bike\data\BIKE_DETAILS.csv')

# Handle missing values in 'ex_showroom_price' column
df['ex_showroom_price'] = df['ex_showroom_price'].fillna(df['ex_showroom_price'].median())

# Feature and target selection
X = df[['name', 'year', 'km_driven', 'seller_type', 'owner', 'ex_showroom_price']]
y = df['selling_price']

# Convert categorical features to numeric using one-hot encoding
X = pd.get_dummies(X, columns=['name', 'seller_type', 'owner'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle NaN values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Create a pipeline
pipeline = Pipeline([
    ('imputer', imputer),
    ('model', RandomForestRegressor())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline
score = pipeline.score(X_test, y_test)
print("Pipeline Model score:", score)

# Define the parameter grid for RandomizedSearchCV
random_grid = {
    'model__n_estimators': [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
    'model__max_features': ['auto', 'sqrt'],
    'model__max_depth': [int(x) for x in np.linspace(5, 30, num=6)],
    'model__min_samples_split': [2, 5, 10, 15, 100],
    'model__min_samples_leaf': [1, 2, 5, 10]
}

# Set up RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator=pipeline, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)

# Fit the randomized search model
rf_random.fit(X_train, y_train)

# Get the best parameters
best_params = rf_random.best_params_
print("Best Parameters: ", best_params)

# Get the best model
best_model = rf_random.best_estimator_

# Make predictions using the best model
predictions = best_model.predict(X_test)

# Calculate metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Save the best model
dump(best_model, 'model1.joblib')

# Save columns used for training
dump(X.columns, 'model_columns.joblib')
from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd

app = Flask(__name__)

@app.route('/')
def show():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def pred():
    try:
        model = load('model1.joblib')
        model_columns = load('model_columns.joblib')
    except EOFError as e:
        return f"Error loading model: EOFError - {e}"
    except Exception as e:
        return f"Error loading model: {e}"
    
    name = request.form.get('name')
    year = int(request.form.get('year'))
    km_driven = int(request.form.get('km_driven'))
    seller_type = request.form.get('seller_type')
    owner = request.form.get('owner')
    ex_showroom_price = float(request.form.get('ex_showroom_price'))

    # Prepare the features for prediction
    input_features = pd.DataFrame([[name, year, km_driven, ex_showroom_price, seller_type, owner]], 
                                  columns=['name', 'year', 'km_driven', 'ex_showroom_price', 'seller_type', 'owner'])

    # Convert categorical features to numeric using one-hot encoding
    input_features = pd.get_dummies(input_features, columns=['name', 'seller_type', 'owner'], drop_first=True)

    # Ensure that the feature columns match the training data
    missing_cols = set(model_columns) - set(input_features.columns)
    for col in missing_cols:
        input_features[col] = 0
    input_features = input_features[model_columns]

    try:
        predicted_value = model.predict(input_features)
        return render_template('index.html', prediction_text="Estimated selling price: ₹{:.2f}".format(predicted_value[0]))
    except Exception as e:
        return f"Error predicting price: {e}"

if __name__ == "__main__":
    app.run(debug=True)

# Import helpful libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Load the data, and separate the target
trian_path = "./train.csv"
train_data = pd.read_csv(trian_path)
y = train_data.target

# Create X 
features = ['gravity','ph','osmo','cond','urea','calc']

# Select columns corresponding to features
X = train_data[features]


# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# create Random Forest model which will train on all training data
rf_model = RandomForestRegressor(random_state=1)
# fit rf_model_on_full_data on all data from the training data
rf_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = rf_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions)) 






# right_scores = 0
# wrong_scores = 0
# for p in rf_train_predictions:
#     print(p)

# print(rf_train_mae)

test_path = "./test.csv"
test_data = pd.read_csv(test_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
test_X = test_data[features]

# make predictions 
test_preds = rf_model.predict(test_X)




# Save predictions in the format used for competition scoring
output = pd.DataFrame({"id": test_data.id, "target": test_preds})
output.to_csv("submission.csv", index = False)









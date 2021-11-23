from flwr.common import parameter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as plt
import utils
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Create LogisticRegression Model
model = LogisticRegression(
    penalty="l2",
    max_iter=10,  # local epoch
    warm_start=True,  # prevent refreshing weights when fitting
)
model.classes_ = np.array([0, 1])

# Load last session weights if they exist
sessions = ['no session']
for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        if name.find('Session') != -1:
            sessions.append(name)

print(sessions)

if os.path.exists(f'{sessions[-1]}/global_session_model.npy'):
    parameters = np.load(f"{sessions[-1]}/global_session_model.npy", allow_pickle=True)
    weights = parameters[0]
else:
    weights = None


# set model weights with last session global weights
utils.set_model_params(model, parameter.parameters_to_weights(weights))


# Load dataset
df = pd.read_csv('cleaned_dataset.csv')
X = df.drop(['Target'], axis=1)
y = df['Target']

prediction = model.predict(X)
reviews = pd.DataFrame(prediction, index=df.index, columns=['Prediction']).astype(int)

compare = pd.Series(reviews['Prediction'] == y).value_counts()
print(compare)
reviews.plot(legend=True, marker='o')
y.plot(legend=True, marker='o')
plt.show()

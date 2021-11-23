import flwr as fl
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils
import sys
import warnings
from fastapi import FastAPI

if not sys.warnoptions:
    warnings.simplefilter("ignore")

app = FastAPI()


# Parse command line arguments
# parser = argparse.ArgumentParser(description="restaurant rating")
# parser.add_argument("--train_start", type=int, required=True)
# parser.add_argument("--train_end", type=int, required=True)
# args = parser.parse_args()

@app.post("/participateFL")
def listen_and_participate(train_start:int, train_end:int, ipaddress:str ,port:int):
    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=10,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)


    # Load dataset
    df = pd.read_csv('cleaned_dataset.csv')
    X = df.drop(['Target'], axis=1)
    y = df['Target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, shuffle=False)
    x_train, y_train = x_train[train_start:train_end], y_train[train_start:train_end]


    # Define Flower client
    class FlowerClient(fl.client.NumPyClient):

        def get_parameters(self):
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):
            utils.set_model_params(model, parameters)
            model.fit(x_train, y_train)
            return utils.get_model_parameters(model), len(x_train), {}

        def evaluate(self, parameters, config):
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(x_test))
            accuracy = model.score(x_test, y_test)
            print("accuracy: ", accuracy)
            return loss, len(x_test), {"accuracy": accuracy}


    # Start Flower client
    fl.client.start_numpy_client(
        server_address=ipaddress + ':' + str(port),
        client=FlowerClient(),
        grpc_max_message_length=1024 * 1024 * 1024
    )



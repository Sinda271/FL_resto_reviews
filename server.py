import flwr as fl
import numpy as np
import os
from typing import Callable, Dict
import argparse
import datetime as dt

from fastapi import FastAPI
# Parse command line argument
parser = argparse.ArgumentParser(description="Food waste")
parser.add_argument("--num_rounds", type=int, required=True)
parser.add_argument("--ipadress", type=str, required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--resume", default=False, action="store_true")
args = parser.parse_args()



#app = FastAPI()

#@app.post("/FLsession")
#def start_fl_session(num_rounds:int,ipaddress:str ,port:int,resume:bool):

# define date and time to save weights in directories
today = dt.datetime.today()
session = today.strftime("%d-%m-%Y-%H-%M-%S")

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            rnd,
            results,
            failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:

            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")

            if not os.path.exists(f"Session-{session}"):
                os.makedirs(f"Session-{session}")
                if rnd < args.num_rounds:
                    np.save(f"Session-{session}/round-{rnd}-weights.npy", aggregated_weights)
                elif rnd == args.num_rounds:
                    np.save(f"Session-{session}/global_session_model.npy", aggregated_weights)
            else:
                if rnd < args.num_rounds:
                    np.save(f"Session-{session}/round-{rnd}-weights.npy", aggregated_weights)
                elif rnd == args.num_rounds:
                    np.save(f"Session-{session}/global_session_model.npy", aggregated_weights)

        return aggregated_weights


# Define batch-size, nb of epochs and verbose for fitting
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
            "batch_size": 32,
            "epochs": 50,
            "verbose": 0,
        }
        return config

    return fit_config


# Define hyper-parameters for evaluation
def evaluate_config(rnd: int):
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps, "verbose": 0}


# Create strategy and run server
# Load last session weights if they exist
sessions = ['no session']
for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        if name.find('Session') != -1:
            sessions.append(name)

if os.path.exists(f'{sessions[-1]}/global_session_model.npy'):
    initial_parameters = np.load(f"{sessions[-1]}/global_session_model.npy", allow_pickle=True)
    initial_weights = initial_parameters[0]
else:
    initial_weights = None

if args.resume:
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        initial_parameters=initial_weights,
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=evaluate_config,
    )
else:
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        initial_parameters=None,
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=evaluate_config,
    )


fl.server.start_server(
    server_address=args.ipadress + ':' + str(args.port),
    config={"num_rounds": args.num_rounds},
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)



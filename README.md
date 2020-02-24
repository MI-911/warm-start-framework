# Model Framework 

Data loaders and model training/evaluation pipelines used in the pre-specialisation project for group MI911 @ Department of Computer Science, Aalborg University, Denmark.

## Get the most recent data

From the `./data_loading/download_raw_data.py` scripts, run `download_data()` to download the most recent MindReader data.

## Generate datasets
Run the `./data_loading/data_generator.py` script to generate data. Consult `generate()` to adjust whether or not to include top-popular items in the test set.

## Running the experiments
Then, build the Docker image: 
```
docker build -t mi911/runner .
```
When running the container, you have the following options: 
1. `--include [MODEL NAME LIST]` for running only specific models (defaults to all models)
    - Model names: `item-knn`, `user-knn`, `mf`, `svd`, `bpr`, `transe`, `transe-kg`, `transh`, `transh-kg`, `pr-collab`, `pr-kg`, `pr-joint`, `random`, `top-pop`, and `cbf-item-knn`.
2. `--exclude [MODEL NAME LIST]` for running all models except specific ones (defaults to none)
3. `--experiments [EXPERIMENT NAME LIST]` for the experiments to run
    - Experiment names (prefixed `wtp-` and `ntp-` for with and without top-popular items in the test set, respectively): `all_movies`, `all_entities`, `substituting-3-4`, `substituting-2-4`, and `substituting-1-4`.
4. `--debug` for printing debug-level logs to the terminal.

For example, if we want to run the experiment containing all movie ratings with top-popular items in the test set running only the SVD and BPR models, the following command will work:
``` 
docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr svd --experiments wtp-all_movies --debug
```

Run the `./run.sh` script for running all models in all experiments. *Note: Runs all experiments in parallel*.
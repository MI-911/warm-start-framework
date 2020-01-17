# Model Framework 

Data loaders and model training/evaluation pipelines. 

From the `data_loading/download_raw_data.py` scripts, run `download_data()` to download the most recent MindReader data.

## Running the experiments
1. Build the Docker image:

`docker build -t mi911/runner .`

2. Run a container with the newly built image:

`docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --experiments wtp-all_movies wtp-all_entities --debug`

In the example above we run to experiments, `wtp-all_movies` and `wtp-all_entities`. This command will mount `${PWD}/data`, which should contain the experiments. The available data can be found [here](http://mindreader.tech/data.tar.gz).

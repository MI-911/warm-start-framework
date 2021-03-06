# Warm-start framework 
Data loaders and model training/evaluation pipelines written using Python 3.8 used in the MindReader paper published at CIKM 2020, [https://doi.org/10.1145/3340531.3412759](https://doi.org/10.1145/3340531.3412759). 
You can read more about our dataset at [https://mindreader.tech/dataset](https://mindreader.tech/dataset), and remember to cite our work:

```bib
@inproceedings{brams2020mindreader,
  title={MindReader: Recommendation over Knowledge Graph Entities with Explicit User Ratings},
  author={Brams, Anders H and Jakobsen, Anders L and Jendal, Theis E and Lissandrini, Matteo and Dolog, Peter and Hose, Katja},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={2975--2982},
  year={2020}
}
```

## Get the most recent data

Run the `./data_loading/download_raw_data.py` script to download the most recent MindReader data.

## Generate datasets
Run the `./data_generation_entry.py` script to generate data. Consult `generate()` to adjust whether or not to include top-popular items in the test set.

## Running the experiments
Run the `./run.sh` script for running all models in all experiments.
Results are written to `./results/`.

### Running specific models and/or experiments
First, build the Docker image: 
```
docker build -t mi911/runner .
```
When running the container, you have the following options: 
1. `--include [MODEL NAME LIST]` for running only specific models (defaults to all models)
    - Model names: `item-knn`, `user-knn`, `mf`, `svd`, `bpr`, `transe`, `transe-kg`, `transh`, `transh-kg`, `ppr-collab`, `ppr-kg`, `ppr-joint`, `random`, `top-pop`, and `cbf-item-knn`.
2. `--exclude [MODEL NAME LIST]` for running all models except specific ones (defaults to none)
3. `--experiments [EXPERIMENT NAME LIST]` for the experiments to run
    - Experiment names (prefixed `wtp-` and `ntp-` for with and without top-popular items in the test set, respectively): `all_movies`, `all_entities`, `substituting-3-4`, `substituting-2-4`, and `substituting-1-4`.
4. `--debug` for printing debug-level logs to the terminal.

For example, if we want to run the experiment containing all movie ratings with top-popular items in the test set running only the SVD and BPR models, the following command will work:
``` 
docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr svd --experiments wtp-all_movies
```


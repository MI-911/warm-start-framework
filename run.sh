git pull
docker build -t mi911/runner .

# Run all models on all experiments. Results written to ./results.
docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner

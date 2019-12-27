git pull
docker build -t mi911/runner .
docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --exclude mf joint-mf --experiments all_movies

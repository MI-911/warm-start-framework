git pull
docker build -t mi911/runner .
docker run -d -v ${PWD}/data_loading/mindreader:/app/data_loading/mindreader mi911/runner

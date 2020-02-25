git pull
# docker build -t mi911/runner .

python3 entrypoint.py --include svd bpr --experiments wtp-substituting-1-4

#
## With top-popular items in the test set
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner wtp-all_movies
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner wtp-entities
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner wtp-substituting-3-4
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner wtp-substituting-2-4
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner wtp-substituting-1-4
#
## Without top-popular items in the test set
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner ntp-all_movies
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner ntp-entities
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner ntp-substituting-3-4
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner ntp-substituting-2-4
#docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner ntp-substituting-1-4

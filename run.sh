git pull
docker build -t mi911/runner .

#wtp
docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr top-pop random svd --experiments wtp-substituting-1-4
docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr top-pop random svd --experiments wtp-substituting-2-4
docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr top-pop random svd --experiments wtp-substituting-3-4
docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr top-pop random svd --experiments wtp-substituting-4-4

#ntp
docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr top-pop random svd --experiments ntp-substituting-1-4
docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr top-pop random svd --experiments ntp-substituting-2-4
docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr top-pop random svd --experiments ntp-substituting-3-4
docker run -d -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr top-pop random svd --experiments ntp-substituting-4-4
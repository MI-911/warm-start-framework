git pull
# docker build -t mi911/runner .

#wtp
# docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr svd --experiments wtp-all_movies --debug
# docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr svd --experiments wtp-all_entities --debug
# docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner --include transe-kg --experiments wtp-all_movies --debug
# docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner --include transh-kg --experiments wtp-all_movies --debug

python3.7 entrypoint.py --include transh-kg --experiments wtp-all_movies --debug
python3.7 entrypoint.py --include transh-kg --experiments wtp-all_entities --debug
python3.7 entrypoint.py --include transh-kg --experiments wtp-substituting-4-4 --debug
python3.7 entrypoint.py --include transh-kg --experiments wtp-substituting-4-3 --debug
python3.7 entrypoint.py --include transh-kg --experiments wtp-substituting-4-2 --debug
python3.7 entrypoint.py --include transh-kg --experiments wtp-substituting-4-1 --debug

python3.7 entrypoint.py --include transh-kg --experiments ntp-all_movies --debug
python3.7 entrypoint.py --include transh-kg --experiments ntp-all_entities --debug
python3.7 entrypoint.py --include transh-kg --experiments ntp-substituting-4-4 --debug
python3.7 entrypoint.py --include transh-kg --experiments ntp-substituting-4-3 --debug
python3.7 entrypoint.py --include transh-kg --experiments ntp-substituting-4-2 --debug
python3.7 entrypoint.py --include transh-kg --experiments ntp-substituting-4-1 --debug


# python3.7 entrypoint.py --include transe-kg --experiments wtp-all_movies --debug
# python3.7 entrypoint.py --include transe-kg --experiments wtp-all_entities --debug
# python3.7 entrypoint.py --include transe-kg --experiments wtp-substituting-4-4 --debug
# python3.7 entrypoint.py --include transe-kg --experiments wtp-substituting-4-3 --debug
# python3.7 entrypoint.py --include transe-kg --experiments wtp-substituting-4-2 --debug
# python3.7 entrypoint.py --include transe-kg --experiments wtp-substituting-4-1 --debug

# python3.7 entrypoint.py --include transe-kg --experiments ntp-all_movies --debug
# python3.7 entrypoint.py --include transe-kg --experiments ntp-all_entities --debug
# python3.7 entrypoint.py --include transe-kg --experiments ntp-substituting-4-4 --debug
# python3.7 entrypoint.py --include transe-kg --experiments ntp-substituting-4-3 --debug
# python3.7 entrypoint.py --include transe-kg --experiments ntp-substituting-4-2 --debug
# python3.7 entrypoint.py --include transe-kg --experiments ntp-substituting-4-1 --debug

#ntp
# docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr svd --experiments ntp-all_movies --debug
# docker run -d -v ${PWD}/.data:/app/data -v ${PWD}/results:/app/results mi911/runner --include bpr svd --experiments ntp-all_entities --debug
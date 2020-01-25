echo "Adding entity ratings (WTP)"
docker run --rm -v ${PWD}/data:/app/data -v ~/Thesis/results:/app/results mi911/runner --experiments wtp-all_movies wtp-all_entities --summary --table --test wtp-all_movies

echo "Adding entity ratings (NTP)"
docker run --rm -v ${PWD}/data:/app/data -v ~/Thesis/results:/app/results mi911/runner --experiments ntp-all_movies ntp-all_entities --summary --table --test ntp-all_movies

echo "Substituting entity ratings (WTP)"
docker run --rm -v ${PWD}/data:/app/data -v ~/Thesis/results:/app/results mi911/runner --experiments wtp-substituting-4-4 wtp-substituting-3-4 wtp-substituting-2-4 wtp-substituting-1-4 --summary --table --test wtp-substituting-4-4

echo "Substituting entity ratings (NTP)"
docker run --rm -v ${PWD}/data:/app/data -v ~/Thesis/results:/app/results mi911/runner --experiments ntp-substituting-4-4 ntp-substituting-3-4 ntp-substituting-2-4 ntp-substituting-1-4 --summary --table --test ntp-substituting-4-4

echo "Remove only (WTP)"
docker run --rm -v ${PWD}/data:/app/data -v ~/Thesis/results:/app/results mi911/runner --experiments wtp-substituting-4-4 wtp-substituting-1-0 --summary --table --test wtp-substituting-4-4

"""Bibliometric counts from OpenAlex (title search) for the motivation section."""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
import urllib.parse

OUT = sys.argv[1]
API = "https://api.openalex.org/works"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")  # optional: e-mail for the OpenAlex polite pool

MH = ('("metaheuristic" OR "meta heuristic" OR "genetic algorithm" OR "particle swarm" OR "differential evolution" '
      'OR "ant colony" OR "simulated annealing" OR "tabu search" OR "evolutionary algorithm" OR "swarm intelligence" '
      'OR "grey wolf" OR "whale optimization" OR "artificial bee colony" OR "harmony search" OR "firefly algorithm" '
      'OR "cuckoo search" OR "memetic algorithm")')

ALGS = {
    "GA": '"genetic algorithm"', "PSO": '"particle swarm"', "DE": '"differential evolution"',
    "ACO": '"ant colony"', "SA": '"simulated annealing"', "TS": '"tabu search"', "GWO": '"grey wolf"',
    "WOA": '"whale optimization"', "ABC": '"artificial bee colony"', "CS": '"cuckoo search"',
    "FA": '"firefly algorithm"', "HS": '"harmony search"', "LS": '"local search"', "NM": '"Nelder Mead"',
}


CACHE_PATH = OUT + ".cache.json"
CACHE = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def count(search, group_by_year=False, years="2000-2025"):
    q = {"filter": f"title.search:{search},publication_year:{years},type:article|proceedings-article|book-chapter"}
    if MAILTO:
        q["mailto"] = MAILTO
    if os.environ.get("OPENALEX_API_KEY"):
        q["api_key"] = os.environ["OPENALEX_API_KEY"]
    if group_by_year:
        q["group_by"] = "publication_year"
    url = API + "?" + urllib.parse.urlencode(q)
    url_key = url.split("&api_key=")[0]
    if url_key in CACHE:
        return {int(k): v for k, v in CACHE[url_key].items()} if group_by_year else CACHE[url_key]
    for attempt in range(5):
        # curl is used instead of urllib, which stalls on this network (slow IPv6 fallback)
        r = subprocess.run(["curl", "-s", "-4", "--max-time", "60", url], capture_output=True, text=True)
        try:
            d = json.loads(r.stdout)
            if "meta" in d:
                break
        except ValueError:
            pass
        time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(url_key)
    val = {int(g["key"]): g["count"] for g in d["group_by"]} if group_by_year else d["meta"]["count"]
    CACHE[url_key] = val
    json.dump(CACHE, open(CACHE_PATH, "w"))
    return val


res = {"query_date": time.strftime("%Y-%m-%d"), "mh_query": MH}
res["all_by_year"] = count(MH, True)
res["hybrid_by_year"] = count(f'hybrid AND {MH}', True)
res["memetic_by_year"] = count('"memetic algorithm"', True)
res["pairs"] = {}
keys = list(ALGS)
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        res["pairs"][f"{a}|{b}"] = count(f"{ALGS[a]} AND {ALGS[b]}")
res["single"] = {a: count(ALGS[a]) for a in keys}
res["hybrid_single"] = {a: count(f"hybrid AND {ALGS[a]}") for a in keys}
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)

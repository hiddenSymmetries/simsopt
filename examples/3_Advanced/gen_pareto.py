import sys
# sys.path.append("/global/homes/s/shurwitz/Force-Optimizations/" )

from analysis_tools import *

iteration = int(sys.argv[1])
INPUT_DIR = f"../output/QA/without-force-penalty/{iteration-1}/optimizations/"
OUTPUT_DIR = f"../output/QA/without-force-penalty/{iteration-1}/pareto/"
_, df_filtered, _ = get_dfs(INPUT_DIR=INPUT_DIR)
df_sorted = df_filtered.sort_values(by=["normalized_BdotN"])

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    for UUID in df_sorted[:100]['UUID']:
        SOURCE_DIR = glob.glob(f"../**/{UUID}/", recursive=True)[0]
        DEST_DIR = f"{OUTPUT_DIR}{UUID}/"
        shutil.copytree(SOURCE_DIR, DEST_DIR)

from optimization_tools import *

OUTPUT_DIR="./output/QA/TVE/"
INPUT_FILE="../../../tests/test_files/input.LandremanPaul2021_QA"
initial_optimizations(OUTPUT_DIR=OUTPUT_DIR, INPUT_FILE=INPUT_FILE, with_force=True, FORCE_OBJ=TVE)

# sys.path.append("/global/homes/s/shurwitz/Force-Optimizations/" )
from optimization_tools import *

OUTPUT_DIR = "../output/QA/with-force-penalty/1/optimizations/"
INPUT_FILE = "../inputs/input.LandremanPaul2021_QH_magwell"
initial_optimizations(OUTPUT_DIR=OUTPUT_DIR, INPUT_FILE=INPUT_FILE, MAXITER=1)

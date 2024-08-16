import os


parameters = {
    "jit": True
}

if "SIMSGEOJIT" in os.environ:
    parameters["jit"] = os.environ["SIMSGEOJIT"].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly']

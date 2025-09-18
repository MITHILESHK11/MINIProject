# src/aging/aging_model.py
# Simple wrapper / placeholder for aging modules.
# Real, high-quality aging requires SAM or MyTimeMachine — see links below.
import os
from typing import Union

# Recommended: use SAM (Style-based Age Manipulation) or MyTimeMachine.
# SAM: https://github.com/yuval-alaluf/SAM
# MyTimeMachine: https://mytimemachine.github.io/

def apply_aging_image_external(input_image_path: str, target_age: int, method: str = "sam") -> str:
    """
    Wrapper that calls an external aging tool (SAM / MyTimeMachine / Fast-AgingGAN).
    This function does NOT implement aging itself — you must clone the chosen repo,
    set up its environment, and provide the path to the executable/script.

    Returns the path to the generated aged image (or raises informative error).
    """
    if method == "sam":
        # Example: if you have SAM scripts locally, call them here (subprocess).
        raise RuntimeError("SAM integration not installed. See https://github.com/yuval-alaluf/SAM for instructions.")
    elif method == "mytimemachine":
        raise RuntimeError("MyTimeMachine integration not installed. See https://mytimemachine.github.io/ for instructions.")
    elif method == "fast-aginggan":
        raise RuntimeError("Fast-AgingGAN integration not installed. See https://github.com/HasnainRaz/Fast-AgingGAN for instructions.")
    else:
        raise ValueError("Unknown aging method: " + str(method))

def list_supported_methods():
    return ["sam (StyleGAN-based)", "mytimemachine (personalized)", "fast-aginggan (GAN)"]


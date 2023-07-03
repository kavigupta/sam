import os
import modular_splicing

DATA_DIRECTORY = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(modular_splicing.__file__)))
    ),
    "data",
)

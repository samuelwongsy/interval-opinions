import os.path
import numpy as np
import numpy.typing as npt


class ResultSerializer:
    """
    ResultSerializer is the serializer class to store IntervalOpinions as files.

    Attributes
    ----------
    base_folder: str
        Path of the folder to save in.
    base_file_name: str
        Name of the file.
        File is saved as {base_file_name}_{step_number}.{file_format}.
    """
    def __init__(self, base_file_name: str, base_folder: str):
        self.base_file_name = base_file_name
        self.base_folder = base_folder
        self.count = 0

    def save_results(self, opinions: npt.ArrayLike, dynamic_matrix: npt.ArrayLike) -> None:
        """
        Save opinions and dynamic_matrix as a dictionary in npz format.
        """
        file_path = f"./{os.path.join(self.base_folder, self.base_file_name)}_{self.count}"
        np.savez(file_path, opinions=opinions, dynamic_matrix=dynamic_matrix)
        self.count += 1

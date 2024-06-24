# Double Phase Hologram Simulation

This project simulates the use of double phase holography to reconstruct a given target image.

## Project Structure

- `data/`: Directory for storing images.
- `notebooks/`: Directory for Jupyter notebooks.
- `src/`: Directory for source code.
- `tests/`: Directory for test cases.
- `.gitignore`: List of files and directories to be ignored by Git.
- `README.md`: This file.
- `requirements.txt`: List of dependencies.
- `setup.py`: Setup script for the project.

## Setup

1. Clone the repository:

    ```sh
    git clone https://github.com/genesis-lite/double_phase_hologram.git
    cd double_phase_hologram
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Run the simulation:

    ```sh
    python src/hologram.py
    ```

## Dependencies

- torch
- numpy
- matplotlib
- pillow

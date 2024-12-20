# DEA and FDH Analysis Project

This project is focused on implementing and analyzing various Data Envelopment Analysis (DEA) models, including CCR, BCC, and FDH models. The project includes time complexity analysis, efficiency distribution, and frontier comparison using Python and Jupyter Notebooks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to have Python 3 installed. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Importing Modules**: The project uses several custom modules and utility functions. Ensure that your Python path is set correctly to import these modules.

2. **Running Notebooks**: Open the Jupyter Notebooks in the project directory to explore different analyses:
   - `Tutorials.ipynb`: Provides examples of DEA models and their usage.
   - `DistributionAnalysis.ipynb`: Analyzes the efficiency distribution of different models.
   - `TimeComplexityAnalysis.ipynb`: Evaluates the time complexity of DEA models with varying inputs.
   - `FrontierComparision.ipynb`: Compares the efficiency frontiers of different models.

3. **Executing Code**: Each notebook contains code cells that can be executed to perform the analyses. Follow the instructions within each notebook to understand the workflow.

## Project Structure

- **models**: Contains the implementation of DEA and FDH models.
- **utils**: Utility functions for data manipulation, plotting, and efficiency calculations.
- **notebooks**: Jupyter Notebooks for tutorials, analysis, and comparisons.

## Key Features

- **DEA Models**: Implementation of CCR, BCC, and FDH models.
- **Efficiency Analysis**: Tools to calculate and compare efficiency scores.
- **Time Complexity**: Analysis of computational efficiency with varying data sizes.
- **Visualization**: Plotting capabilities to visualize efficiency frontiers and distributions.

## Dependencies

- Python 3.12.7
- NumPy
- Matplotlib
- Seaborn
- Gurobi (for optimization tasks)
- SciPy

Ensure all dependencies are installed using the `requirements.txt` file.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code is well-documented and follows the project's coding standards.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

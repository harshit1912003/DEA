# Data Envelopment Analysis Project
This project, undertaken as part of our Master’s Thesis and Winter Internship for 2024 under Professor Aparna Mehra (HOD, Mathematics IITD), explores advanced methodologies in Data Envelopment Analysis (DEA).

Most available DEA software is either paid, proprietary, or lacks a wide spectrum of the latest models.

In our project, we investigated foundational DEA models under CRS and VRS assumptions, analyzed input- and output-oriented frameworks, and implemented sophisticated efficiency measures such as SBM and Additive models. Our work extends to Free Disposal Hull (FDH) models, complemented by the implementation of the Li-Test to statistically validate differences in efficiency and rankings between FDH and DEA models. The Li-Test, developed from scratch in Python for discrete data, enhances the robustness of our findings.

To facilitate visual analysis, we have developed various plotting functions, ensuring computational efficiency and minimal redundancy in runtime and memory usage. During the winter internship, we extended our models to handle non-homogeneous outputs, verifying results against Pooja Bansal’s implementation.

Future work includes integrating Envelopment Analysis Tree (EAT) methodology and **developing a user-friendly interface** to enhance accessibility. This project is entirely free and open-source, available for anyone to use and build upon.



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

## Project Structure

- **models**: Contains the implementation of the models.
- **utils**: Utility functions for data initialization and processing, plotting, and the Li-Test.

## Experiments (stored as jupyter notebooks)

- **Tutorial**: A tutorial on how to use this repository.


- **DistributionAnalysis**: Analysis of the distribution of efficiency scores between FDH and CCR models with VRS and CRS assumptions.
- **FrontierComparision**: Analysis of the frontier between FDH and CCR models.
-**TimeComplexityAnalysis**: Comparison of computational efficiency between the models.
<!-- - **PoojaBansal**:  -->
- **largeN_DEA**: Comparison of efficiency scores between CCR, BCC, and SBM models.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.

## Connect with Us  
**This repository is actively maintained, and we welcome suggestions and improvements.
**

Feel free to contact Ananya and/or Harshit via following links:

- **LinkedIn**:  
- **Twitter/X**:  
- **Email**:  

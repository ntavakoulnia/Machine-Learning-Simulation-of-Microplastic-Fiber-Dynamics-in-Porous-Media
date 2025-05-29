# Machine Learning Simulation of Microplastic Fiber Dynamics in Porous Media

![Predicting Microfiber Path with Machine Learning-VEED](https://github.com/user-attachments/assets/97dac43d-5885-45e7-b5bc-7b2f18185b79)

This project uses machine learning to predict the movement and localization of microplastic fibers within porous media, combining data processing, predictive modeling, and 3D visualization to simulate fiber dynamics in realistic environments.

## System Components

The system consists of three main components:

1. Data Preparation (MATLAB): Processes raw simulation data into training/test CSVs
2. Machine Learning Model (Python): Implements a Random Forest model to predict fiber paths
3. 3D Visualization (Python): Creates an animated simulation of predicted vs actual fiber paths

## Specifications

### Data Processing
- MATLAB scripts to convert VTK simulation data into structured CSV format

### Machine Learning
- Random Forest Regressor with MultiOutput wrapper for path prediction

### 3D Visualization
- Mayavi-based animation showing fiber movement through porous media

### Physics Constraints
- Grain avoidance and movement limitation algorithms ensure physically plausible predictions

## Requirements

### MATLAB
- Required for data processing

### Python
- Python 3.8+

### Required Python Packages
- numpy
- pandas
- scikit-learn
- scipy
- mayavi
- jupyter

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Machine-Learning-Simulation-of-Microplastic-Fiber-Dynamics-in-Porous-Media.git
cd Machine-Learning-Simulation-of-Microplastic-Fiber-Dynamics-in-Porous-Media

````
## Setup Process

### Prepare the Data
1. Place your VTK simulation files in the appropriate directory structure
2. Run `create_training_data.m` in MATLAB to generate training CSVs

### Train the Model
1. Run `Fiber_Predictions.py` to train the Random Forest model
2. This will generate prediction CSV files

### Visualize Results
1. Run `3D_Visulization.py` to create the 3D animation
2. The animation shows both predicted and actual fiber paths through the porous media

### Order to Run
```bash
python Fiber_Predictions.py
python 3D_Visualization.py
```

## Key Features

- Grain Avoidance: Physical constraints prevent fibers from penetrating solid grains
- Movement Smoothing: Algorithms ensure realistic fiber bending and movement  
- Comparative Visualization: Side-by-side display of predicted vs actual fiber paths
- Parameter Tuning: Grid search for optimal Random Forest hyperparameters

## File Structure

- `create_training_data.m`: MATLAB script for processing simulation data and creating training.csv
- `Fiber_Predictions.py`: Python file for model training and prediction
- `3D_Visulization.py`: Python file for 3D visualization
- `x_train.csv`: Sample training input data for all fibers with case numbers of 0.15
- `y_train.csv`: Sample training output data for all fibers with case numbers of 0.15
- `vtkRead/`: Directory containing VTK file reading utilities
  - Contains functions to read and process fiber VTK simulation files for create_training_data.m script
  - Handles data extraction and formatting for training
- `actualvalues.csv`: Actual fiber paths for comparison from Simulation
- `predictedvalues.csv`: Model predicted fiber paths based on the training data for Simulation

## Usage Notes

- Adjust file paths in the notebooks to match your local directory structure
- The model currently predicts paths for 1 randomly selected fiber (can be modified)
- Visualization parameters (colors, opacity, camera angles) can be customized

## Demonstration

The 3D visualization shows:
- Orange spheres representing solid grains
- Blue lines showing actual fiber paths
- Red lines showing predicted fiber paths
- A wavy blue surface representing fluid flow

The animation progresses through time steps, showing how fibers move through the porous domain.

## Future Work

- Incorporate more sophisticated physics constraints
- Implement realistic prediction capabilities
- Add quantitative error metrics to the visualization
- Support for 3D porous media simulations


# Machine-Learning-Simulation-of-Microplastic-Fiber-Dynamics-in-Porous-Media

This project uses machine learning to predict the movement and localization of fibers within a porous media domain. The implementation includes data processing, model training, and 3D visualization components.

Project Overview
The system consists of three main components:

Data Preparation (MATLAB): Processes raw simulation data into training/test CSVs

Machine Learning Model (Python): Implements a Random Forest model to predict fiber paths

3D Visualization (Python): Creates an animated simulation of predicted vs actual fiber paths

Specifications
Data Processing: MATLAB scripts to convert VTK simulation data into structured CSV format

Machine Learning: Random Forest Regressor with MultiOutput wrapper for path prediction

3D Visualization: Mayavi-based animation showing fiber movement through porous media

Physics Constraints: Grain avoidance and movement limitation algorithms ensure physically plausible predictions

Requirements
MATLAB (for data processing)

Python 3.8+

Required Python packages:

numpy

pandas

scikit-learn

scipy

mayavi

jupyter

Setup Instructions
Clone the repository:

bash
git clone https://github.com/yourusername/fiber-localization-prediction.git
cd fiber-localization-prediction
Set up Python environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
Prepare the data:

Place your VTK simulation files in the appropriate directory structure

Run testing.m in MATLAB to generate training CSVs

Train the model:

Run the Jupyter notebook Untitled1.ipynb to train the Random Forest model

This will generate prediction CSV files

Visualize results:

Run the Jupyter notebook Untitled2.ipynb to create the 3D animation

The animation shows both predicted and actual fiber paths through the porous media

Key Features
Grain Avoidance: Physical constraints prevent fibers from penetrating solid grains

Movement Smoothing: Algorithms ensure realistic fiber bending and movement

Comparative Visualization: Side-by-side display of predicted vs actual fiber paths

Parameter Tuning: Grid search for optimal Random Forest hyperparameters

File Structure
testing.m: MATLAB script for processing simulation data

Fiber_Predictions.py: Python file for model training and prediction

3D_Visulization.py: Python file for 3D visualization

x_train.csv: Sample training input data for all fibers with case numbers of 0.1.

y_train.csv: Sample training output data for all fibers with case numbers of 0.1.

actualvalues.csv: Actual fiber paths for comparison from Simulation

predictedvalues.csv: Model predicted fiber paths based on the training data for Simulation

Usage Notes
Adjust file paths in the notebooks to match your local directory structure

The model currently predicts paths for 3 randomly selected fibers (can be modified)

Visualization parameters (colors, opacity, camera angles) can be customized

Demonstration
The 3D visualization shows:

Orange spheres representing solid grains

Blue lines showing actual fiber paths

Red lines showing predicted fiber paths

A wavy blue surface representing fluid flow

The animation progresses through time steps, showing how fibers move through the porous domain.

Future Work
Incorporate more sophisticated physics constraints

Implement realistic prediction capabilities

Add quantitative error metrics to the visualization

Support for 3D porous media simulations

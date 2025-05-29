#!/usr/bin/env python
# coding: utf-8

# # Random Forest Tree Model For Microfiber In Porous Domain

# In[1]:


# Imported the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


# # Grain Positioning

# In[3]:


# Defined the positions and sizes of circular grains in the domain

# xc = x-coordinates of grain centers
xc = np.array([
    0.861923248843825, 0.530696997759020, 0.230307598040601, 0.756672529035511, 
    0.355867138671012, 0.130476259199288, 0.220108297265764, 0.512254287950566, 
    0.853010619375759, 0.624608543201807, 0.399531820716510, 0.673837801757064, 
    0.423871761032729, 0.444901712442721, 0.0763596893777989, 0.740776643667191, 
    0.225335687516795, 0.124745007906295, 0.937333120266778, 0.643235524280937, 
    0.315817148501495, 0.948789100265014, 0.845062624129590, 0.766136269897956, 
    0.312410891980905, 0.937354884834047, 0.584419981171847
])
# yc = y coordinates of grain centers
yc = np.array([
    0.142796932100205, 0.198844509849833, 0.0607917660763774, 0.0745644494427684, 
    0.187623611851049, 0.142989530591084, 0.248877868420829, 0.0764081824936653, 
    0.233655852292708, 0.149672589418820, 0.0587718368398146, 0.262647712025980, 
    0.256322619055910, 0.146533463339791, 0.229171565776848, 0.165170241238701, 
    0.156444465196309, 0.0510264285911348, 0.0921321846391164, 0.0604868778843833, 
    0.0970057373862096, 0.180886855038447, 0.0539017190811026, 0.255967961683161, 
    0.270294678374019, 0.269581191387885, 0.270540939483969
])
# rc: radius of grains (all set to 0.04)
rc = np.array([0.04] * len(xc))


# In[5]:


def avoid_grains(x_points, y_points, xc, yc, rc):
    # Fiber Points have been adjusted so that they wont penetrate into the grains
    adjusted_x = np.copy(x_points)
    adjusted_y = np.copy(y_points)
    
    for i in range(len(x_points)):
        # The distance to all grain centers are checked for this case
        distances = np.sqrt((xc - x_points[i])**2 + (yc - y_points[i])**2)
        
        # Find out which grains this point is inside
        penetrating = distances < rc
        
        if np.any(penetrating):
            # For each grain that is being penetrated, push the point to the grain boundary
            for j in np.where(penetrating)[0]:
                dx = x_points[i] - xc[j]
                dy = y_points[i] - yc[j]
                dist = np.sqrt(dx**2 + dy**2)
                # Prevent division by zero
                if dist > 0: 
                    # Calculate the new position on the grain boundary
                    scale_factor = rc[j] / dist
                    new_x = xc[j] + dx * scale_factor
                    new_y = yc[j] + dy * scale_factor
                    
                    # Update the new position
                    adjusted_x[i] = new_x
                    adjusted_y[i] = new_y
                    
    return adjusted_x, adjusted_y


# In[7]:


# Prevent any large jumps if occurred between consecutive time steps
def limit_movement_between_timesteps(x_coords, y_coords, max_movement=0.03):
    for t in range(1, x_coords.shape[0]):
        # Calculates how much each point has moved
        dx = x_coords[t] - x_coords[t-1]
        dy = y_coords[t] - y_coords[t-1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Finds the points that moved too far
        mask = distances > max_movement
        
        if np.any(mask):
            # Reduced the movement to the allowed maximum movement
            scale = max_movement / distances[mask]
            x_coords[t][mask] = x_coords[t-1][mask] + dx[mask] * scale
            y_coords[t][mask] = y_coords[t-1][mask] + dy[mask] * scale
    
    return x_coords, y_coords


# # Smooth Fiber Shape

# In[9]:


# Smoothed out the fiber'S shape while preserving its overall structure
def smooth_fiber_shape(x_coords, y_coords, window=15, alpha=0.9):
    for t in range(x_coords.shape[0]):
        # The Savitzky-Golay filter technique is used to allow for smoothing
        # based on the window_length and polyorder to indicate any bending.
        x_smooth = savgol_filter(x_coords[t], window_length=window, polyorder=2)
        y_smooth = savgol_filter(y_coords[t], window_length=window, polyorder=2) 
        
        # Blend the original data with the smoothed version
        x_coords[t] = (1 - alpha) * x_coords[t] + alpha * x_smooth
        y_coords[t] = (1 - alpha) * y_coords[t] + alpha * y_smooth
        
    return x_coords, y_coords


# # Loading the Data

# In[11]:


# Load training data from CSV files
x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')


# In[13]:


# Obtain all the unique fiber IDs from the first column, any exclusions are given a placeholder of -1
unique_values = [x for x in x_train['Y0'].unique() if x != -1]


# # Model Training and Prediction

# In[ ]:

list1=[]
list2=[]
count=0
while count < 1:
    # Randomly select 1 fibers to predict from the set of data
    x = np.random.choice(unique_values)
    # if the fiber is not a placeholder
    if x != -1:
        # Get data for the selected fiber
        rows = x_train[x_train['Y0'] == x]
        
        # Prepare training data for all fibers except the selected one.
        x_train_filtered = x_train[x_train['Y0'] != x]
        y_train_filtered = y_train.loc[x_train['Y0'] != x]
        
        # Split the data into training and validation sets
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train_filtered, y_train_filtered, 
            test_size=0.8, 
            random_state=42
        )
        
        
        # Setup the Random Forest Method with MultiOutput wrapper
        base_model = RandomForestRegressor(random_state=42)
        multi_output_model = MultiOutputRegressor(base_model)
        
        # Through hypertuning,the best parameters shown below were used to test during grid search
        param_grid = {
            'estimator__n_estimators': [200],
            'estimator__max_depth': [20],
            'estimator__min_samples_leaf': [2],
            'estimator__min_samples_split': [10],
            'estimator__max_features': ['sqrt']
        }

        # Create the grid search
        grid_search = GridSearchCV(
            estimator=multi_output_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(x_tr, y_tr)
        
        # Obtain the best performing model
        print("Best parameters:", grid_search.best_params_)
        model = grid_search.best_estimator_

        # Make predictions for our 3 fibers
        x_pred = rows.copy() 
        y_pred = model.predict(x_pred)
        
        # Reshape the predictions into a 2D array
        y_pred_reshaped = y_pred.reshape(-1, 100)
        
        # Apply smoothing to the predictions
        for i in range(y_pred_reshaped.shape[1]):
            y_pred_reshaped[:, i] = savgol_filter(
                y_pred_reshaped[:, i], 
                window_length=15,
                polyorder=2
            )
        
        # Split that data into x and y coordinates
        x_coords = y_pred_reshaped[:, :50]
        y_coords = y_pred_reshaped[:, 50:]
        
        # Apply smoothing to the spatial coordinates
        x_coords, y_coords = smooth_fiber_shape(x_coords, y_coords, window=15)
        
        # Limit the movement between time steps
        x_coords, y_coords = limit_movement_between_timesteps(x_coords, y_coords, max_movement=0.03)
        
        # Make sure the fibers don't penetrate the grains
        for t in range(x_coords.shape[0]):
            x_coords[t], y_coords[t] = avoid_grains(
                x_coords[t], y_coords[t], xc, yc, rc
            )
        
        # Smooth the fiber shape
        x_coords, y_coords = smooth_fiber_shape(x_coords, y_coords, window=15)
        
        # Store the predictions in the DataFrame
        new_df = pd.DataFrame({
            'x': x_coords.flatten(),
            'y': y_coords.flatten()
        })
        
        # Add the new data for each of the columns given below
        time_values = rows['TIME'].values
        new_df['time'] = np.repeat(time_values, 50)
        new_df['case'] = str(x)
        new_df = new_df[['case', 'time', 'x', 'y']]
        list1.append(new_df)
        
        # Process and store actual values for comparison
        filtered_rows = y_train[x_train['Y0'] == x]
        first_column_values = filtered_rows.iloc[:, :50].values.flatten()
        second_column_values = filtered_rows.iloc[:, 50:].values.flatten()
        
        new_df1 = pd.DataFrame({
            'x': first_column_values,
            'y': second_column_values
        })
        
        new_df1['time'] = np.repeat(time_values, 50)
        new_df1['case'] = str(x)
        new_df1 = new_df1[['case', 'time', 'x', 'y']]
        list2.append(new_df1)
        
        count += 1


# # Results

# In[ ]:


# Combine all results into single DataFrames
new_df = pd.concat(list1, ignore_index=True)
new_df1 = pd.concat(list2, ignore_index=True)

# Save to CSV files
new_df.to_csv('predictedvalues.csv', index=False)
new_df1.to_csv('actualvalues.csv', index=False)


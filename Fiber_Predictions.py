{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "00ce0679-bc52-4012-9e4e-928cdaaa7f6f",
   "metadata": {},
   "source": [
    "# Random Forest Tree Model For Microfiber In Porous Domain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "151d0346-79d9-4e0e-96fc-ef7a841758f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Imported the necessary libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from scipy.signal import savgol_filter\n",
    "from scipy.spatial.distance import cdist\n",
    "from sklearn.neural_network import MLPRegressor\n",
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.preprocessing import StandardScaler, PolynomialFeatures\n",
    "from sklearn.neural_network import MLPRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.multioutput import MultiOutputRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "435a303a-2ff6-4f0f-b664-914a7beef033",
   "metadata": {},
   "source": [
    "# Grain Positioning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "0d9c9f91-4af1-46bf-81eb-1da8ba7b072a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Defined the positions and sizes of circular grains in the domain\n",
    "\n",
    "# xc = x-coordinates of grain centers\n",
    "xc = np.array([\n",
    "    0.861923248843825, 0.530696997759020, 0.230307598040601, 0.756672529035511, \n",
    "    0.355867138671012, 0.130476259199288, 0.220108297265764, 0.512254287950566, \n",
    "    0.853010619375759, 0.624608543201807, 0.399531820716510, 0.673837801757064, \n",
    "    0.423871761032729, 0.444901712442721, 0.0763596893777989, 0.740776643667191, \n",
    "    0.225335687516795, 0.124745007906295, 0.937333120266778, 0.643235524280937, \n",
    "    0.315817148501495, 0.948789100265014, 0.845062624129590, 0.766136269897956, \n",
    "    0.312410891980905, 0.937354884834047, 0.584419981171847\n",
    "])\n",
    "# yc = y coordinates of grain centers\n",
    "yc = np.array([\n",
    "    0.142796932100205, 0.198844509849833, 0.0607917660763774, 0.0745644494427684, \n",
    "    0.187623611851049, 0.142989530591084, 0.248877868420829, 0.0764081824936653, \n",
    "    0.233655852292708, 0.149672589418820, 0.0587718368398146, 0.262647712025980, \n",
    "    0.256322619055910, 0.146533463339791, 0.229171565776848, 0.165170241238701, \n",
    "    0.156444465196309, 0.0510264285911348, 0.0921321846391164, 0.0604868778843833, \n",
    "    0.0970057373862096, 0.180886855038447, 0.0539017190811026, 0.255967961683161, \n",
    "    0.270294678374019, 0.269581191387885, 0.270540939483969\n",
    "])\n",
    "# rc: radius of grains (all set to 0.04)\n",
    "rc = np.array([0.04] * len(xc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "7d5389db-95b1-42c3-bfeb-9e13d98cfc94",
   "metadata": {},
   "outputs": [],
   "source": [
    "def avoid_grains(x_points, y_points, xc, yc, rc):\n",
    "    # Fiber Points have been adjusted so that they wont penetrate into the grains\n",
    "    adjusted_x = np.copy(x_points)\n",
    "    adjusted_y = np.copy(y_points)\n",
    "    \n",
    "    for i in range(len(x_points)):\n",
    "        # The distance to all grain centers are checked for this case\n",
    "        distances = np.sqrt((xc - x_points[i])**2 + (yc - y_points[i])**2)\n",
    "        \n",
    "        # Find out which grains this point is inside\n",
    "        penetrating = distances < rc\n",
    "        \n",
    "        if np.any(penetrating):\n",
    "            # For each grain that is being penetrated, push the point to the grain boundary\n",
    "            for j in np.where(penetrating)[0]:\n",
    "                dx = x_points[i] - xc[j]\n",
    "                dy = y_points[i] - yc[j]\n",
    "                dist = np.sqrt(dx**2 + dy**2)\n",
    "                # Prevent division by zero\n",
    "                if dist > 0: \n",
    "                    # Calculate the new position on the grain boundary\n",
    "                    scale_factor = rc[j] / dist\n",
    "                    new_x = xc[j] + dx * scale_factor\n",
    "                    new_y = yc[j] + dy * scale_factor\n",
    "                    \n",
    "                    # Update the new position\n",
    "                    adjusted_x[i] = new_x\n",
    "                    adjusted_y[i] = new_y\n",
    "                    \n",
    "    return adjusted_x, adjusted_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "54d1e4cd-1121-4aa7-ae35-50a23cfc25f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prevent any large jumps if occurred between consecutive time steps\n",
    "def limit_movement_between_timesteps(x_coords, y_coords, max_movement=0.03):\n",
    "    for t in range(1, x_coords.shape[0]):\n",
    "        # Calculates how much each point has moved\n",
    "        dx = x_coords[t] - x_coords[t-1]\n",
    "        dy = y_coords[t] - y_coords[t-1]\n",
    "        distances = np.sqrt(dx**2 + dy**2)\n",
    "        \n",
    "        # Finds the points that moved too far\n",
    "        mask = distances > max_movement\n",
    "        \n",
    "        if np.any(mask):\n",
    "            # Reduced the movement to the allowed maximum movement\n",
    "            scale = max_movement / distances[mask]\n",
    "            x_coords[t][mask] = x_coords[t-1][mask] + dx[mask] * scale\n",
    "            y_coords[t][mask] = y_coords[t-1][mask] + dy[mask] * scale\n",
    "    \n",
    "    return x_coords, y_coords"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7d4c0484-a0a2-45d8-b355-87b31cc06298",
   "metadata": {},
   "source": [
    "# Smooth Fiber Shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "01e2613b-ca16-4119-9035-09dd9b54aab8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Smoothed out the fiber'S shape while preserving its overall structure\n",
    "def smooth_fiber_shape(x_coords, y_coords, window=15, alpha=0.9):\n",
    "    for t in range(x_coords.shape[0]):\n",
    "        # The Savitzky-Golay filter technique is used to allow for smoothing\n",
    "        # based on the window_length and polyorder to indicate any bending.\n",
    "        x_smooth = savgol_filter(x_coords[t], window_length=window, polyorder=2)\n",
    "        y_smooth = savgol_filter(y_coords[t], window_length=window, polyorder=2) \n",
    "        \n",
    "        # Blend the original data with the smoothed version\n",
    "        x_coords[t] = (1 - alpha) * x_coords[t] + alpha * x_smooth\n",
    "        y_coords[t] = (1 - alpha) * y_coords[t] + alpha * y_smooth\n",
    "        \n",
    "    return x_coords, y_coords\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20022657-8ebf-4dec-b1f7-39d8524c493a",
   "metadata": {},
   "source": [
    "# Loading the Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "eb77aed5-371d-49b9-ab87-fe1fa1d1d3bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load training data from CSV files\n",
    "x_train = pd.read_csv('x_train.csv')\n",
    "y_train = pd.read_csv('y_train.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "59600455-9306-496f-abcf-7b1b79360c38",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Obtain all the unique fiber IDs from the first column, any exclusions are given a placeholder of -1\n",
    "unique_values = [x for x in x_train['Y0'].unique() if x != -1]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dd2aaa90-1bca-46dd-a597-7b0f502f4c5f",
   "metadata": {},
   "source": [
    "# Model Training and Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c270efc6-4e4f-4888-bc9f-a63c98e86eaa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 1 candidates, totalling 3 fits\n"
     ]
    }
   ],
   "source": [
    "count=0\n",
    "while count < 1:\n",
    "    # Randomly select 1 fibers to predict from the set of data\n",
    "    x = np.random.choice(unique_values)\n",
    "    # if the fiber is not a placeholder\n",
    "    if x != -1:\n",
    "        # Get data for the selected fiber\n",
    "        rows = x_train[x_train['Y0'] == x]\n",
    "        \n",
    "        # Prepare training data for all fibers except the selected one.\n",
    "        x_train_filtered = x_train[x_train['Y0'] != x]\n",
    "        y_train_filtered = y_train.loc[x_train['Y0'] != x]\n",
    "        \n",
    "        # Split the data into training and validation sets\n",
    "        x_tr, x_val, y_tr, y_val = train_test_split(\n",
    "            x_train_filtered, y_train_filtered, \n",
    "            test_size=0.8, \n",
    "            random_state=42\n",
    "        )\n",
    "        \n",
    "        \n",
    "        # Setup the Random Forest Method with MultiOutput wrapper\n",
    "        base_model = RandomForestRegressor(random_state=42)\n",
    "        multi_output_model = MultiOutputRegressor(base_model)\n",
    "        \n",
    "        # Through hypertuning,the best parameters shown below were used to test during grid search\n",
    "        param_grid = {\n",
    "            'estimator__n_estimators': [200],\n",
    "            'estimator__max_depth': [20],\n",
    "            'estimator__min_samples_leaf': [2],\n",
    "            'estimator__min_samples_split': [10],\n",
    "            'estimator__max_features': ['sqrt']\n",
    "        }\n",
    "\n",
    "        # Create the grid search\n",
    "        grid_search = GridSearchCV(\n",
    "            estimator=multi_output_model,\n",
    "            param_grid=param_grid,\n",
    "            cv=3,\n",
    "            scoring='neg_mean_squared_error',\n",
    "            verbose=2,\n",
    "            n_jobs=-1\n",
    "        )\n",
    "        \n",
    "        # Train the model\n",
    "        grid_search.fit(x_tr, y_tr)\n",
    "        \n",
    "        # Obtain the best performing model\n",
    "        print(\"Best parameters:\", grid_search.best_params_)\n",
    "        model = grid_search.best_estimator_\n",
    "\n",
    "        # Make predictions for our 3 fibers\n",
    "        x_pred = rows.copy() \n",
    "        y_pred = model.predict(x_pred)\n",
    "        \n",
    "        # Reshape the predictions into a 2D array\n",
    "        y_pred_reshaped = y_pred.reshape(-1, 100)\n",
    "        \n",
    "        # Apply smoothing to the predictions\n",
    "        for i in range(y_pred_reshaped.shape[1]):\n",
    "            y_pred_reshaped[:, i] = savgol_filter(\n",
    "                y_pred_reshaped[:, i], \n",
    "                window_length=15,\n",
    "                polyorder=2\n",
    "            )\n",
    "        \n",
    "        # Split that data into x and y coordinates\n",
    "        x_coords = y_pred_reshaped[:, :50]\n",
    "        y_coords = y_pred_reshaped[:, 50:]\n",
    "        \n",
    "        # Apply smoothing to the spatial coordinates\n",
    "        x_coords, y_coords = smooth_fiber_shape(x_coords, y_coords, window=15)\n",
    "        \n",
    "        # Limit the movement between time steps\n",
    "        x_coords, y_coords = limit_movement_between_timesteps(x_coords, y_coords, max_movement=0.03)\n",
    "        \n",
    "        # Make sure the fibers don't penetrate the grains\n",
    "        for t in range(x_coords.shape[0]):\n",
    "            x_coords[t], y_coords[t] = avoid_grains(\n",
    "                x_coords[t], y_coords[t], xc, yc, rc\n",
    "            )\n",
    "        \n",
    "        # Smooth the fiber shape\n",
    "        x_coords, y_coords = smooth_fiber_shape(x_coords, y_coords, window=15)\n",
    "        \n",
    "        # Store the predictions in the DataFrame\n",
    "        new_df = pd.DataFrame({\n",
    "            'x': x_coords.flatten(),\n",
    "            'y': y_coords.flatten()\n",
    "        })\n",
    "        \n",
    "        # Add the new data for each of the columns given below\n",
    "        time_values = rows['TIME'].values\n",
    "        new_df['time'] = np.repeat(time_values, 50)\n",
    "        new_df['case'] = str(x)\n",
    "        new_df = new_df[['case', 'time', 'x', 'y']]\n",
    "        list1.append(new_df)\n",
    "        \n",
    "        # Process and store actual values for comparison\n",
    "        filtered_rows = y_train[x_train['Y0'] == x]\n",
    "        first_column_values = filtered_rows.iloc[:, :50].values.flatten()\n",
    "        second_column_values = filtered_rows.iloc[:, 50:].values.flatten()\n",
    "        \n",
    "        new_df1 = pd.DataFrame({\n",
    "            'x': first_column_values,\n",
    "            'y': second_column_values\n",
    "        })\n",
    "        \n",
    "        new_df1['time'] = np.repeat(time_values, 50)\n",
    "        new_df1['case'] = str(x)\n",
    "        new_df1 = new_df1[['case', 'time', 'x', 'y']]\n",
    "        list2.append(new_df1)\n",
    "        \n",
    "        count += 1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7bb61e4-0427-4fd9-9557-a10de431d6cd",
   "metadata": {},
   "source": [
    "# Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd2f541a-bd29-4b9b-b2e6-70f1d7914aa7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Combine all results into single DataFrames\n",
    "new_df = pd.concat(list1, ignore_index=True)\n",
    "new_df1 = pd.concat(list2, ignore_index=True)\n",
    "\n",
    "# Save to CSV files\n",
    "new_df.to_csv('predictedvalues.csv', index=False)\n",
    "new_df1.to_csv('actualvalues.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

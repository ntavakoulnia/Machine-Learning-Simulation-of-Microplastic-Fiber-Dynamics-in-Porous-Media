{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4eb2512e-0a87-4966-a36e-e55502bde9a0",
   "metadata": {},
   "source": [
    "# Simulation Animation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "7bc8e901-15df-4ddb-9e64-b2795f5f5ab5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Imported visualization and data processing libraries\n",
    "# Mayavi for 3D visualization\n",
    "from mayavi import mlab\n",
    "import numpy as np  \n",
    "import pandas as pd  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2bc96521-cc88-4055-a4d3-b0b82321a99a",
   "metadata": {},
   "source": [
    "# Grain Definition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "26df2350-164b-4d17-950d-d8b8172d11fa",
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
   "cell_type": "markdown",
   "id": "dba0b9a8-29bc-4ca4-aa23-5fb90f6f1313",
   "metadata": {},
   "source": [
    "# Data Loading Section"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "a0cb96c7-10fd-4549-956b-86506587fc5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loaded the fiber data both actual and predicted paths\n",
    "# Actual fiber paths\n",
    "fiber_data1 = pd.read_csv(r'E:\\machine learning\\actualvalues.csv')  \n",
    "# Predicted paths\n",
    "fiber_data2 = pd.read_csv(r'E:\\machine learning\\predictedvalues.csv')  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "a30481da-f542-4ee5-bff4-c8a5d44b011d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# List of unique fiber cases to visualize\n",
    "unique_cases = sorted(fiber_data1['case'].unique())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99efe2f5-fa8f-4f81-9cd6-cbc28c2e303a",
   "metadata": {},
   "source": [
    "# Visualization Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "ad7b243b-e672-4157-8d3b-b168d391319d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def render_grains():\n",
    "    \"\"\"Created 3D sphere objects for each grain in the scene\"\"\"\n",
    "    # Create spherical mesh coordinates\n",
    "    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]\n",
    "    \n",
    "    for i in range(len(xc)):\n",
    "        # Positioned each sphere at grain center\n",
    "        x = rc[i] * np.cos(u) * np.sin(v) + xc[i]\n",
    "        y = rc[i] * np.sin(u) * np.sin(v) + yc[i]\n",
    "        z = rc[i] * np.cos(v)\n",
    "        \n",
    "        # Created the grain with orange color\n",
    "        grain = mlab.mesh(x, y, z, color=(1, 0.5, 0))\n",
    "        grain.actor.property.lighting = False  \n",
    "        grain.actor.property.shading = False\n",
    "        grain.actor.property.opacity = 0.7  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "1be9eb80-257b-445c-80b7-8c519fc68db6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_river_flow():\n",
    "    \"\"\"Generate a wavy surface to represent water flow\"\"\"\n",
    "    x, y = np.mgrid[-0.1:1.1:200j, -0.1:0.4:100j]\n",
    "    \n",
    "    # Created a wave pattern using sine functions\n",
    "    z = (0.003 * np.sin(x * 8 + y * 6) + \n",
    "         0.002 * np.sin(x * 12 - y * 8) +\n",
    "         0.001 * np.sin(x * 15 + y * 10)) - 0.05\n",
    "    \n",
    "    # Created a blue water surface\n",
    "    water = mlab.surf(x, y, z, \n",
    "                     color=(0.1, 0.4, 0.8),\n",
    "                     opacity=0.3,\n",
    "                     representation='surface')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "aba50cfa-84e9-4e8a-a9b9-01448d17fd78",
   "metadata": {},
   "outputs": [],
   "source": [
    "def render_fibers(frame_data, is_predicted=False):\n",
    "    \"\"\"Drawed a single fiber at a specific timestep\"\"\"\n",
    "    x, y = frame_data[\"x\"].values, frame_data[\"y\"].values\n",
    "    # All fibers lie flat in z-plane\n",
    "    z = np.zeros_like(x) \n",
    "    \n",
    "    # Color coding = red for predicted, blue for actual\n",
    "    color = (1, 0, 0) if is_predicted else (0, 0, 1)\n",
    "    \n",
    "    # Drawed as thin tubes\n",
    "    return mlab.plot3d(x, y, z, color=color, tube_radius=0.001, opacity=0.8)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "969fae26-aed7-4bd7-8e65-d5c73402a928",
   "metadata": {},
   "source": [
    "# Animation Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "f5fa3262-0da9-4979-a569-9190e672c21c",
   "metadata": {},
   "outputs": [],
   "source": [
    "@mlab.animate(delay=10)\n",
    "def animate_fibers():\n",
    "    \"\"\"Animate the fibers through all timesteps\"\"\"\n",
    "    # Tracked the fiber objects to remove them\n",
    "    dynamic_fiber_objects = [] \n",
    "    \n",
    "    for timestep in unique_timesteps:\n",
    "        # Cleared the previous frame's fibers\n",
    "        for obj in dynamic_fiber_objects:\n",
    "            obj.remove()\n",
    "        dynamic_fiber_objects.clear()\n",
    "        \n",
    "        # Drawn actual fibers (blue)\n",
    "        for case in unique_cases:\n",
    "            frame_data1 = fiber_data1[\n",
    "                (fiber_data1[\"case\"] == case) & \n",
    "                (fiber_data1[\"time\"] == timestep)\n",
    "            ].head(50) \n",
    "            \n",
    "            if not frame_data1.empty:\n",
    "                fiber = render_fibers(frame_data1, is_predicted=False)\n",
    "                dynamic_fiber_objects.append(fiber)\n",
    "        \n",
    "        # Drew predicted fibers (red)\n",
    "        for case in unique_cases:\n",
    "            frame_data2 = fiber_data2[\n",
    "                (fiber_data2[\"case\"] == case) & \n",
    "                (fiber_data2[\"time\"] == timestep)\n",
    "            ].head(50)\n",
    "            \n",
    "            if not frame_data2.empty:\n",
    "                fiber = render_fibers(frame_data2, is_predicted=True)\n",
    "                dynamic_fiber_objects.append(fiber)\n",
    "        \n",
    "        # Setting camera view (rotating around the scene)\n",
    "        mlab.view(azimuth=0, elevation=360, distance=1.05, focalpoint=(0.51, 0.161, 0))\n",
    "        # Pause until next frame using yield\n",
    "        yield  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "930add82-33f8-4a3c-a730-a7b8f3aa6910",
   "metadata": {},
   "source": [
    "# Scene Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "03002624-4abb-4ea0-a6f8-f025bc8635cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Warning: the range of your scalar values differs by more than a factor 100 than the range of the grid values and you did not specify a warp_scale. You could try warp_scale=\"auto\".\n"
     ]
    }
   ],
   "source": [
    "# Created a figure with white background and large size\n",
    "fig = mlab.figure(bgcolor=(1, 1, 1), size=(2560,1440))\n",
    "\n",
    "# Added background elements\n",
    "# Water surface\n",
    "create_river_flow()  \n",
    "# Stationary grains\n",
    "render_grains()     \n",
    "\n",
    "# Obtain all unique timesteps from both datasets\n",
    "unique_timesteps = sorted(set(fiber_data1[\"time\"].unique()) | set(fiber_data2[\"time\"].unique()))\n",
    "\n",
    "# Set the initial camera position\n",
    "mlab.view(azimuth=0, elevation=360, distance=0.5, focalpoint=(0.51, 0.161, 0))\n",
    "\n",
    "# Disabled the mouse interaction \n",
    "mlab.gcf().scene.interactor.interactor_style = None\n",
    "\n",
    "# How to Start the animation\n",
    "animation = animate_fibers()\n",
    "# Display the visualization\n",
    "mlab.show()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f75dc89d-aa2f-4b3a-8941-de08e37d858d",
   "metadata": {},
   "outputs": [],
   "source": []
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

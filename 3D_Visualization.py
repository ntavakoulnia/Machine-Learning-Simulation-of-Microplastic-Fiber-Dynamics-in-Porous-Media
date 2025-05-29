#!/usr/bin/env python
# coding: utf-8

# # Simulation Animation

# In[44]:


# Imported visualization and data processing libraries
# Mayavi for 3D visualization
from mayavi import mlab
import numpy as np  
import pandas as pd  


# # Grain Definition

# In[47]:


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


# # Data Loading Section

# In[50]:


# Loaded the fiber data both actual and predicted paths
# Actual fiber paths
fiber_data1 = pd.read_csv(r'E:\machine learning\actualvalues.csv')  
# Predicted paths
fiber_data2 = pd.read_csv(r'E:\machine learning\predictedvalues.csv')  


# In[52]:


# List of unique fiber cases to visualize
unique_cases = sorted(fiber_data1['case'].unique())


# # Visualization Functions

# In[55]:


def render_grains():
    """Created 3D sphere objects for each grain in the scene"""
    # Create spherical mesh coordinates
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    
    for i in range(len(xc)):
        # Positioned each sphere at grain center
        x = rc[i] * np.cos(u) * np.sin(v) + xc[i]
        y = rc[i] * np.sin(u) * np.sin(v) + yc[i]
        z = rc[i] * np.cos(v)
        
        # Created the grain with orange color
        grain = mlab.mesh(x, y, z, color=(1, 0.5, 0))
        grain.actor.property.lighting = False  
        grain.actor.property.shading = False
        grain.actor.property.opacity = 0.7  


# In[57]:


def create_river_flow():
    """Generate a wavy surface to represent water flow"""
    x, y = np.mgrid[-0.1:1.1:200j, -0.1:0.4:100j]
    
    # Created a wave pattern using sine functions
    z = (0.003 * np.sin(x * 8 + y * 6) + 
         0.002 * np.sin(x * 12 - y * 8) +
         0.001 * np.sin(x * 15 + y * 10)) - 0.05
    
    # Created a blue water surface
    water = mlab.surf(x, y, z, 
                     color=(0.1, 0.4, 0.8),
                     opacity=0.3,
                     representation='surface')


# In[59]:


def render_fibers(frame_data, is_predicted=False):
    """Drawed a single fiber at a specific timestep"""
    x, y = frame_data["x"].values, frame_data["y"].values
    # All fibers lie flat in z-plane
    z = np.zeros_like(x) 
    
    # Color coding = red for predicted, blue for actual
    color = (1, 0, 0) if is_predicted else (0, 0, 1)
    
    # Drawed as thin tubes
    return mlab.plot3d(x, y, z, color=color, tube_radius=0.001, opacity=0.8)


# # Animation Function

# In[67]:


@mlab.animate(delay=10)
def animate_fibers():
    """Animate the fibers through all timesteps"""
    # Tracked the fiber objects to remove them
    dynamic_fiber_objects = [] 
    
    for timestep in unique_timesteps:
        # Cleared the previous frame's fibers
        for obj in dynamic_fiber_objects:
            obj.remove()
        dynamic_fiber_objects.clear()
        
        # Drawn actual fibers (blue)
        for case in unique_cases:
            frame_data1 = fiber_data1[
                (fiber_data1["case"] == case) & 
                (fiber_data1["time"] == timestep)
            ].head(50) 
            
            if not frame_data1.empty:
                fiber = render_fibers(frame_data1, is_predicted=False)
                dynamic_fiber_objects.append(fiber)
        
        # Drew predicted fibers (red)
        for case in unique_cases:
            frame_data2 = fiber_data2[
                (fiber_data2["case"] == case) & 
                (fiber_data2["time"] == timestep)
            ].head(50)
            
            if not frame_data2.empty:
                fiber = render_fibers(frame_data2, is_predicted=True)
                dynamic_fiber_objects.append(fiber)
        
        # Setting camera view (rotating around the scene)
        mlab.view(azimuth=0, elevation=360, distance=1.05, focalpoint=(0.51, 0.161, 0))
        # Pause until next frame using yield
        yield  


# # Scene Setup

# In[70]:


# Created a figure with white background and large size
fig = mlab.figure(bgcolor=(1, 1, 1), size=(2560,1440))

# Added background elements
# Water surface
create_river_flow()  
# Stationary grains
render_grains()     

# Obtain all unique timesteps from both datasets
unique_timesteps = sorted(set(fiber_data1["time"].unique()) | set(fiber_data2["time"].unique()))

# Set the initial camera position
mlab.view(azimuth=0, elevation=360, distance=0.5, focalpoint=(0.51, 0.161, 0))

# Disabled the mouse interaction 
mlab.gcf().scene.interactor.interactor_style = None

# How to Start the animation
animation = animate_fibers()
# Display the visualization
mlab.show()  


# In[ ]:





# DataMiningProject


--Load the given two csv files (House_Rent_Dataset.csv and New dataset.xlsx) into Jupyter Notebook files and they should same directory as jupyter notebook (.ipynb file)

--Make sure to install all these python libraries if your python version does not have any of these

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor  
import webbrowser
import os




-- upgrade pandas numpy in needed
!pip install --upgrade pandas numpy




-- Run each each step one by one (AS step1 result will be used in step2 etc)






--Notebook has headings for steps and last code part is for testing model accuracy using other dataset from top3 models of first dataset



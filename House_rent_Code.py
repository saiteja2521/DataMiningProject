#!/usr/bin/env python
# coding: utf-8

# In[7]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[ ]:





# In[ ]:





# In[2]:


get_ipython().system('pip install --upgrade pandas numpy')


# # loading the dataset 

# In[8]:



rentdata=pd.read_csv("House_Rent_Dataset.csv")
rentdata.tail()


# # structure of the data

# In[9]:



rentdata.shape


# # checking datatypes

# In[10]:



rentdata=pd.read_csv("House_Rent_Dataset.csv")
rentdata.dtypes


# # converting variables to integers

# In[11]:



rentdata=pd.read_csv("House_Rent_Dataset.csv")
rentdata[['BHK','Rent','Size','Bathroom']] = rentdata[['BHK','Rent','Size','Bathroom']].astype(int)


# # checking missing values

# In[12]:



rentdata.isnull().sum()


# # checking duplicated values

# In[13]:



rentdata.duplicated().sum()


# # handling outliers using box plot

# In[14]:



plt.figure(figsize=(18, 6))
plt.scatter(rentdata['Rent'], rentdata['Size'], c='green', marker='x', label='Outliers')
plt.title("Scatter Plot with Outliers Highlighted")
plt.xlabel("Rent")
plt.ylabel("Size")
plt.legend()
plt.show()


# # Exploratory Data Analysis

# In[15]:


plt.figure(figsize=(10, 6))
ax = sns.countplot(x="BHK", data=rentdata)
plt.title("Count Plot of BHK")
plt.xlabel("BHK")
plt.ylabel("Count")

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

plt.show()





# In[16]:


rentdata.columns


# In[17]:


rentdata["Area Locality"].value_counts()


# In[18]:



import pandas as pd
import numpy as np
rentdata.describe().T


# In[19]:



rentdata.corr()


# In[20]:


rentdata["Area Type"].value_counts()


# 

# In[21]:


plt.figure(figsize = (20, 7))
sns.barplot(x = rentdata["City"], y = rentdata["Rent"], palette = "nipy_spectral")


# In[22]:



sns.countplot("Area Type",data=rentdata)


# In[23]:



sns.countplot("Furnishing Status",data=rentdata)


# In[ ]:





# Visulaizing numerical variables

# In[24]:



rentdata['BHK'].hist(bins=12)
plt.suptitle('BHK distribution of rented house')
plt.xlabel('BHK')
plt.ylabel('Count')
plt.show()


# In[25]:



rentdata['Bathroom'].hist(bins=25)
plt.suptitle('Bathroom distribution of rented house')
plt.xlabel('Bathroom')
plt.ylabel('Count')
plt.show()


# encoding

# In[26]:



rentdata2 = rentdata.join(pd.get_dummies(rentdata[['Area Type','City','Furnishing Status','Tenant Preferred','Point of Contact']], drop_first=True))
rentdata2.drop(columns = ['Area Type','City','Furnishing Status','Tenant Preferred','Point of Contact'], inplace = True)
rentdata.head()


# In[27]:


plt.figure(figsize=(15,15))
sns.heatmap(rentdata2.corr(),annot=True,fmt='.2f')
plt.show()


# # Model Training

# In[28]:


from sklearn.preprocessing import LabelEncoder
X1 = rentdata.drop(['Rent'],axis=1)
y = rentdata['Rent']


# In[29]:


X2 = X1.select_dtypes('O')

for col in X2.columns:
    lb = LabelEncoder()
    X1[col] = lb.fit_transform(X2[col].values)
X1


# In[ ]:





# Data Splitting

# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X1, y)


# In[31]:



random_forest_model=RandomForestRegressor()
random_forest_model.fit(X_train,y_train)
print("Random Forest Training Accuracy:", random_forest_model.score(X_train,y_train))
print("Random Forest Testing Accuracy:", random_forest_model.score(X_test,y_test))


# In[32]:



gb_model=GradientBoostingRegressor()
gb_model.fit(X_train,y_train)
print("Gradient Boost Training Accuracy:", gb_model.score(X_train,y_train))
print("Gradient Boost Testing Accuracy:", gb_model.score(X_test,y_test))


# In[33]:


from sklearn.linear_model import Lasso, LassoCV
lasso_cv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lasso_cv.fit(X_train, y_train)


# In[34]:


alpha = lasso_cv.alpha_
alpha
lasso = Lasso(alpha = lasso_cv.alpha_)
lasso.fit(X_train, y_train)


# In[35]:


print("Lasso Regresion Training Accuracy:", lasso.score(X_train, y_train))
print("Lasso REgression Testing Accuracy:", lasso.score(X_test,y_test))


# Model Evalutaion

# In[36]:


def evaluate(model):
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    
    print('MAE:', mean_absolute_error(y_test, pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))
    print('R2 Score:', r2_score(y_test, pred))


# In[37]:


evaluate(RandomForestRegressor())


# In[38]:


evaluate(GradientBoostingRegressor())


# In[39]:


lasso.score(X_test, y_test)


# In[40]:


def adj_r2(X, y, model):
    r2 = model.score(X, y)
    n = X.shape[0]
    p = X.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return adjusted_r2


# In[41]:


print(adj_r2(X_train, y_train, lasso))


# In[42]:


print(adj_r2(X_test, y_test, lasso))


# In[43]:


import matplotlib.pyplot as plt

models = [random_forest_model, gb_model, lasso]
model_names = ['Random Forest', 'Gradient Boost', 'Lasso']

train_scores = []
test_scores = []

for model in models:
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure(figsize=(12, 7))

barWidth = 0.35
r1 = range(len(train_scores))
r2 = [x + barWidth for x in r1]

bars1 = plt.bar(r1, train_scores, width=barWidth, color='blue', edgecolor='grey', label='Training R-squared', alpha=0.7)
bars2 = plt.bar(r2, test_scores, width=barWidth, color='cyan', edgecolor='grey', label='Testing R-squared', alpha=0.7)

plt.xlabel('Models', fontweight='bold')
plt.ylabel('R-squared Score', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(train_scores))], model_names)
plt.title('Model Comparison', fontweight='bold')

def label_bars(bars):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

label_bars(bars1)
label_bars(bars2)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()


# In[44]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_plot(model, model_name):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    return mae, rmse, r2

rf_mae, rf_rmse, rf_r2 = evaluate_and_plot(RandomForestRegressor(), 'RandomForestRegressor')

gb_mae, gb_rmse, gb_r2 = evaluate_and_plot(GradientBoostingRegressor(), 'GradientBoostingRegressor')

metrics = ['MAE', 'RMSE', 'R2 Score']
rf_scores = [rf_mae, rf_rmse, rf_r2]
gb_scores = [gb_mae, gb_rmse, gb_r2]

bar_width = 0.35
index = np.arange(len(metrics))

plt.figure(figsize=(10, 6))
bar1 = plt.bar(index - bar_width/2, rf_scores, bar_width, label='RandomForestRegressor', alpha=0.7)
bar2 = plt.bar(index + bar_width/2, gb_scores, bar_width, label='GradientBoostingRegressor', alpha=0.7)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Comparison - MAE, RMSE, and R2 Score')
plt.xticks(index, metrics)
plt.legend()

def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

annotate_bars(bar1)
annotate_bars(bar2)

plt.show()


# In[45]:


from sklearn.neural_network import MLPRegressor

nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)


nn_train_score = nn_model.score(X_train, y_train)
print(f"Neural Network Training Performance: {nn_train_score}")
nn_test_score = nn_model.score(X_test, y_test)
print(f"Neural Network Testing Performance: {nn_test_score}")


# In[46]:


from sklearn.svm import SVR
from sklearn.metrics import r2_score

svr_model = SVR()
svr_model.fit(X_train, y_train)

svr_train_score = svr_model.score(X_train, y_train)
print(f"SVR Training Performance: {svr_train_score}")

svr_predictions = svr_model.predict(X_test)

svr_test_score = r2_score(y_test, svr_predictions)
print(f"SVR Testing Performance: {svr_test_score}")



# In[ ]:





# In[47]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5) 

knn.fit(X_train, y_train)

print("KNN Training Accuracy:", knn.score(X_train, y_train))
print("KNN Testing Accuracy:", knn.score(X_test, y_test))


# In[48]:




from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()


gnb.fit(X_train, y_train)


print("Gaussian Naive Bayes Training Accuracy:", gnb.score(X_train, y_train))
print("Gaussian Naive Bayes Testing Accuracy:", gnb.score(X_test, y_test))


# In[ ]:





# In[49]:


import matplotlib.pyplot as plt
import seaborn as sns


models = ['SVR', 'Neural Network', 'Lasso Regression', 'Gradient Boost', 'Random Forest']
training_scores = [svr_train_score, nn_train_score, lasso.score(X_train, y_train), 
                   gb_model.score(X_train, y_train), random_forest_model.score(X_train, y_train)]

testing_scores = [svr_test_score, nn_test_score, lasso.score(X_test, y_test), 
                  gb_model.score(X_test, y_test), random_forest_model.score(X_test, y_test)]

from sklearn.neighbors import KNeighborsRegressor


knn = KNeighborsRegressor(n_neighbors=5) 


knn.fit(X_train, y_train)


models.append('KNN')
training_scores.append(knn.score(X_train, y_train))
testing_scores.append(knn.score(X_test, y_test))

from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()


gnb.fit(X_train, y_train)


models.append('Gaussian Naive Bayes')
training_scores.append(gnb.score(X_train, y_train))
testing_scores.append(gnb.score(X_test, y_test))


plt.figure(figsize=(18, 7))
barWidth = 0.35
r1 = range(len(training_scores))
r2 = [x + barWidth for x in r1]

bars1 = plt.bar(r1, training_scores, width=barWidth, color='blue', edgecolor='grey', label='Training Score')
bars2 = plt.bar(r2, testing_scores, width=barWidth, color='cyan', edgecolor='grey', label='Testing Score')


def label_bars(bars):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')


label_bars(bars1)
label_bars(bars2)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.title('Comparison of Model Performances', fontweight='bold')
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Scores', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(training_scores))], models)

plt.legend()
plt.tight_layout()
plt.show()



# In[ ]:





# In[50]:


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor  

def evaluate(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    print(model.__class__.__name__)
    print('MAE:', mean_absolute_error(y_test, pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))
    print('-------------------------')


svr_model = SVR()
nn_model = MLPRegressor()  
rf_model = RandomForestRegressor()
lasso_model = Lasso()
gb_model = GradientBoostingRegressor()
knn_model = KNeighborsRegressor()
gnb_model = GaussianNB()


models = [svr_model, nn_model, rf_model, lasso_model, gb_model, knn_model, gnb_model]

for model in models:
    evaluate(model)


# In[77]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB

def evaluate(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    
    return mae, rmse

svr_model = SVR()
nn_model = MLPRegressor()
rf_model = RandomForestRegressor()
lasso_model = Lasso()
gb_model = GradientBoostingRegressor()
knn_model = KNeighborsRegressor()

models = [svr_model, nn_model, rf_model, lasso_model, gb_model, knn_model]

model_names = [model.__class__.__name__ if model.__class__.__name__ != 'MLPRegressor' else 'Neural Networks' for model in models]

maes = []
rmses = []

for model in models:
    if model.__class__.__name__ == 'GaussianNB':
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
    else:
        mae, rmse = evaluate(model)
    maes.append(mae)
    rmses.append(rmse)


barWidth = 0.35
r1 = range(len(maes))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(20, 7))

plt.bar(r1, maes, width=barWidth, color='blue', edgecolor='grey', label='MAE')
plt.bar(r2, rmses, width=barWidth, color='cyan', edgecolor='grey', label='RMSE')

plt.title('Comparison of Model Evaluation Metrics', fontweight='bold')
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Value', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(maes))], model_names)


for i in range(len(r1)):
    plt.text(r1[i], maes[i] + 0.01, f"{maes[i]:.2f}", ha='center')
    plt.text(r2[i], rmses[i] + 0.01, f"{rmses[i]:.2f}", ha='center')

plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


import pandas as pd
import ipywidgets as widgets
from IPython.display import display


file_path = "House_Rent_Dataset.csv"
df = pd.read_csv(file_path)


city_dropdown = widgets.Dropdown(
    options=df['City'].unique(),
    description="Select City:",
    disabled=False
)


output = widgets.Output()


def calculate_average_rent(change):
    selected_city = city_dropdown.value
    average_rent = df[df['City'] == selected_city]['Rent'].mean()
    
    with output:
        output.clear_output()
        print(f"Predicted Average Rent for {selected_city}: {average_rent:.2f}")

city_dropdown.observe(calculate_average_rent, names='value')

display(city_dropdown, output)


# In[ ]:





# In[ ]:





# In[24]:


import pandas as pd
import ipywidgets as widgets
from IPython.display import display


file_path = "House_Rent_Dataset.csv"
df = pd.read_csv(file_path)


city_dropdown = widgets.Dropdown(
    options=df['City'].unique(),
    description="Select City:",
    disabled=False
)

bhk_dropdown = widgets.Dropdown(
    options=df['BHK'].unique(),
    description="Select BHK:",
    disabled=False
)

bathroom_dropdown = widgets.Dropdown(
    options=df['Bathroom'].unique(),
    description="Select Bathroom:",
    disabled=False
)


output = widgets.Output()


def calculate_average_rent(change):
    selected_city = city_dropdown.value
    selected_bhk = bhk_dropdown.value
    selected_bathroom = bathroom_dropdown.value
    
    filtered_df = df[
        (df['City'] == selected_city) &
        (df['BHK'] == selected_bhk) &
        (df['Bathroom'] == selected_bathroom)
    ]
    
    if len(filtered_df) == 0:
        with output:
            output.clear_output()
            print("No matching data found.")
        return
    
    average_rent = filtered_df['Rent'].mean()
    
    with output:
        output.clear_output()
        print(f"Predicted Average Rent for {selected_city}, {selected_bhk} BHK, {selected_bathroom} Bathroom: {average_rent:.2f}")

city_dropdown.observe(calculate_average_rent, names='value')
bhk_dropdown.observe(calculate_average_rent, names='value')
bathroom_dropdown.observe(calculate_average_rent, names='value')

display(city_dropdown, bhk_dropdown, bathroom_dropdown, output)


# In[25]:


import pandas as pd
import ipywidgets as widgets
from IPython.display import display

file_path = "House_Rent_Dataset.csv"
df = pd.read_csv(file_path)

city_dropdown = widgets.Dropdown(
    options=['', *df['City'].unique()],
    value='',
    description="Select City:",
    disabled=False
)

bhk_dropdown = widgets.Dropdown(
    options=['', *df['BHK'].unique()],
    value='',
    description="Select BHK:",
    disabled=False
)

bathroom_dropdown = widgets.Dropdown(
    options=['', *df['Bathroom'].unique()],
    value='',
    description="Select Bathroom:",
    disabled=False
)

output = widgets.Output()

def calculate_average_rent(change):
    selected_city = city_dropdown.value
    selected_bhk = bhk_dropdown.value
    selected_bathroom = bathroom_dropdown.value
    
    filtered_df = df[
        (df['City'] == selected_city if selected_city else True) &
        (df['BHK'] == selected_bhk if selected_bhk else True) &
        (df['Bathroom'] == selected_bathroom if selected_bathroom else True)
    ]
    
    if len(filtered_df) == 0:
        with output:
            output.clear_output()
            print("No matching data found.")
        return
    
    average_rent = filtered_df['Rent'].mean()
    
    with output:
        output.clear_output()
        if selected_city or selected_bhk or selected_bathroom:
            print(f"Predicted Average Rent for {selected_city}, {selected_bhk} BHK, {selected_bathroom} Bathroom: {average_rent:.2f}")

city_dropdown.observe(calculate_average_rent, names='value')
bhk_dropdown.observe(calculate_average_rent, names='value')
bathroom_dropdown.observe(calculate_average_rent, names='value')

display(city_dropdown, bhk_dropdown, bathroom_dropdown, output)


# # GUI

# In[52]:


import pandas as pd
import webbrowser
import os

df = pd.read_csv("House_Rent_Dataset.csv")
data_json = df.to_json(orient='records')

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Rent Prediction Tool</title>
</head>
<body>
    <div>
        <h2>Rent Prediction Tool</h2>
        <label for="city">Select City:</label>
        <select id="city" onchange="updateRent()">
            <option value="">Select City</option>
            <!-- Options will be filled by JavaScript -->
        </select>

        <label for="bhk">Select BHK:</label>
        <select id="bhk" onchange="updateRent()">
            <option value="">Select BHK</option>
            <!-- Options will be filled by JavaScript -->
        </select>

        <label for="bathroom">Select Bathroom:</label>
        <select id="bathroom" onchange="updateRent()">
            <option value="">Select Bathroom</option>
            <!-- Options will be filled by JavaScript -->
        </select>

        <p id="result">Predicted Average Rent: </p>
    </div>

    <script type="text/javascript">
        var data = {data_json};

        function fillDropdowns() {{
            var cities = new Set(data.map(item => item.City));
            var bhks = new Set(data.map(item => item.BHK));
            var bathrooms = new Set(data.map(item => item.Bathroom));

            cities.forEach(city => {{
                document.getElementById('city').innerHTML += '<option value="' + city + '">' + city + '</option>';
            }});

            bhks.forEach(bhk => {{
                document.getElementById('bhk').innerHTML += '<option value="' + bhk + '">' + bhk + '</option>';
            }});

            bathrooms.forEach(bathroom => {{
                document.getElementById('bathroom').innerHTML += '<option value="' + bathroom + '">' + bathroom + '</option>';
            }});
        }}

        function updateRent() {{
            var selectedCity = document.getElementById('city').value;
            var selectedBHK = document.getElementById('bhk').value;
            var selectedBathroom = document.getElementById('bathroom').value;

            var filteredData = data.filter(function(item) {{
                return (item.City == selectedCity || selectedCity === '') &&
                       (item.BHK == selectedBHK || selectedBHK === '') &&
                       (item.Bathroom == selectedBathroom || selectedBathroom === '');
            }});

            var averageRent = filteredData.reduce((acc, curr) => acc + curr.Rent, 0) / filteredData.length;
            document.getElementById('result').innerText = 'Predicted Average Rent: ' + (isNaN(averageRent) ? 'N/A' : averageRent.toFixed(2));
        }}

        fillDropdowns();
    </script>
</body>
</html>
"""

# Write the HTML content to a file
file_name = 'rent_prediction_tool.html'
with open(file_name, 'w') as file:
    file.write(html_content)

# Open the HTML file in the default web browser
webbrowser.open('file://' + os.path.realpath(file_name))


# # Model testing on other Dataset

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_excel('New dataset.xlsx')

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('price')
categorical_cols = df.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
])

models = {
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

X = df.drop(['price'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    print(model.__class__.__name__)
    print('Training Metrics:')
    print(' - MAE:', mean_absolute_error(y_train, pred_train))
    print(' - RMSE:', np.sqrt(mean_squared_error(y_train, pred_train)))
    print(' - R2 Score:', r2_score(y_train, pred_train))
    print('Testing Metrics:')
    print(' - MAE:', mean_absolute_error(y_test, pred_test))
    print(' - RMSE:', np.sqrt(mean_squared_error(y_test, pred_test)))
    print(' - R2 Score:', r2_score(y_test, pred_test))
    print()

for name, model in models.items():
    print(name)
    evaluate_model(model, X_train_transformed, y_train, X_test_transformed, y_test)


# In[ ]:





# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel('New dataset.xlsx')


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('price')
categorical_cols = df.select_dtypes(include=['object']).columns


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
])


models = {
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}


X = df.drop(['price'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    train_r2 = r2_score(y_train, pred_train)
    
    test_mae = mean_absolute_error(y_test, pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    test_r2 = r2_score(y_test, pred_test)
    
    return train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2


model_names = []
train_maes = []
train_rmses = []
train_r2s = []
test_maes = []
test_rmses = []
test_r2s = []

for name, model in models.items():
    model_names.append(name)
    train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2 = evaluate_model(model, X_train_transformed, y_train, X_test_transformed, y_test)
    train_maes.append(train_mae)
    train_rmses.append(train_rmse)
    train_r2s.append(train_r2)
    test_maes.append(test_mae)
    test_rmses.append(test_rmse)
    test_r2s.append(test_r2)


bar_width = 0.2
index = np.arange(len(model_names))

plt.figure(figsize=(12, 8))

plt.bar(index - bar_width, train_maes, bar_width, label='Training MAE', align='center')
plt.bar(index - bar_width, train_rmses, bar_width, label='Training RMSE', align='edge')
plt.bar(index, test_maes, bar_width, label='Testing MAE', align='center')
plt.bar(index, test_rmses, bar_width, label='Testing RMSE', align='edge')

plt.xlabel('Models')
plt.ylabel('Metrics')
plt.title('Model Performance Comparison')
plt.xticks(index, model_names, rotation=30)
plt.legend(loc='best')
plt.tight_layout()

plt.show()


# In[ ]:





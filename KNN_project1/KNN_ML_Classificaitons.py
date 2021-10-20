import pandas as pd # for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series
pd.set_option('display.max_columns', None) # set pd.options.display features
pd.set_option('display.max_rows', None)
import numpy as np # for multi array manipÃ¼lation process,Linear algebra
import seaborn as sns # for data visulations,its very usefully
import matplotlib.pyplot as plt # for data visulations
plt.rcParams["figure.figsize"]=(18,8)
from matplotlib.colors import ListedColormap # for color palette for plots
sns.set_theme(style="darkgrid")
from sklearn import utils
from sklearn.preprocessing import StandardScaler #for normalization and data scaler
from sklearn.model_selection import train_test_split, GridSearchCV # fro trainins,test ,validaitons ana model hyperparameters tuning
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,classification_report # for models evaulations 
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor # for model build, outlier detections
from sklearn.decomposition import PCA # for dimentional reductions .Principle compomnet Analys
from sklearn.datasets import load_breast_cancer # for datasets 
# closed warning library
import warnings
warnings.filterwarnings("ignore")

# include data 
try:
    
    data = pd.read_csv("KNN_project/breast_cancer.csv")
    
except Exception as e:
    
    print(type(e))
    print(e)
    #from import datasets from sklearn if CSV FİLE İS NOT FOUND
    data=load_breast_cancer()

#create df  if CSV FİLE İS NOT FOUND
if type(data)==utils.Bunch:
    df=np.concatenate((data.target.reshape(-1,1),data.data), axis=1)
    data=pd.DataFrame(df,columns=np.insert(data.feature_names,0,"diagnosis"))
else:
    # remove unnecesssary columns
    data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)

data = data.rename(columns = {"diagnosis":"Dependent"})
data_=data.copy()
sns.countplot(y=data["Dependent"],
                   linewidth=1,
                   edgecolor=sns.color_palette("colorblind", 2))
print(data.Dependent.value_counts())

if not data.Dependent.dtype in [np.float64,np.int64]:
    
    data["Dependent"] = [0 if i.strip() == "M" else 1 for i in data.Dependent]

elif  data.Dependent.dtype in [np.float64,np.int64]:
    data_["Dependent"] = ["Malignant" if i==0 else  "Belign"  for i in data.Dependent]

print(len(data))

print(data.head())

print("Data shape ", data.shape)

data.info()

describe = data.describe()

columns_features=data.columns.to_numpy()

#%% exploratory data analysis
y_depent="Dependent"
x_indepent=columns_features[1:]
# fully Feature Correlation
corr_matrix = data.corr()
#print(corr_matrix)

#dependent variable correlations with independent other features
sns.heatmap(corr_matrix[[y_depent]].sort_values(by=y_depent),annot = True, fmt = ".2f",xticklabels=True, yticklabels=True)
plt.title("Correlation Between Features")
plt.show()


# for dependent subsample correlations
thresh = 0.60 # for target variable
filtre = np.abs(corr_matrix[y_depent]) < thresh
corr_features=np.append(corr_matrix.columns[filtre].to_numpy(),[y_depent])
thres_corr_m=data[corr_features].corr()
sns.clustermap(thres_corr_m, annot = True, fmt = ".2f")
plt.title(f"Correlation Between Features threshold= {thresh}")
plt.show()

#there some correlated features.We drop out these features because of we improve models
filtres_corr=(np.abs(thres_corr_m)== 1) | (np.abs(thres_corr_m)< 0.5)
maske = thres_corr_m[filtres_corr] # for filter 
sns.heatmap(maske, cmap="Reds",annot = True, fmt = ".2g",xticklabels=True, yticklabels=True)
plt.show()

masked=np.tril(maske)
sns.heatmap(maske,mask=masked ,annot = True, fmt = ".2g",xticklabels=True, yticklabels=True)
plt.title("Correlation Between Features")






for i,ser in enumerate(filtres_corr):
    ser.reset_index(drop=True,inplace=True)
    corr_matrix[corr_features].corr()[]

corr_matrix_filer=[corr_matrix<.75]

corr_pairs=corr_matrix.unstack()

sorted_pairs = corr_pairs.sort_values(kind="quicksort")

strong_pairs = sorted_pairs[(abs(sorted_pairs) <0.60)]

threscorr_pairs=strong_pairs.sort_index(level=0).unstack()

threscorr_pairs.isnull().any()
threscorr_pairs.isnull().sum().sort_values()





for i,ser in enumerate(filtres_corr):
    ser.reset_index(drop=True,inplace=True)
    corr_matrix[corr_features].corr()[]

corr_matrix_filer=[corr_matrix<.75]

corr_pairs=corr_matrix.unstack()

sorted_pairs = corr_pairs.sort_values(kind="quicksort")

strong_pairs = sorted_pairs[(abs(sorted_pairs) <0.60)]

threscorr_pairs=strong_pairs.sort_index(level=0).unstack()

threscorr_pairs.isnull().any()
threscorr_pairs.isnull().sum().sort_values()

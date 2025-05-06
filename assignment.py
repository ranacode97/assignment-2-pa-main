import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_csv('C:/WSU/Spring 2024/Predictive Analytics/images.csv')

# Printing basic information about the dataset
print(data.shape)
print(data.head())
print(data.isnull().sum())  # Checking for the missing values


# Define the function to calculate evaluation metrics
def scores(y_test, y_pred):
    # Calculate accuracy, confusion matrix, and F1 score
    score = accuracy_score(y_test, y_pred)
    mat = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print the evaluation results
    print("Confusion Matrix:\n", mat)
    print("Accuracy Score:", score)
    print("F1 Score:", f1)

# Load the dataset
data = pd.read_csv('C:/WSU/Spring 2024/Predictive Analytics/images.csv')

# Replace missing values with the mean of each column and convert the DataFrame to a numpy array.
data_filled = data.fillna(data.mean())
data = data_filled.values
print(data.shape)

# Split the features and labels
x = data[:, :-1]  # All columns except the last one 
y = data[:, -1]  # The last column is the target label

# Normalize the features
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
# the data set is then divided into the training dataset and the testing dataset (70% training, 30% testing)

# Define & train the Neural Network Model
clf = MLPClassifier(activation='relu', hidden_layer_sizes=(150, 10), verbose=True, max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
#Defining the structure of the neural network with two hidden layers of sizes 150 and 10

# Predictions on test set
y_pred = clf.predict(X_test)

# Printing the results for the Neural Network Model
print("\nNeural Network Model Evaluation:")
scores(y_test, y_pred)

plt.plot(clf.loss_curve_)
plt.title('Loss Curve of Neural Network')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid()
plt.show()from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

##  loops
def convert_image(path):
        img = Image.open(path) #open image from path
        img = img.convert('L')  # converting image to grayscale
        img_arr = np.array(img) #coverting that image to array
        return img_arr.reshape(1, -1) #reshape to 1D


cid=[0,1,2,3,4]
folders = ['thread error', 'oil spot', 'objects', 'hole', 'good']

all_images = []
path='C:/Users/wambu/Downloads/paa/'

for folder,cls in zip(folders,cid):
    pth=str(path+folder)
    # list files
    files=os.listdir(pth)
    print('Working with class:',folder)
    for f in files:
        im_pth=str(pth+'/'+f)
        ci=convert_image(im_pth) #fetch image
        ci=np.append(ci, [cls])
        #print(ci.shape)
        all_images.append(ci) #add processed img to list

all_images_arr = np.array(all_images)

df = pd.DataFrame(all_images_arr)
df.to_csv('images.csv', index=False)


data = pd.read_csv('images.csv')
print(data.head())

# this is a classification task as the target variable has distinct
# values according to the defects of the images.

print(data.isnull().sum()) # check for missing values

data_filled = data.fillna(data.mean()) # fill NA's with column mean


# relevant variables
# all pixel values from 0 to 4095 show grayscale image data.
# the last column 4096 is the target variable

# irrelevant variables
# no irrelevant variables as all pixels form part of an image.

data=data_filled.values # convert to array
print(data_filled.shape)


x=data[:,:-1] #all columns except last one
y=data[:,data.shape[1]-1]
print(np.unique(x))
print(np.unique(y))


# use pca to reduce x components (pixel values of images)
# principal components == transformed features
# get x pca value and then use it for train test split


# Normalizing before pca
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x) # improving the performance

pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x)

print(f"Number of components after PCA: {x_pca.shape[1]}")


##

X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.3, random_state=42)

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

# classifier model
clf = MLPClassifier(activation='relu',hidden_layer_sizes=(150,10),verbose=True,max_iter=1000)

# 5-fold cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)  #ensures model is not overfitting
print("CV scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## pca

plt.figure(figsize=(10, 7))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('PCA of Image Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class Labels')
plt.show()


## hist of pixel values
# distribution of features. The spread of pixel values, dark/bright. outliers?

plt.hist(data.flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Distribution of pixel values")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

## boxplot
# visualizes spread of pixel values in different classes. 
sns.boxplot(x=y, y=np.mean(x, axis=1))
plt.title("Boxplot of Pixel Intensity")
plt.xlabel("Class")
plt.ylabel("Pixel Intensity")
plt.show()
import timeit
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('images.csv')

# Print basic information about the dataset
print(data.shape)
print(data.head())
print(data.isnull().sum())  # Check for missing values

# Fill missing values with column means and convert to numpy array
data_filled = data.fillna(data.mean())
data = data_filled.values
print(data.shape)

# Separate features and target labels
x = data[:, :-1]  # All columns except the last one (features)
y = data[:, -1]  # The last column is the target label

# Normalizing the features before applying PCA
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Apply PCA to reduce dimensions of the features
pca = PCA(n_components=0.95)  # Retain 95% variance
x_pca = pca.fit_transform(x_scaled)
print(f"Number of components after PCA: {x_pca.shape[1]}")

# Split the data into training and testing sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Create an MLPClassifier model
model = MLPClassifier(random_state=42, hidden_layer_sizes=(150, 10), max_iter=1000)

# Perform cross-validation on the training set
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")

# Train the final model on the entire training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the final model using the test set
def scores(y_test, y_pred):
    score = accuracy_score(y_test, y_pred)
    mat = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Confusion Matrix:\n", mat)
    print("Accuracy Score:", score)
    print("F1 Score:", f1)

print("\nFinal Model Evaluation:")
scores(y_test, y_pred)

###SVM
#Linear SVM
model = LinearSVC(C=1, loss="hinge", max_iter=10000)
#Start model fit timer
start = timeit.default_timer()
#Model training
model.fit(X_train, y_train)
#Stop model fit timer
stop = timeit.default_timer()
#Calculate train time
timeTaken = stop - start
print('Time to run the training: ', timeTaken)

#Predict the fitted model
y_pred = model.predict(X_test)
#Evaluate model
print("\nLinear SVM model evaluation:")
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred) * 100)
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))


def plot_learning_curve(model, X_train, y_train, X_test, y_test, train_sizes):
    """Plots the learning curve by using varying amounts of training data."""
    train_scores = []
    test_scores = []

    # Iterate over different training sizes
    for size in train_sizes:
        # Use only a subset of the training data
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]

        # Train the model on the subset
        model.fit(X_train_subset, y_train_subset)

        # Record accuracy for both train and test data
        train_pred = model.predict(X_train_subset)
        test_pred = model.predict(X_test)

        train_scores.append(accuracy_score(y_train_subset, train_pred))
        test_scores.append(accuracy_score(y_test, test_pred))

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, label='Training Accuracy', color='blue')
    plt.plot(train_sizes, test_scores, label='Validation Accuracy', color='orange')

    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Define different training sizes to test
train_sizes = np.linspace(0.1, 1.0, 10) * len(X_train)
train_sizes = train_sizes.astype(int)

plot_learning_curve(model, X_train, y_train, X_test, y_test, train_sizes)

#Cross validation for finding good margin
Margins = [0.01, 0.1, 1, 10]
resAccuracy = []

for m in Margins:
    svm_clf_cv = LinearSVC(C=m, loss="hinge", max_iter=10000)
    svm_clf_cv.fit(X_train,y_train)
    y_pred = svm_clf_cv.predict(X_test)
    accCurr = accuracy_score(y_test, y_pred)*100
    print(f'C = {m}, accuracy = {accCurr}')
    resAccuracy.append(accCurr)
plt.figure(figsize=(6, 4))
plt.semilogx(Margins, resAccuracy, '*-')
plt.show()

# The plot shows the classes are well separated so it has the same accuracy no matter what margin is used

##SVM with kernel
svm_clf2 = SVC(kernel="rbf", gamma='scale', C=1, max_iter=10000)
#Start timer for model train
start = timeit.default_timer()
#Train model
svm_clf2.fit(X_train,y_train)
#Stop timer for model train
stop = timeit.default_timer()
#Calculate train time
timeTaken = stop - start
print('Time to run the training: ', timeTaken)
#Predict the fitted model
y_pred2 = svm_clf2.predict(X_test)
#Model evaluation
plot_learning_curve(svm_clf2, X_train, y_train, X_test, y_test, train_sizes)

print("\nKernel SVM model evaluation:")
print(confusion_matrix(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2)*100)
print("F1 Score:", f1_score(y_test, y_pred2, average='weighted'))


# Check the shape of the DataFrame (rows, columns)
#print(f'Shape of the DataFrame: {data.shape}')
#last col is 4096 which is our target variable that has values of class,
# 0 = good
# 1 = hole
# 2 = objects
# 3 = oil spot
# 4 = thread error
"""
Image size: 64x64
Classes: ['good', 'hole', 'objects', 'oil spot', 'thread error']
Image count (per class): [23170, 337, 873, 636, 620]
"""
#shape of 25600 rows and 4097 columns


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.tree import export_graphviz, plot_tree, DecisionTreeClassifier
import sklearn
import matplotlib.pyplot as plt


#loading dataset from .csv file
data = pd.read_csv('D:/1 madiha/term 3/predictive analytics/assignment 2/images.csv')

#printing the shape of dataset
print(data.shape)
print(data.head())
#checking for any missing values
print(data.isnull().sum()) 
#filling the missing values with column means and convert to numpy array
data_filled = data.fillna(data.mean())
data = data_filled.values
print(data.shape)
#allocating the features to X and target variable to y
X = data[:, :-1]  # All columns except the last one (features)
y = data[:, -1]  # The last column is the target label
#splitting entire dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#decision tree model 
dt_classifier = DecisionTreeClassifier()
#training the model
dt_classifier.fit(X_train, y_train)
#predicting on test set
y_pred_dt = dt_classifier.predict(X_test)
print(y_pred_dt)
#evaluation on predictions
print("Decision Tree Classification Report:", classification_report(y_test, y_pred_dt))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred_dt))
print("Accuracy Score:", accuracy_score(y_test, y_pred_dt))
#plotting the tree
export_graphviz(dt_classifier, 
                out_file = image_path("detect_tree.dot"),
                rounded = True,
                filled= True)
plt.figure(figsize=[10,6])
sklearn.tree.plot_tree(dt_classifier)
plt.show()



###model random forest
#model fitting on train set
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
#checking feature inportance
importances = rf_model.feature_importances_
feature_names = X.columns
#check lengths before creating DataFrame
print(f"Number of features: {X.shape[1]}")
print(f"Length of importances: {len(importances)}")
#create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
#proceed to plot feature importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
#evaluaitng model
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
cm = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:",cm)
print("Classification Report:",classification_report(y_test, y_pred_rf))
#plotting the confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()







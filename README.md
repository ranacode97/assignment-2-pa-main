⸻

Image Classification with Neural Networks and SVM

This project performs image classification on grayscale defect images using Machine Learning models including MLP Neural Networks and Support Vector Machines (SVM). The dataset is created from raw image files, preprocessed, and transformed using Principal Component Analysis (PCA) to improve model performance.

⸻

Project Structure

The original codebase consisted of four separate Python scripts, now combined into one streamlined script performing the following steps:
	1.	Image Loading and Preprocessing
	2.	Dataset Creation from Image Files
	3.	Model Training with MLPClassifier and SVM
	4.	Dimensionality Reduction with PCA
	5.	Evaluation with Accuracy, F1 Score, and Confusion Matrix
	6.	Visualization: Loss Curve, PCA Scatter Plot, Histogram, and Boxplot
	7.	Learning Curve Plotting (Partially Implemented)

⸻

Dataset

The dataset is built from grayscale images in five categories:
	•	thread error
	•	oil spot
	•	objects
	•	hole
	•	good

These images are flattened into 1D arrays, labeled with class IDs (0–4), and saved as a CSV file named images.csv.

⸻

Requirements

Install required packages using:

pip install numpy pandas scikit-learn matplotlib seaborn pillow



⸻

How to Run
	1.	Organize Images
Ensure image folders (thread error, oil spot, etc.) are placed in a directory, and update the path variable in the code accordingly.
	2.	Run Script
Execute the combined script to:
	•	Convert images to CSV
	•	Apply PCA
	•	Train and evaluate models
	3.	Output
	•	Evaluation metrics for MLP and SVM
	•	Visualizations for PCA, loss curve, pixel distribution

⸻

Evaluation Metrics

Both models are evaluated using:
	•	Accuracy Score
	•	F1 Score (Weighted)
	•	Confusion Matrix
	•	5-Fold Cross-Validation (MLP)
	•	Training Time (SVM)

⸻

Visualizations
	•	PCA Scatter Plot for visualizing the first two components
	•	Loss Curve for MLP training convergence
	•	Histogram of pixel intensity distribution
	•	Boxplot of pixel intensity by class

⸻

Models Used
	•	MLPClassifier from sklearn.neural_network
	•	2 hidden layers: (150, 10)
	•	ReLU activation
	•	1000 max iterations
	•	Support Vector Machine
	•	Linear SVM with hinge loss
	•	Max iterations: 10,000

⸻

File Outputs
	•	images.csv: Flattened grayscale image dataset
	•	Visualizations displayed using matplotlib

⸻

Notes
	•	PCA reduces dimensions to preserve 95% of the variance.
	•	The code automatically handles missing values using column means.
	•	Ensure the dataset path and image folder structure are correct before execution.

⸻

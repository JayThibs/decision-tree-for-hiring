import pandas as pd
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydotplus
from sklearn.ensemble import RandomForestClassifier

# Load dataset
previousApplicantsDataset = "/Users/jacquesthibodeau/Desktop/Media/Online Courses/Coding courses/Data Science and Machine Learning with Python/Python Data Science course material/PastHires.csv"
previousApplicantDataFrame = pd.read_csv(previousApplicantsDataset, header = 0)

# Map all Y and N (Yes and No) to 1 and 0, respectively
yesOrNoBool = {'Y': 1, 'N':0}
previousApplicantDataFrame['Hired'] = previousApplicantDataFrame['Hired'].map(yesOrNoBool)
previousApplicantDataFrame['Employed?'] = previousApplicantDataFrame['Employed?'].map(yesOrNoBool)
previousApplicantDataFrame['Top-tier school'] = previousApplicantDataFrame['Top-tier school'].map(yesOrNoBool)
previousApplicantDataFrame['Interned'] = previousApplicantDataFrame['Interned'].map(yesOrNoBool)

# Map all BS (Bachelor's), MS (Master's) and PhD (Doctorate) to 0, 1 and 2, 
# respectively
degreeLevelBool = {'BS': 0, 'MS': 1, 'PhD': 2}
previousApplicantDataFrame['Level of Education'] = previousApplicantDataFrame['Level of Education'].map(degreeLevelBool)

# Place all columns from the dataset containing the features in list
features = list(previousApplicantDataFrame.columns[:6])

# Place the decision of whether the applicant was hired or not in y and
# place all features in X
y = previousApplicantDataFrame["Hired"]
X = previousApplicantDataFrame[features]

# Constuct the decision tree with the features and the label
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X,y)

# Save the decision tree image
dot_data = StringIO()  
tree.export_graphviz(classifier, out_file=dot_data, feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('test.png')

## Random Forest Classifier
# This runs the dataset n times and takes the predicts the best outcome
classifier = RandomForestClassifier(n_estimators=1000)
classifier = classifier.fit(X, y)

# Predict employment of a particular person
print(classifier.predict([[10, 1, 4, 0, 0, 0]]))
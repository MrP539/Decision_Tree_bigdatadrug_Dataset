import sklearn
import pandas as pd
import numpy as np
import os

import sklearn.model_selection
import sklearn.preprocessing
import sklearn.tree
from tool import find_nan_values_at_column
from tool import create_confusion_matrix

raw_data = pd.read_csv(os.path.join(r"D:\machine_learning_AI_Builders\ML_Algorithm\Decision_Tree_bigdatadrug_Dataset\bigdatadrug.csv"))

print(raw_data.shape)


labels_encode = {}
for feature_ in raw_data.columns:
    if raw_data[feature_].dtype == type(object):
        Labels_Encoder = sklearn.preprocessing.LabelEncoder()
        encoded_values = Labels_Encoder.fit_transform(raw_data[feature_])
        raw_data[feature_] = encoded_values
        labels_encode[feature_] = Labels_Encoder
    

decode = {column:{value:encoded_value for value,encoded_value in zip(values.classes_,range(len(values.classes_)))} for column,values in labels_encode.items()}
    
int_to_target_name = {v:k for k,v in decode["Drug"].items()}


feature = raw_data.drop("Drug",axis=1).values
target = raw_data["Drug"].values


x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(feature ,target, test_size=0.2,shuffle=True,random_state=42)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

decision_tree_model = sklearn.tree.DecisionTreeClassifier()

decision_tree_model.fit(X=x_train,y=y_train)

eval = decision_tree_model.score(X=x_test,y=y_test)
pred = decision_tree_model.predict(x_test)

print(pred)

pred_unit = decision_tree_model.predict([x_test[0]])
print(f"{int_to_target_name[int(pred_unit.item())]} VS {int_to_target_name[y_test[0]]}")

print(f"Accuracy : {eval*100.}")
create_confusion_matrix.CREATE_CONFUSION_MATRICS(y_pred=decision_tree_model.predict(x_test),y_actual=y_test,numclass=len(decode))

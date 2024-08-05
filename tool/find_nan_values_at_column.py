
import pandas as pd
import numpy as np

def check_type_and_nav_values(class_list,df):

    result= []
    for feature in class_list:

        #print(type(df[feature][0]) == type("A"))
        
        if df[feature].isnull().values.any():
            result.append(feature)
    if result == []:
        print("Your_data_is_not_nan_values")
    else:
        print(result)

def check_type_and_any_values(class_list,df,any):

    result= {}
    for feature in class_list:

        #print(type(df[feature][0]) == type("A"))
        
        if any in df[feature].values.tolist():
            result[feature] = df[feature].dtype
    if result == {}:
        print("Your_data_is_not_nan_values")
    else:
        print(result)
    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statistics
import pprint
from sklearn.decomposition import PCA
import re
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 1 - Exploratory functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def check_data_type(dataframe):
    print(dataframe.dtypes)


def find_missing_values(dataframe):
    columns_with_nulls = dataframe.columns[dataframe.isnull().any()]
    df_columns_with_nulls = dataframe[columns_with_nulls]
    print(df_columns_with_nulls)


def find_rows_with_zero_values(dataframe):
    """
    find the rows with zero values
    :param dataframe:
    :return:
    """
    zero_values = []
    for col in dataframe.columns:

        for row in dataframe[col].items():
            if row[1] == '0':
                if row[0] not in zero_values:
                    zero_values.append(row[0])
                # zero_values.append(row[0])
            # zero_values = [row if val == 0 else None for val in row[1]]
        # zero_values = (dataframe[] == 0).all(axis=1)
    # for row in zero_values.index.values.tolist():
    #     print(row)
    #     print(zero_values[row])
    #     print("")
    return zero_values


def find_cells_with_missing_values(dataframe):
    """
    find the cells with missing values
    :param dataframe:
    :return:
    """
    missing_values = dataframe.isnull()
    for column in missing_values.columns.values.tolist():
        print(column)
        print(missing_values[column].value_counts())
        print("")


def find_number_rows_of_missing_values(dataframe):
    """
    find the number of rows with missing values
    :param dataframe:
    :return:
    """
    missing_values = dataframe.isnull()
    num_rows = missing_values.any(axis=1).sum()
    print(f"Number of rows with missing values: {num_rows}")
    return num_rows


def find_mean_of_column(dataframe, column_name):
    mean = dataframe[column_name].mean()
    return mean


def find_median_of_column(dataframe, column_name):
    median = dataframe[column_name].median()
    return median


def find_mode_of_column(dataframe, column_name):
    """
    find the mode of the column - the value that appears the most
    :param dataframe:
    :param column_name:
    :return:
    """
    mode = dataframe[column_name].mode()[0]
    return mode


def statistics_dictionary(dataframe):
    """
    create a dictionary of the mean, median, and mode of each column
    :param dataframe:
    :return:
    """
    column_names = dataframe.columns.tolist()
    dict_of_mean_median_mode = {}
    for column in column_names:
        if dataframe[column].dtype == 'float64' or dataframe[column].dtype == 'int64':
            if column in ['RecordNumber', 'CustomerId']:
                continue
            mean = find_mean_of_column(dataframe, column)
            median = find_median_of_column(dataframe, column)
            mode = find_mode_of_column(dataframe, column)
            std = statistics.stdev(dataframe[column])
            cv = round(std / mean, 2)
            min_obs = dataframe[column].min()
            max_obs = dataframe[column].max()
            observations = len(dataframe[column])
            unique_count = len(dataframe[column].unique())
            dict_of_mean_median_mode[column] = {'mean': mean,
                                                'median': median,
                                                'mode': mode,
                                                'std': std,
                                                'cv': cv,
                                                'min': min_obs,
                                                'max': max_obs,
                                                'observations': observations,
                                                'unique_count': unique_count}
        else:
            mode = find_mode_of_column(dataframe, column)
            dict_of_mean_median_mode[column] = {'mode': mode}
    return dict_of_mean_median_mode


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 2 - Data Cleaning and Preparation Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def numeric_is_positive(df, column_name):
    """
    check if the values in the column are positive
    :param df:
    :param column_name:
    :return: true if all values are positive, otherwise a list of record numbers where the values are negative
    """
    vals = df[column_name].values
    negative_indices = np.where(vals < 0)[0]
    if negative_indices.size > 0:
        lst = [val for val in negative_indices]
        return lst
    return True


def positive_column(dataframe, column_name):
    """
    find the rows where the column has negative values
    :param dataframe:
    :param column_name:
    :return: true if there are no negative values, otherwise a list of record numbers where the column has negative values
    """
    vals = dataframe[column_name].values
    negative_indices = np.where(vals < 0)[0]
    if negative_indices.size > 0:
        lst = [val for val in negative_indices]
        return lst
    return True


def fill_missing_numerical_values(dataframe):
    """
    fill the missing values with the mean of the column
    :param dataframe:
    :return:
    """
    column_names = dataframe.columns.tolist()
    for column in column_names:
        if dataframe[column].dtype == 'float64' or dataframe[column].dtype == 'int64':
            mean = round(find_mean_of_column(dataframe, column), 0)
            dataframe[column].fillna(mean, inplace=True)
    return dataframe


def fill_missing_categorical_values(dataframe):
    """
    fill the missing values with the mode of the column
    :param dataframe:
    :return:
    """
    column_names = dataframe.columns.tolist()
    for column in column_names:
        if dataframe[column].dtype == 'object':
            mode = find_mode_of_column(dataframe, column)
            dataframe[column].fillna(mode, inplace=True)
    return dataframe


def fill_all_missing_values(dataframe):
    """
    fill all missing values in the dataset
    :param dataframe:
    :return:
    """
    fill_missing_numerical_values(dataframe)
    fill_missing_categorical_values(dataframe)
    return dataframe


def make_cols_numeric(dataframe, col_list):
    for col in col_list:
        if dataframe[col].dtype == 'object':
            dataframe[col] = dataframe[col].astype(str)
            # Step 1: Remove commas
            dataframe[col] = dataframe[col].str.replace(',', '')
            # Step 2: Convert to numeric (integer type)
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    return dataframe


def encode_data(dataframe, cols: list[str]):
    """
    encode the categorical columns
    :param dataframe:
    :param cols:
    :return:
    """
    for col in cols:
        unique_vals = dataframe[col].unique()
        unique_dict = {item: ind for ind, item in enumerate(unique_vals)}
        dataframe[col] = dataframe[col].map(unique_dict)
    return dataframe


def del_rows(dataframe, row_indices: list[int]) -> pd.DataFrame:
    """
    delete the rows from the dataframe
    :param dataframe:
    :param row_indices:
    :return:
    """
    print(dataframe)
    df = dataframe.copy()
    df_without_zero = df.drop(row_indices, axis=0)
    print(df_without_zero)
    return df_without_zero


def shorten_column_names(dataframe, col_list):
    """
    shorten the column names
    :param dataframe:
    :return:
    """
    # Use regex to find everything before the first ':'
    for col in col_list:
        new_name = re.sub(r':.*', '', col)
        dataframe.rename(columns={col: new_name}, inplace=True)

    return dataframe


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 3 - Linear Regression
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def perform_linear_regression(dataframe, training_set_fraction, target_var):
    """
    perform linear regression on the dataset
    :param dataframe:
    :param training_set_fraction:
    :return: plot with regression line and y=x line
    """
    # dataframe = dataframe.drop(columns=["LastName", "RecordNumber", "CustomerId"])
    x = dataframe.drop(columns=[target_var])
    y = dataframe[target_var]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_set_fraction, random_state=1)

    # scikit-learn model
    lm = LinearRegression().fit(x_train, y_train)
    y_pred = lm.predict(x_test)

    print(f"the mean absolute error is: {mean_absolute_error(y_test, y_pred)}")
    print(f"the mean squared error is: {mean_squared_error(y_test, y_pred)}")

    # statsmodels model
    formula = f'{target_var} ~ ' + '+'.join(x.columns)
    smf_model = smf.ols(formula=formula, data=dataframe)
    results = smf_model.fit()
    y_predicted = results.predict(x_test)

    # Show the summary of the statsmodels linear regression
    print(results.summary())

    # Plot
    plt.scatter(y_test, y_predicted)
    plt.xlabel('actual data'), plt.ylabel('predicted data')

    # Adding the line y=x
    lims = [np.min([y_test.min(), y_predicted.min()]),
            np.max([y_test.max(), y_predicted.max()])]
    plt.plot(lims, lims, 'r--', alpha=0.75, label='y = x')

    # Adding the regression line
    regression_slope, regression_intercept = np.polyfit(y_test, y_predicted, 1)
    regression_line = regression_slope * np.array(lims) + regression_intercept
    plt.plot(lims, regression_line, 'b-', alpha=0.75, label='Regression Line')

    # Adding a legend, title and showing the plot
    plt.legend()
    plt.title(f'Linear Regression Model Predicting {target_var}')
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 4 - Logistic Regression
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def turn_col_to_binary(dataframe, col_name):
    """
    turn the column into a binary column, depending on whether the value is greater than the mean
    :param dataframe:
    :param col_name:
    :return:
    """
    new_df = dataframe.copy()
    mean = dataframe[col_name].mean()
    new_df[col_name] = (new_df[col_name] >= mean).astype(int)
    return new_df


def split_log_regression(dataframe, training_set_frac):
    """
    split the data into training and test sets
    :param dataframe:
    :param training_set_frac:
    :return:
    """
    random_rows = np.random.rand(len(dataframe)) < training_set_frac
    data_train = dataframe[random_rows]
    data_test = dataframe[~random_rows]
    return data_train, data_test


def initiate_log_regression(train_data, test_data, target_var):
    """
    perform logistic regression on the dataset, print evaluation metrics, show confusion matrix
    :param train_data:
    :param test_data:
    :return:
    """
    # part 1 - perform logistic regression
    lr = LogisticRegression(solver='lbfgs', max_iter=10000)
    scaler = StandardScaler()

    y_train = train_data[target_var]
    x_train = train_data.drop(columns=[target_var])
    y_test = test_data[target_var]
    x_test = test_data.drop(columns=[target_var])

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    log_model = lr.fit(x_train_scaled, y_train)
    predictions = log_model.predict(x_test_scaled)

    # part 2 - evaluate the model
    print('Model accuracy on training set: {:.2f}'.format(lr.score(x_train_scaled, y_train)))
    print('Model accuracy on test set: {:.2f}'.format(lr.score(x_test_scaled, y_test)))

    cm = confusion_matrix(y_test, predictions)
    pre = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)

    print(f'Precision score: {pre:3.3f}')
    print(f'Recall score: {rec:3.3f}')
    print(f'F1 score: {f1:3.3f}')
    print(f'AUC score: {auc:3.3f}')
    print('Confusion matrix:')
    print(cm)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 16})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("Confusion Matrix for Logistic Regression Model")
    plt.show()


def main_old():
    """
    main function to run all of the above
    :return:
    """
    df = pd.read_csv("RetentionBusinessCustomers (1).csv").copy()

    # phase 1 - data validation
    loyalty_larger_than_age = is_loyalty_years_smaller_than_age(df)
    negative_spending = positive_column(df, 'AnnualSpending')
    rows_to_delete = loyalty_larger_than_age + negative_spending
    del_rows(df, rows_to_delete)

    # phase 2 - data preparation
    df_without_blanks = fill_all_missing_values(df)
    cols_to_encode = ['Location', 'CardType', 'Gender']
    print('Dataframe Statistics:\n')
    pprint.pprint(statistics_dictionary(df_without_blanks), indent=4, width=40)
    encoded_df = encode_data(df_without_blanks, cols_to_encode)
    encoded_df.to_csv('refactored_output.csv', index=False)

    # phase 3 - linear regression
    print("\nLinear Regression!:\n")
    perform_linear_regression(encoded_df)

    # phase 4 - logistics regression
    print("\nLogistic Regression!:\n")
    binary_tree_df = turn_col_to_binary(encoded_df, 'AnnualSpending')
    log_data_train, log_data_test = split_log_regression(binary_tree_df, 0.7)
    initiate_log_regression(log_data_train, log_data_test)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 5 - Model Training and Evaluation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def create_random_forest_regression_model(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    # print(classification_report(y_test, y_predict))
    print('Random Forest Score: {0:2.2f}'.format(rf.score(X_test, y_test)))

    print('Scores on Training Set:')
    print('Random Forest Score: {0:2.2f}'.format(rf.score(X_train, y_train)))
    print('Scores on Test Set:')
    print('Random Forest Score: {0:2.2f}'.format(rf.score(X_test, y_test)))
    # precision = precision_score(y_test, y_predict, average='macro', zero_division=0)
    # recall = recall_score(y_test, y_predict, average='macro', zero_division=0)
    # f1 = f1_score(y_test, y_predict, average='macro', zero_division=0)
    # print('Random Forest \n precision: ', precision, '\n', 'recall: ', recall, '\n', 'f1: ', f1)
    print('Random Forest accuracy_score:', accuracy_score(y_test, y_predict))
    return rf


def create_random_forest_classification_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, n_jobs=1)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)

    # Print classification metrics
    print('Classification Report:\n', classification_report(y_test, y_predict, zero_division=0))
    print('Accuracy Score: {0:.2f}'.format(accuracy_score(y_test, y_predict)))
    print('Precision Score: {0:.2f}'.format(precision_score(y_test, y_predict, average='macro', zero_division=0)))
    print('Recall Score: {0:.2f}'.format(recall_score(y_test, y_predict, average='macro', zero_division=0)))
    print('F1 Score: {0:.2f}'.format(f1_score(y_test, y_predict, average='macro', zero_division=0)))

    print('Scores on Training Set:')
    print('Random Forest Training Score: {0:.2f}'.format(rf.score(X_train, y_train)))
    print('Scores on Test Set:')
    print('Random Forest Test Score: {0:.2f}'.format(rf.score(X_test, y_test)))

    return rf


# Import Support Vector Classifier


def create_ada_boost_classifier(X_train, y_train, X_test, y_test):
    abc = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
    abc.fit(X_train, y_train)
    y_predict = abc.predict(X_test)
    print(classification_report(y_test, y_predict, zero_division=0))
    print('ada_boost_classifier Score: {0:2.2f}'.format(abc.score(X_test, y_test)))

    print('Scores on Training Set:')
    print('ada_boost_classifier Score: {0:2.2f}'.format(abc.score(X_train, y_train)))
    print('Scores on Test Set:')
    print('ada_boost_classifier Score: {0:2.2f}'.format(abc.score(X_test, y_test)))
    print('abc accuracy_score:', accuracy_score(y_test, y_predict))
    return y_predict


def create_ada_boost_classifier1(X_train, y_train, X_test, y_test):
    abc = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),  # Simplify the base estimator
        n_estimators=100,  # Reduce the number of estimators
        learning_rate=0.01  # Lower learning rate
    )

    abc.fit(X_train, y_train)
    y_predict = abc.predict(X_test)

    print(classification_report(y_test, y_predict, zero_division=0))
    print('ada_boost_classifier Score: {0:2.2f}'.format(abc.score(X_test, y_test)))
    print('Scores on Training Set:')
    print('ada_boost_classifier Score: {0:2.2f}'.format(abc.score(X_train, y_train)))
    print('Scores on Test Set:')
    print('ada_boost_classifier Score: {0:2.2f}'.format(abc.score(X_test, y_test)))
    precision = precision_score(y_test, y_predict, average='macro', zero_division=0)
    recall = recall_score(y_test, y_predict, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_predict, average='macro', zero_division=0)
    print('ada_boost \n precision: ', precision, '\n', 'recall: ', recall, '\n', 'f1: ', f1)
    print('abc accuracy_score:', accuracy_score(y_test, y_predict))

    return abc


def create_adaboost_regressor(X_train, y_train, X_test, y_test):
    # Initialize AdaBoost Regressor with a DecisionTreeRegressor as the base estimator
    adaboost_reg = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),  # Base estimator for AdaBoost
        n_estimators=100,  # Number of boosting stages
        learning_rate=0.01  # Step size for updating the weights
    )

    # Fit the model
    adaboost_reg.fit(X_train, y_train)

    # Predict on the test set
    y_predict = adaboost_reg.predict(X_test)

    # Print evaluation metrics for regression
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    print('AdaBoost Regressor Mean Squared Error: {0:.2f}'.format(mse))
    print('AdaBoost Regressor Mean Absolute Error: {0:.2f}'.format(mae))
    print('AdaBoost Regressor R^2 Score: {0:.2f}'.format(r2))

    print('Scores on Training Set:')
    print('AdaBoost Training Score: {0:.2f}'.format(adaboost_reg.score(X_train, y_train)))
    print('Scores on Test Set:')
    print('AdaBoost Test Score: {0:.2f}'.format(adaboost_reg.score(X_test, y_test)))

    return adaboost_reg

def create_xgboost_classifier(X_train, y_train, X_test, y_test):
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_predict = xgb.predict(X_test)
    print(classification_report(y_test, y_predict))
    return y_predict


def create_xgboost_classifier1(X_train, y_train, X_test, y_test):
    # # Step 1: Identify unique values in the target variable
    # unique_values = y_train.unique()
    # print("Unique values in y_train before mapping:", unique_values)
    #
    # # Step 2: Create a mapping from existing values to consecutive integers
    # value_mapping = {value: idx for idx, value in enumerate(sorted(unique_values))}
    # print("Value mapping:", value_mapping)
    #
    # # Step 3: Map the target variable to the new set of consecutive integers
    # y_train_mapped = y_train.map(value_mapping)
    # y_test_mapped = y_test.map(value_mapping)
    #
    # # Step 4: Use the mapped target variable for training the model
    # xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.01)
    # xgb.fit(X_train, y_train_mapped)
    # y_predict = xgb.predict(X_test)
    #
    # # Step 5: Map the predictions back to the original values
    # inverse_value_mapping = {v: k for k, v in value_mapping.items()}
    # y_predict_original = pd.Series(y_predict).map(inverse_value_mapping)
    #
    # print(classification_report(y_test, y_predict_original, zero_division=0))
    # print('xgb_boost_classifier Score: {0:2.2f}'.format(xgb.score(X_test, y_test)))
    # print('Scores on Training Set:')
    # print('xgb_boost_classifier Score: {0:2.2f}'.format(xgb.score(X_train, y_train_mapped)))
    # print('Scores on Test Set:')
    # print('xgb_boost_classifier Score: {0:2.2f}'.format(xgb.score(X_test, y_test)))
    # precision = precision_score(y_test, y_predict, average='macro', zero_division=0)
    # recall = recall_score(y_test, y_predict, average='macro', zero_division=0)
    # f1 = f1_score(y_test, y_predict, average='macro', zero_division=0)
    # print('xgb_boost \n precision: ', precision, '\n', 'recall: ', recall, '\n', 'f1: ', f1)
    # print('xgb accuracy_score:', accuracy_score(y_test, y_predict_original))
    # return xgb
    # Encode the target variable
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Train the model
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.01)
    xgb_model.fit(X_train, y_train_encoded)

    # Predict
    y_predict_encoded = xgb_model.predict(X_test)

    # Decode predictions
    y_predict = le.inverse_transform(y_predict_encoded)

    # Print evaluation metrics
    print(classification_report(y_test, y_predict, zero_division=0))
    print('xgb_boost_classifier Score: {0:2.2f}'.format(xgb_model.score(X_test, y_test_encoded)))
    print('Scores on Training Set:')
    print('xgb_boost_classifier Score: {0:2.2f}'.format(xgb_model.score(X_train, y_train_encoded)))
    print('Scores on Test Set:')
    print('xgb_boost_classifier Score: {0:2.2f}'.format(xgb_model.score(X_test, y_test_encoded)))
    precision = precision_score(y_test, y_predict, average='macro', zero_division=0)
    recall = recall_score(y_test, y_predict, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_predict, average='macro', zero_division=0)
    print('xgb_boost \n precision: ', precision, '\n', 'recall: ', recall, '\n', 'f1: ', f1)
    print('xgb accuracy_score:', accuracy_score(y_test, y_predict))

    return xgb_model


def create_xgb_regressor(X_train, y_train, X_test, y_test):
    # Initialize XGBRegressor
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=6)  # Adjust parameters as needed

    # Fit the model
    xgb_reg.fit(X_train, y_train)

    # Predict on the test set
    y_predict = xgb_reg.predict(X_test)

    # Print evaluation metrics for regression
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    print('XGB Regressor Mean Squared Error: {0:.2f}'.format(mse))
    print('XGB Regressor Mean Absolute Error: {0:.2f}'.format(mae))
    print('XGB Regressor R^2 Score: {0:.2f}'.format(r2))

    print('Scores on Training Set:')
    print('XGB Training Score: {0:.2f}'.format(xgb_reg.score(X_train, y_train)))
    print('Scores on Test Set:')
    print('XGB Test Score: {0:.2f}'.format(xgb_reg.score(X_test, y_test)))

    return xgb_reg

# def create_decision_tree_classifier(X_train, y_train, X_test, y_test):
#     clf_tree = tree.DecisionTreeClassifier()
#     clf_tree.fit(X_train, y_train)
#     obs_1 = np.array((4.9, 3.4, 1.4, 0.2)).reshape(1, -1)
#     clf_tree.predict(obs_1)
#
#     # visualise tree
#     dot_data = tree.export_graphviz(clf_tree, out_file=None)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_png('tree.png')
#     fig, ax = plt.subplots(figsize=(24, 24))
#     ax.imshow(plt.imread('tree.png'))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     plt.show()

def create_decision_tree_classifier1(X_train, y_train, X_test, y_test):
    clf_tree = tree.DecisionTreeClassifier(max_depth=7, criterion='entropy')
    clf_tree.fit(X_train, y_train)
    y_predict = clf_tree.predict(X_test)

    precision = precision_score(y_test, y_predict, zero_division=0, average='weighted')
    recall = recall_score(y_test, y_predict, zero_division=0, average='weighted')
    f1 = f1_score(y_test, y_predict, zero_division=0, average='weighted')

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    accuracy = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_predict, zero_division=0))
    # visualise_tree(clf_tree, X_train.columns.tolist())
    print('clfTree_classifier Score: {0:2.2f}'.format(clf_tree.score(X_test, y_test)))
    print('Scores on Training Set:')
    print('clfTree_classifier Score: {0:2.2f}'.format(clf_tree.score(X_train, y_train)))
    print('Scores on Test Set:')
    print('clfTree_classifier Score: {0:2.2f}'.format(clf_tree.score(X_test, y_test)))
    precision = precision_score(y_test, y_predict, average='macro', zero_division=0)
    recall = recall_score(y_test, y_predict, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_predict, average='macro', zero_division=0)
    print('clfTree \n precision: ', precision, '\n', 'recall: ', recall, '\n', 'f1: ', f1)
    print('clfTree accuracy_score:', accuracy_score(y_test, y_predict))

    return clf_tree


def create_decision_tree_regressor(X_train, y_train, X_test, y_test):
    # Initialize DecisionTreeRegressor
    dt_reg = DecisionTreeRegressor(max_depth=5, criterion='squared_error')  # Adjust parameters as needed

    # Fit the model
    dt_reg.fit(X_train, y_train)

    # Predict on the test set
    y_predict = dt_reg.predict(X_test)

    # Print evaluation metrics for regression
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    print('Decision Tree Regressor Mean Squared Error: {0:.2f}'.format(mse))
    print('Decision Tree Regressor Mean Absolute Error: {0:.2f}'.format(mae))
    print('Decision Tree Regressor R^2 Score: {0:.2f}'.format(r2))

    print('Scores on Training Set:')
    print('Decision Tree Training Score: {0:.2f}'.format(dt_reg.score(X_train, y_train)))
    print('Scores on Test Set:')
    print('Decision Tree Test Score: {0:.2f}'.format(dt_reg.score(X_test, y_test)))

    return dt_reg

def find_optimal_tree_depth(X_train, y_train, X_test, y_test):
    depths = range(1, 21)
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_predict)
        print(f"Accuracy for depth {depth}: {accuracy}")


def visualise_tree(clf, feature_cols):
    fig = plt.figure(figsize=(25, 20))
    _ = plot_tree(clf,
                  feature_names=feature_cols,
                  filled=True)
    plt.show()
    fig.savefig('tree.jpeg')


#
# def visualize_decision_tree(clf_tree, feature_names):
#     dot_data = tree.export_graphviz(clf_tree, out_file=None, feature_names=feature_names)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_png('tree.png')
#     fig, ax = plt.subplots(figsize=(24, 24))
#     ax.imshow(plt.imread('tree.png'))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     plt.show()


def evaluate_model(y_test, y_predict):
    y_predict_binary = [1 if pred > 0.5 else 0 for pred in y_predict]
    accuracy = accuracy_score(y_test, y_predict_binary)
    precision = precision_score(y_test, y_predict_binary)
    recall = recall_score(y_test, y_predict_binary)
    f1 = f1_score(y_test, y_predict_binary)
    roc_auc = roc_auc_score(y_test, y_predict_binary)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'ROC AUC Score: {roc_auc}')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 6 - Model Optimization
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def optimize_xgboost_classifier(X_train, y_train):
    # Define the parameter grid
    param_grid = {'n_estimators': [50, 100, 150],
                  'max_depth': [3, 5, 7],
                  'learning_rate': [0.1, 0.01, 0.001]}

    # Create an XGBoost classifier
    xgb = XGBClassifier()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and score
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)
    # print("Feature Importance Score: ", grid_search.best_estimator_.feature_importances_)
    feature_names = X_train.columns.tolist()
    feature_dict = {feature_names[i]: grid_search.best_estimator_.feature_importances_[i] for i in
                    range(len(feature_names))}
    print(feature_dict)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 7 - Clustering
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def feature_scaling(X_train):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled


def dimensional_reduction(X_train_scaled, n_components=2, use_tsne=False):
    if n_components == 2:
        if use_tsne:
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
    elif n_components == 3:
        if use_tsne:
            reducer = TSNE(n_components=3, random_state=42)
        else:
            reducer = PCA(n_components=3)
    else:
        raise ValueError("n_components must be 2 or 3 for visualization.")

    reduced_data = reducer.fit_transform(X_train_scaled)
    return reduced_data, reducer


def find_number_of_clusters(reduced_data):
    # List to store the inertia values
    inertia = []

    # Range of k values to try
    k_range = range(1, 12)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(reduced_data)
        inertia.append(kmeans.inertia_)

    # Plot the inertia as a function of k
    plt.plot(k_range, inertia, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show(),


def perform_kmeans_clustering(reduced_data, k, n_components=2, use_tsne=False):
    """
    Performs K-Means clustering on the provided dataset and visualizes the results.

    Parameters:
    df_or_x_train_scaled (DataFrame or ndarray): The dataset (scaled and encoded) for clustering.
    k (int): The number of clusters for K-Means.
    n_components (int): Number of components for dimensionality reduction (2 or 3).
    use_tsne (bool): Whether to use t-SNE for dimensionality reduction (if False, PCA will be used).

    Returns:
    cluster_labels (ndarray): Cluster labels for each data point.
    cluster_centers_df (DataFrame): DataFrame containing the cluster centers.
    """

    # Initialize K-Means with the specified number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the K-Means model
    kmeans.fit(reduced_data)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # If df_or_x_train_scaled is a DataFrame, use its column names for the cluster centers DataFrame
    if isinstance(reduced_data, pd.DataFrame):
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=reduced_data.columns)
    else:
        # If input is ndarray, column names are not available, use generic feature names
        cluster_centers_df = pd.DataFrame(cluster_centers,
                                          columns=[f'Feature_{i}' for i in range(cluster_centers.shape[1])])

    # Dimensionality Reduction for Visualization
    if n_components == 2:
        if use_tsne:
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
    elif n_components == 3:
        if use_tsne:
            reducer = TSNE(n_components=3, random_state=42)
        else:
            reducer = PCA(n_components=3)
    else:
        raise ValueError("n_components must be 2 or 3 for visualization.")

    reduced_data = reducer.fit_transform(reduced_data)

    # Plotting
    plt.figure(figsize=(10, 8))
    if n_components == 2:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=50)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'K-Means Clustering Visualization (2D) with k={k}')
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=cluster_labels,
                             cmap='viridis', s=50)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(f'K-Means Clustering Visualization (3D) with k={k}')
        plt.colorbar(scatter)

    plt.show()

    return kmeans.fit_predict(reduced_data)


def analyze_pca_components(pca_components, pca, numeric_features):
    """
    Analyzes PCA components and visualizes feature loadings.

    Parameters:
    - pca_components: np.ndarray with PCA-reduced components.
    - pca: PCA object after fitting.
    - numeric_features: list of numeric features used for PCA.

    Returns:
    - None (prints and shows results).
    """
    # Check if components_ attribute is available
    if not hasattr(pca, 'components_'):
        raise AttributeError(
            "PCA object does not have 'components_' attribute. Ensure you are using the correct version of scikit-learn.")

    # Get the loadings for each component
    loadings = pca.components_

    # Create a DataFrame with feature loadings
    loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i + 1}' for i in range(loadings.shape[0])],
                               index=numeric_features)
    print("PCA Component Loadings:")
    print(loadings_df)

    # Visualize Component Loadings
    plt.figure(figsize=(12, 6))

    for i in range(loadings.shape[0]):
        plt.subplot(1, 2, i + 1)
        sns.barplot(x=loadings_df.index, y=loadings_df[f'PC{i + 1}'])
        plt.title(f'Feature Loadings for Principal Component {i + 1}')
        plt.xticks(rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Loading')

    plt.tight_layout()
    plt.show()


def discretize_target(y, n_bins=3):
    # Discretize target variable into bins
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_discretized = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten()
    return y_discretized


def create_xgboost_classifier(X_train, y_train, X_test, y_test):
    # Train XGBClassifier
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.01)
    xgb_model.fit(X_train, y_train)

    # Predict
    y_predict = xgb_model.predict(X_test)

    # Print evaluation metrics
    print(classification_report(y_test, y_predict, zero_division=0))
    print('xgb_classifier Score: {0:2.2f}'.format(xgb_model.score(X_test, y_test)))
    print('Scores on Training Set:')
    print('xgb_classifier Score: {0:2.2f}'.format(xgb_model.score(X_train, y_train)))
    print('Scores on Test Set:')
    print('xgb_classifier Score: {0:2.2f}'.format(xgb_model.score(X_test, y_test)))
    print('Accuracy Score:', accuracy_score(y_test, y_predict))

    return xgb_model


def plot_models(rf, abc, dt, xgb, X_train, y_train):
    # Create a plot to compare the performance of the models:
    fig = plt.figure(figsize=(12, 8))
    rf_plot = plt.plot(y_train, rf.predict(X_train), 'ob',
                       label='Random Forest - {0:2.2f}'.format(rf.score(X_train, y_train)))
    abc_plot = plt.plot(y_train, abc.predict(X_train), 'or',
                       label='AdaBoost - {0:2.2f}'.format(abc.score(X_train, y_train)))
    dt_plot = plt.plot(y_train, dt.predict(X_train), 'og',
                       label='Decision Tree - {0:2.2f}'.format(dt.score(X_train, y_train)))
    xgb_plot = plt.plot(y_train, xgb.predict(X_train), 'oy',
                       label='XGB - {0:2.2f}'.format(xgb.score(X_train, y_train)))
    plt.legend(loc=2)
    plt.xlabel('Observed (True value)')
    plt.ylabel('Predicted')
    plt.show()


def all_regressors(target_var, df_encoded):
    df_without_target = df_encoded.drop(columns=[target_var])
    X_train, X_test, y_train, y_test = train_test_split(df_without_target, df_encoded[target_var],
                                                        test_size=0.3, random_state=42)
    print("random forest regressor:")
    rf = create_random_forest_regression_model(X_train, y_train, X_test, y_test)
    print("ada boost regressor:")
    abc = create_adaboost_regressor(X_train, y_train, X_test, y_test)
    print("decision tree regressor:")
    dt = create_decision_tree_regressor(X_train, y_train, X_test, y_test)
    print("xgboost regressor:")
    xgb = create_xgb_regressor(X_train, y_train, X_test, y_test)
    plot_models(rf, abc, dt, xgb, X_train, y_train)


def all_classifiers(target_var, df_encoded):
    df_encoded[target_var + '_binned'] = discretize_target(df_encoded[target_var], n_bins=5)

    df_without_target = df_encoded.drop(columns=[target_var])
    X_train, X_test, y_train, y_test = train_test_split(df_without_target, df_encoded[target_var + '_binned'],
                                                        test_size=0.3, random_state=42)

    print("XGBoost Classifier:")
    xgb_model = create_xgboost_classifier(X_train, y_train, X_test, y_test)
    # # All the classifiers:
    print("xgboost classifier:")
    xgb = create_xgboost_classifier1(X_train, y_train, X_test, y_test)
    print("decision tree classifier:")
    dt = create_decision_tree_classifier1(X_train, y_train, X_test, y_test)
    print("ada boost classifier:")
    abc = create_ada_boost_classifier1(X_train, y_train, X_test, y_test)
    print("random forest classifier:")
    rf = create_random_forest_classification_model(X_train, y_train, X_test, y_test)
    plot_models(rf, abc, dt, xgb, X_train, y_train)

def main():
    df = pd.read_csv("most_subscribed_youtube_channels.csv").copy()
    # print("data type:")
    # print(check_data_type(df))
    # print("missing values:")
    # print(find_missing_values(df))
    # fill_missing_categorical_values(df)
    # rows_with_zero_val = find_rows_with_zero_values(df)
    # print("rows with zero values:")
    # print(rows_with_zero_val)

    # rows_to_delete = find_rows_with_zero_values(df)
    # df = del_rows(df, rows_with_zero_val)
    # print("after filling rows:")
    # print(df)
    # print("missing cells:")
    # find_cells_with_missing_values(df)
    # print("missing rows:")
    # print(find_number_rows_of_missing_values(df))
    # print("statisti")
    # print(statistics_dictionary(df))

    # phase 1 - data validation
    # col_list = df.columns.tolist()
    # print(col_list)
    # Remove non-numeric columns from the list
    # non_numeric_cols = ['Youtuber', 'category']
    # numeric_cols = [col for col in col_list if col not in non_numeric_cols]
    # df_with_numeric = make_cols_numeric(df.copy(), numeric_cols)
    #
    # print("data type:")
    # print(check_data_type(df_with_numeric))
    # df_encoded = encode_data(df_with_numeric, non_numeric_cols)
    # print("data type:", check_data_type(df_encoded))
    # print("encoded data:")
    # print(df_encoded)
    # df_encoded.to_csv('youtube_encoded_data.csv', index=False)
    df_encoded = pd.read_csv("youtube_encoded_data.csv")
    target_var = 'subscribers'

    # phase 3 - linear regression
    print("\nLinear Regression!:\n")
    perform_linear_regression(df_encoded, 0.7, target_var)

    # phase 4 - logistics regression
    print("\nLogistic Regression!:\n")
    binary_tree_df = turn_col_to_binary(df_encoded, target_var)
    log_data_train, log_data_test = split_log_regression(binary_tree_df, 0.7)
    initiate_log_regression(log_data_train, log_data_test, target_var)

    # phase 3 - Model Training and Evaluation

    all_regressors(target_var, df_encoded.copy())
    all_classifiers(target_var, df_encoded.copy())
    # X_train, X_test, y_train, y_test = train_test_split(df_without_target, df_encoded[target_var + '_binned'],
    #                                                     test_size=0.3, random_state=42)

    # print("XGBoost Classifier:")
    # xgb_model = create_xgboost_classifier(X_train, y_train, X_test, y_test)
    # # df_without_target = df_encoded.drop(columns=[target_var])
    # # X_train, X_test, y_train, y_test = train_test_split(df_without_target, df_encoded[target_var],
    # #                                                     test_size=0.3, random_state=42)
    # # # All the classifiers:
    # print("xgboost classifier:")
    # xgb = create_xgboost_classifier1(X_train, y_train, X_test, y_test)
    # print("decision tree classifier:")
    # dt = create_decision_tree_classifier1(X_train, y_train, X_test, y_test)
    # print("ada boost classifier:")
    # abc = create_ada_boost_classifier1(X_train, y_train, X_test, y_test)
    # print("random forest classifier:")
    # rf = create_random_forest_classification_model(X_train, y_train, X_test, y_test)

    # # all the regressors:
    # X_train, X_test, y_train, y_test = train_test_split(df_without_target, df_encoded[target_var],
    #                                                     test_size=0.3, random_state=42)
    # print("random forest regressor:")
    # rf = create_random_forest_regression_model(X_train, y_train, X_test, y_test)
    # print("ada boost regressor:")
    # abc = create_adaboost_regressor(X_train, y_train, X_test, y_test)
    # print("decision tree regressor:")
    # dt = create_decision_tree_regressor(X_train, y_train, X_test, y_test)
    # print("xgboost regressor:")
    # xgb = create_xgb_regressor(X_train, y_train, X_test, y_test)

    # # Create a plot to compare the performance of the models:
    # fig = plt.figure(figsize=(12, 8))
    # rf_plot = plt.plot(y_train, rf.predict(X_train), 'ob',
    #                    label='Random Forest - {0:2.2f}'.format(rf.score(X_train, y_train)))
    # abc_plot = plt.plot(y_train, abc.predict(X_train), 'or',
    #                    label='AdaBoost - {0:2.2f}'.format(abc.score(X_train, y_train)))
    # dt_plot = plt.plot(y_train, dt.predict(X_train), 'og',
    #                    label='Decision Tree - {0:2.2f}'.format(dt.score(X_train, y_train)))
    # xgb_plot = plt.plot(y_train, xgb.predict(X_train), 'oy',
    #                    label='XGB - {0:2.2f}'.format(xgb.score(X_train, y_train)))
    # plt.legend(loc=2)
    # plt.xlabel('Observed (True value)')
    # plt.ylabel('Predicted')
    # plt.show()

    # phase 4 - Model Optimization

    return 0


if __name__ == '__main__':
    main()

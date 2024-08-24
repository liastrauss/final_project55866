import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning Models
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn import tree

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, KBinsDiscretizer

# Dimensionality Reduction & Manifold Learning
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Model Evaluation
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, accuracy_score, classification_report, r2_score
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics

# Statistical Models
import statsmodels.formula.api as smf

# Miscellaneous
import statistics


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


def plot_missing_and_zero_values(dataframe):
    """
    Create and plot a bar plot to visualize the number of missing and zero values for each column in the DataFrame.
    :param dataframe: The input DataFrame
    """
    # Calculate missing values
    missing_values = dataframe.isnull().sum()

    # Calculate zero values (only for numeric columns)
    zero_values = dataframe.apply(lambda x: (x == 0).sum() if pd.api.types.is_numeric_dtype(x) else 0)

    # Combine the counts into a DataFrame
    summary_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Zero Values': zero_values
    }).fillna(0)  # Replace NaNs with 0 for columns with no zeros

    # Plot the results
    summary_df.plot(kind='bar', figsize=(14, 8))
    plt.title('Number of Missing and Zero Values per Column')
    plt.xlabel('Columns')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    return summary_df


def plot_correlation_matrix(dataframe):
    """
    Compute and plot the correlation matrix for all numeric columns in the DataFrame.
    :param dataframe: The input DataFrame
    :return: A DataFrame containing the correlation matrix
    """
    # Compute the correlation matrix
    correlation_matrix = dataframe.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.yticks(rotation=0)
    plt.title('Correlation Matrix')
    plt.show()

    return correlation_matrix


def plot_all_columns_heatmap(dataframe):
    """
    Create and plot a heatmap to visualize all column values.
    :param dataframe: The input DataFrame
    """
    plt.figure(figsize=(14, 10))
    sns.heatmap(dataframe, cmap='viridis', annot=False, cbar=True, linewidths=0.5)
    plt.title('Heatmap of All Column Values')
    plt.show()


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


def turn_col_to_binary(dataframe, col_name, percentile=50):
    """
    Turn the column into a binary column, depending on whether the value is greater than the specified percentile.
    :param dataframe: The input DataFrame
    :param col_name: The column name to convert
    :param percentile: The percentile to use for dividing the data (default is 50, which is the median)
    :return: A new DataFrame with the column converted to binary
    """
    new_df = dataframe.copy()
    threshold = np.percentile(new_df[col_name], percentile)
    new_df[col_name] = (new_df[col_name] >= threshold).astype(int)
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
    lr = LogisticRegression(solver='liblinear', max_iter=10000)
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


def create_ada_boost_classifier(X_train, y_train, X_test, y_test):
    abc = AdaBoostClassifier(
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
        estimator=DecisionTreeRegressor(max_depth=5),  # Base estimator for AdaBoost
        n_estimators=50,  # Number of boosting stages
        learning_rate=0.01  # Step size for updating the weights
    )
    initial_score = cross_val_score(adaboost_reg, X_train, y_train, cv=5).mean()
    print('Initial Cross-Validation Score: ', initial_score)
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
    # print('AdaBoost accuracy_score:', accuracy_score(y_test, y_predict))

    return adaboost_reg


def create_xgboost_classifier(X_train, y_train, X_test, y_test):
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
    xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)  # Adjust parameters as needed

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
    # print('XGB accuracy_score:', accuracy_score(y_test, y_predict))

    return xgb_reg


def create_decision_tree_classifier(X_train, y_train, X_test, y_test):
    clf_tree = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy')
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
    dt_reg = DecisionTreeRegressor(max_depth=3, criterion='squared_error')  # Adjust parameters as needed

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
    # print('Decision Tree accuracy_score:', accuracy_score(y_test, y_predict))
    return dt_reg


def find_optimal_tree_depth(X_train, y_train, X_test, y_test):
    """
    Find the optimal depth for a Decision Tree Regressor model by training multiple models with different depths
    Args:
        X_train:
        y_train:
        X_test:
        y_test:

    Returns:

    """
    depths = range(1, 21)
    optimal_depth = {'depth': 0, 'accuracy': 0}
    for depth in depths:
        clf = DecisionTreeRegressor(max_depth=depth)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        accuracy = r2_score(y_test, y_predict)
        # accuracy = accuracy_score(y_test, y_predict)
        print(f"Accuracy for depth {depth}: {accuracy}")
        if accuracy > optimal_depth['accuracy']:
            optimal_depth['depth'] = depth
            optimal_depth['accuracy'] = accuracy
    return optimal_depth


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


def optimize_decision_tree_regressor(X_train, y_train):
    # Define the parameter grid
    param_grid = {'max_depth': range(1, 21)}

    # Create a DecisionTreeRegressor
    dt_reg = DecisionTreeRegressor()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=dt_reg, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and score
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)


def optimize_adaboost_regressor(X_train, y_train):
    # Define the parameter grid
    param_grid = {'n_estimators': [50, 100, 150],
                  'learning_rate': [0.1, 0.01, 0.001],
                  'estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5)]}

    # Create an AdaBoost Regressor
    adaboost_reg = AdaBoostRegressor()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=adaboost_reg, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and score
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("best Estimator: ", grid_search.best_estimator_)
    print("Best Score: ", grid_search.best_score_)
    return grid_search.best_estimator_


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 7 - Clustering
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def discretize_target(y, n_bins=3):
    """
    Discretize the target variable into bins.
    Args:
        y: y values
        n_bins: number of bins

    Returns: y_discretized values as an array

    """
    # Discretize target variable into bins
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_discretized = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten()
    return y_discretized


def plot_models(rf, abc, dt, xgb, X_train, y_train):
    """
    Plot the performance of the models.
    Args:
        rf: random forest model
        abc: adaboost model
        dt: decision tree model
        xgb: xgboost model
        X_train: x train
        y_train: y train

    Returns: none

    """
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
    plt.title('Model Performance on Training Set')
    plt.show()


def plot_feature_importance(model, column_names):
    """
    Plot the feature importance of the model.
    Args:
        model:
        column_names:

    Returns:

    """
    sort_order = np.argsort(model.feature_importances_)
    x = range(len(sort_order))
    y = model.feature_importances_[sort_order]
    y_ticks = np.array(column_names)[sort_order]
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.barh(x, y)
    plt.title(f'Feature Importance {type(model).__name__}')
    plt.xlabel('Mean effect size')
    plt.ylabel('Feature')
    ax.set_yticks(x)
    ax.set_yticklabels(y_ticks)
    plt.show()


def feature_importance_print(model, column_names):
    """
    Print the feature importance of the model.
    Args:
        model:
        column_names:

    Returns: none

    """
    sort_order = np.argsort(model.feature_importances_)[::-1]

    for i, feature_index in enumerate(sort_order):
        feature = column_names[feature_index]
        # feature = column_names
        imp = model.feature_importances_[feature_index]
        print(i)
        print('{0:d}. {1:s} Weight\t- {2:4.4f}'.format(i + 1, feature, imp))


def all_regressors(target_var, df_encoded):
    """
    Perform regression on the dataset using multiple regressors and print the results.
    Args:
        target_var: target variable
        df_encoded: dataframe with encoded data

    Returns: none

    """
    df_without_target = df_encoded.drop(columns=[target_var])
    X_train, X_test, y_train, y_test = train_test_split(df_without_target, df_encoded[target_var],
                                                        test_size=0.3, random_state=42)
    # print("random forest regressor optimal depth:")
    # find_optimal_tree_depth(X_train, y_train, X_test, y_test)
    print("\nrandom forest regressor:")
    rf = create_random_forest_regression_model(X_train, y_train, X_test, y_test)
    print("\nada boost regressor:")
    abc = create_adaboost_regressor(X_train, y_train, X_test, y_test)
    print("\ndecision tree regressor:")
    dt = create_decision_tree_regressor(X_train, y_train, X_test, y_test)
    print("\nxgboost regressor:")
    xgb = create_xgb_regressor(X_train, y_train, X_test, y_test)
    # print("\noptimize decision tree regressor:")
    # optimize_decision_tree_regressor(X_train, y_train)
    # print("\noptimize adaboost regressor:")
    # optimize_adaboost_regressor(X_train, y_train)
    plot_models(rf, abc, dt, xgb, X_train, y_train)
    plot_feature_importance(rf, df_without_target.columns.tolist())
    plot_feature_importance(abc, df_without_target.columns.tolist())
    plot_feature_importance(dt, df_without_target.columns.tolist())
    plot_feature_importance(xgb, df_without_target.columns.tolist())
    print("feature importance:")
    print("random forest regressor:")
    feature_importance_print(rf, df_without_target.columns.tolist())
    print("\nada boost regressor:")
    feature_importance_print(abc, df_without_target.columns.tolist())
    print("\ndecision tree regressor:")
    feature_importance_print(dt, df_without_target.columns.tolist())
    print("\nxgboost regressor:")
    feature_importance_print(xgb, df_without_target.columns.tolist())


def all_classifiers(target_var, df_encoded):
    """
    Perform classification on the dataset using multiple classifiers and print the results.
    Args:
        target_var: target variable
        df_encoded: dataframe with encoded data

    Returns: none

    """
    df_encoded[target_var + '_binned'] = discretize_target(df_encoded[target_var], n_bins=10)

    df_without_target = df_encoded.drop(columns=[target_var])
    X_train, X_test, y_train, y_test = train_test_split(df_without_target, df_encoded[target_var + '_binned'],
                                                        test_size=0.3, random_state=42)

    # # All the classifiers:
    print("xgboost classifier:")
    xgb = create_xgboost_classifier(X_train, y_train, X_test, y_test)
    print("decision tree classifier:")
    dt = create_decision_tree_classifier(X_train, y_train, X_test, y_test)
    print("ada boost classifier:")
    abc = create_ada_boost_classifier(X_train, y_train, X_test, y_test)
    print("random forest classifier:")
    rf = create_random_forest_classification_model(X_train, y_train, X_test, y_test)
    plot_models(rf, abc, dt, xgb, X_train, y_train)


def main():
    df = pd.read_csv("most_subscribed_youtube_channels.csv").copy()
    print("data type:")
    print(check_data_type(df))
    print("missing values:")
    print(find_missing_values(df))
    # plot_missing_and_zero_values(df)
    fill_missing_categorical_values(df)
    rows_with_zero_val = find_rows_with_zero_values(df)
    print("rows with zero values:")
    print(rows_with_zero_val)

    df = del_rows(df, rows_with_zero_val)

    find_cells_with_missing_values(df)
    print("missing rows:")
    print(find_number_rows_of_missing_values(df))
    print("statistic")
    print(statistics_dictionary(df))

    # phase 1 - data validation
    col_list = df.columns.tolist()
    print(col_list)
    # Remove non-numeric columns from the list
    non_numeric_cols = ['Youtuber', 'category']
    numeric_cols = [col for col in col_list if col not in non_numeric_cols]
    df_with_numeric = make_cols_numeric(df.copy(), numeric_cols)

    print("data type:")
    print(check_data_type(df_with_numeric))
    df_encoded = encode_data(df_with_numeric, non_numeric_cols)
    print("data type:", check_data_type(df_encoded))
    print("encoded data:")
    print(df_encoded)
    df_encoded.to_csv('youtube_encoded_data.csv', index=False)
    df_encoded = pd.read_csv("youtube_encoded_data.csv")
    target_var = 'subscribers'
    plot_correlation_matrix(df_encoded.drop(columns=['rank', 'Youtuber', 'category']))
    # # phase 3 - linear regression
    print("\nLinear Regression!:\n")
    perform_linear_regression(df_encoded, 0.7, target_var)

    # # phase 4 - logistics regression
    print("\nLogistic Regression!:\n")
    binary_tree_df = turn_col_to_binary(df_encoded, target_var, percentile=80)
    log_data_train, log_data_test = split_log_regression(binary_tree_df, 0.7)
    initiate_log_regression(log_data_train, log_data_test, target_var)

    # phase 5 - Model Training and Evaluation
    all_regressors(target_var, df_encoded.copy().drop(columns=['rank', 'Youtuber']))
    print("\nOptimize ADAboost Regressor:")  # optimize the model with the highest score
    optimize_adaboost_regressor(df_encoded.copy().drop(columns=[target_var]), df_encoded[target_var])
    all_classifiers(target_var, df_encoded.copy())
    return 0


if __name__ == '__main__':
    main()

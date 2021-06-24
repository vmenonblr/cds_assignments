import requests as req
import zipfile as zip
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

KNN_CLASSIFIER = 1
DECISION_TREE_CLASSIFIER = 2


def pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320):
    # Show more than 10 or 20 rows when a dataframe comes back.
    pd.set_option('display.max_rows', max_rows)
    # Columns displayed in debug view
    pd.set_option('display.max_columns', max_columns)

    pd.set_option('display.width', display_width)


def download_data():
    # Download the file from web
    url = "https://cdn.iisc.talentsprint.com/aiml/Experiment_related_data/B15_Data_Munging.zip"
    r = req.get(url, allow_redirects=True)
    open('./data/B15_Data_Munging.zip', 'wb').write(r.content)
    # Unzip file and extract the csv's
    with zip.ZipFile('./data/B15_Data_Munging.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')
    print("Data Downloaded successfully")
    return


def load_data():
    # Skip rows to load into the df
    df_basic = pd.read_csv("./data/Districtwise_Basicdata.csv", skiprows=1)
    df_enroll = pd.read_csv("./data/Districtwise_Enrollment_details_indicator.csv", skiprows=3)

    return df_basic, df_enroll



def integrate_data(df_b, df_e):
    df_b['unique_id'] = df_b.iloc[:, [0, 1, 3]].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1)
    df_e['unique_id'] = df_e.iloc[:, [0, 1, 3]].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1)

    # Assign index
    df_b.set_index('unique_id', inplace=True)
    df_e.set_index('unique_id', inplace=True)

    df_merge_b_e = pd.merge(df_b, df_e, on=['unique_id'], suffixes=('_b', '_e'))

    return df_merge_b_e


def clean_up_date(df_i):
    df_i.dropna(subset=['overall_lit'], inplace=True)
    # Use LabelEncoder to map values for overall_lit
    le = preprocessing.LabelEncoder()
    df_i['overall_lit_cat'] = le.fit_transform(df_i['overall_lit'])

    # Dealing with nan
    df_i['p_06_pop'].fillna(df_i['p_06_pop'].median(), inplace=True)
    df_i['p_urb_pop'].fillna(df_i['p_urb_pop'].median(), inplace=True)
    df_i['p_sc_pop'].fillna(df_i['p_sc_pop'].median(), inplace=True)
    df_i['sexratio_06'].fillna(df_i['sexratio_06'].median(), inplace=True)
    df_i['p_st_pop'].fillna(df_i['p_st_pop'].median(), inplace=True)
    df_i['Enr Govt2'].fillna(df_i['Enr Govt2'].median(), inplace=True)
    df_i['Enr Govt4'].fillna(df_i['Enr Govt4'].median(), inplace=True)
    df_i['Enr Govt6'].fillna(df_i['Enr Govt6'].median(), inplace=True)
    df_i['Enr Govt9'].fillna(df_i['Enr Govt9'].median(), inplace=True)
    df_i['Enr Pvt1'].fillna(df_i['Enr Pvt1'].median(), inplace=True)
    df_i['Enr Pvt2'].fillna(df_i['Enr Pvt2'].median(), inplace=True)
    df_i['Enr Pvt3'].fillna(df_i['Enr Pvt3'].median(), inplace=True)
    df_i['Enr Pvt5'].fillna(df_i['Enr Pvt6'].median(), inplace=True)
    df_i['Enr Pvt6'].fillna(df_i['Enr Pvt7'].median(), inplace=True)
    df_i['Enr Pvt9'].fillna(df_i['Enr Pvt9'].median(), inplace=True)
    df_i['Enr R Govt4'].fillna(df_i['Enr R Govt4'].median(), inplace=True)
    df_i['Enr R Govt9'].fillna(df_i['Enr R Govt9'].median(), inplace=True)
    df_i['Enr R Pvt9'].fillna(df_i['Enr R Pvt9'].median(), inplace=True)
    df_i['Enr Pvt7'].fillna(df_i['Enr Pvt7'].median(), inplace=True)
    df_i['Gerp Cy'].fillna(df_i['Gerp Cy'].median(), inplace=True)
    df_i['Gerup Cy'].fillna(df_i['Gerup Cy'].median(), inplace=True)
    df_i['Nerp Py2'].fillna(df_i['Nerp Py2'].median(), inplace=True)
    df_i['Nerp Py1'].fillna(df_i['Nerp Py1'].median(), inplace=True)
    df_i['Nerp Cy'].fillna(df_i['Nerp Cy'].median(), inplace=True)
    df_i['Nerup Py2'].fillna(df_i['Nerup Py2'].median(), inplace=True)
    df_i['Nerup Cy'].fillna(df_i['Nerup Cy'].median(), inplace=True)
    df_i['Enr Med1 6'].fillna(df_i['Enr Med1 6'].median(), inplace=True)
    df_i['Enr Med2 4'].fillna(df_i['Enr Med2 4'].median(), inplace=True)
    df_i['Enr Med2 5'].fillna(df_i['Enr Med2 5'].median(), inplace=True)
    df_i['Enr Med3 3'].fillna(df_i['Enr Med3 3'].median(), inplace=True)
    df_i['Enr Med3 6'].fillna(df_i['Enr Med3 6'].median(), inplace=True)
    df_i['Enr Med3 7'].fillna(df_i['Enr Med3 7'].median(), inplace=True)
    df_i['Enr Med3 1'].fillna(df_i['Enr Med3 1'].median(), inplace=True)
    df_i['Rep C3'].fillna(df_i['Rep C3'].median(), inplace=True)
    df_i['Rep C4'].fillna(df_i['Rep C4'].median(), inplace=True)
    df_i['Rep C5'].fillna(df_i['Rep C5'].median(), inplace=True)
    df_i['Rep C6'].fillna(df_i['Rep C6'].median(), inplace=True)

    # Drop columns which does not have significance
    df_i.drop(['Year_b', 'distname_b', 'overall_lit', 'Year_e', 'Statecd_e', 'State Name ',
               'distcd_e', 'distname_e', 'Gerp Py2', 'Gerp Py1', 'Gerup Py2', 'Gerup Py1', 'Nerp Py1',
               'Nerp Py2', 'Nerup Py1', 'Nerup Py2', 'Statecd_b', 'distcd_b', 'distname_e', 'statename'],
              axis=1, inplace=True)

    return df_i


def remove_Highly_Correlated(df, bar=0.9):
    # Creates correlation matrix
    corr = df.corr()

    # Set Up Mask To Hide Upper Triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    tri_df = corr.mask(mask)

    # Finding features with correlation value more than specified threshold value (bar=0.9)
    highly_cor_col = [col for col in tri_df.columns if any(tri_df[col] > bar)]
    print("length of highly correlated columns", len(highly_cor_col))

    # Drop the highly correlated columns
    reduced_df = df.drop(highly_cor_col, axis=1)
    print("shape of total data", df.shape, "shape of reduced data", reduced_df.shape)
    return reduced_df


def normalise_data(df):
    pd_set_df_view_options()
    sc = preprocessing.StandardScaler()
    df_normalised = df.copy()
    # normalise all the columns except for the result label
    sc.fit(df_normalised[df_normalised.columns[:(len(df_normalised.columns) - 1)]])
    scaled = sc.transform(df_normalised[df_normalised.columns[:(len(df_normalised.columns) - 1)]])
    df_normalised[df_normalised.columns[:(len(df_normalised.columns) - 1)]] = scaled
    return df_normalised


def split_data(df, test_sz):
    features = df.iloc[:, :(len(df.columns) - 1)]
    labels = df.iloc[:, (len(df.columns) - 1)]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42, test_size=test_sz)
    return X_train, X_test, y_train, y_test


def try_classifier(X_train, X_test, y_train, y_test, classifier_type):
    classifier = None
    if classifier_type == KNN_CLASSIFIER:
        classifier = KNeighborsClassifier(n_neighbors=3)
    elif classifier_type == DECISION_TREE_CLASSIFIER:
        classifier = tree.DecisionTreeClassifier(criterion='entropy')

    if (classifier == None):
        return -1.0

    clf = classifier.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred)


if __name__ == '__main__':
    pd_set_df_view_options()
    # download_data()
    df_b, df_e = load_data()
    # integrate the data frames
    df_integrated = integrate_data(df_b, df_e)
    # clean the data. Remove nan and null. Remove columns which are not relevant
    df_cleaned = clean_up_date(df_integrated)
    # remove correlated columns
    reduced_df = remove_Highly_Correlated(df_cleaned)
    df_normalised = normalise_data(reduced_df)
    df_train_X, df_test_X, y_train, y_test = split_data(df_normalised, 0.2)
    score = try_classifier(df_train_X, df_test_X, y_train, y_test, KNN_CLASSIFIER)
    # Decision Tree has more than 0.9 accuracy
    score = try_classifier(df_train_X, df_test_X, y_train, y_test, DECISION_TREE_CLASSIFIER)


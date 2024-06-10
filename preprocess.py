import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def load_data(file_path):
    '''
    Load data from a CSV file and return it.
    '''
    df =  pd.read_csv(file_path)
    print(df.head()) 
    return df

def handle_missing_vals(df):
    '''
    Drop rows with missing values in the 'Country' and 'City' columns.
    '''
    print("Before handling missing values:", df.shape)  # Add this line
    df.dropna(subset=['Country'], inplace=True)
    df.dropna(subset=['City'], inplace=True)
    print("After handling missing values:", df.shape)   # Add this line
    return df


''' 
Bag of Words method for numerically representing text, may use later for different dataset.

def get_bow_representation(df, text_feature):
    vectorizer = CountVectorizer()
    extracted_data = df[text_feature].tolist()

    # Fit the vectorizer to the text data and transform it into a BoW representation
    bow_matrix = vectorizer.fit_transform(extracted_data)

    # Convert the BoW matrix to a dense array for easier manipulation
    bow_matrix_dense = bow_matrix.toarray()

    # Get the feature names (vocabulary)
    feature_names = vectorizer.get_feature_names()

    return bow_matrix_dense, feature_names
'''

def drop_city_feature(df):
    '''
    City feature of dataset is not required as of now since main goal of algorithm is to predict country based off Air Quality
    '''
    df.drop(columns=['City'], inplace=True)
    return df

def encode_one_hot(df, text_feature):
    '''
    Encode a categorical feature using one-hot encoding.
    '''
    ohe = OneHotEncoder()
    ohe_transform = ohe.fit_transform(df[[text_feature]])
    return ohe_transform

def encode_ordinal(df, text_feature):
    '''
    Encode a categorical feature using ordinal encoding.
    '''
    categories = ['Very Unhealthy', 'Unhealthy', 'Unhealthy for Sensitive Groups', 'Moderate', 'Good', 'Hazardous']
    enc = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
    return enc.fit_transform(df[[text_feature]])

'''
def normalize_text(df, text_columns):
    for col in text_columns:
        df[col] = df[col].str.lower()
    return df
'''
if __name__ == '__main__':
    df = load_data('datasets/AirPollution.csv')  # Use '/' instead of '\'
    df = handle_missing_vals(df)
    df = drop_city_feature(df)

    country_encoded = encode_one_hot(df, "Country")
    aqi_cat_bow = encode_ordinal(df, "AQI Category")
    ozone_cat_bow = encode_ordinal(df, "Ozone AQI Category")
    no2_cat_bow = encode_ordinal(df, "NO2 AQI Category")
    pm_cat_bow = encode_ordinal(df, "PM2.5 AQI Category")


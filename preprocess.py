import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def load_data(file_path): 
    return pd.read_csv(file_path)

def handle_missing_vals(df):
    df.dropna(subset=['Country'])
    df.dropna(subset=['City'])

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

def handle_city_feature(df):
    df.drop(columns=['City'], inplace=True)
    return df

def normalize_text(df, text_columns):
    for col in text_columns:
        df[col] = df[col].str.lower()
    return df



    
if __name__ == '__main__':
    df = load_data('datasets\AirPollution.csv')
    normalize_text(df)
    handle_missing_vals(df)
    handle_city_feature(df)

    country_bow = get_bow_representation(df, "Country")
    aqi_cat_bow = get_bow_representation(df, "AQI Category")
    ozone_cat_bow = get_bow_representation(df, "Ozone AQI Category")
    no2_cat_bow = get_bow_representation(df, "NO2 AQI Category")
    pm_cat_bow = get_bow_representation(df, "PM2.5 AQI Category")



    
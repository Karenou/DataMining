from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names


def get_embedding_key(movie_df, user_df):
    # extract the year of movie
    movie_df["Year"] = movie_df["Title"].apply(lambda x: x[-5:-1])
    movie_df["Title"] = movie_df["Title"].apply(lambda x: x[:-6].strip())

    # for later embedding
    user_id = sorted(user_df["UserID"].unique())

    movie_id = sorted(movie_df["MovieID"].unique().tolist())
    unknown_movie_id = 0
    movie_id.append(unknown_movie_id)

    age = sorted(user_df["Age"].unique())
    occup_id = sorted(user_df["OccupationID"].unique())
    zip_code = sorted(user_df["Zip-code"].unique())
    year = ["unknown"] + sorted(movie_df["Year"].unique())
    title = ["unknown"] + sorted(movie_df["Title"].unique())
    gender = ["F", "M"]

    embed_key_dict = {
        "UserID": user_id,
        "MovieID": movie_id,
        "OccupationID": occup_id,
        "Age": age,
        "Zip-code": zip_code,
        "Gender": gender,
        "Title": title,
        "Year": year
    }

    return embed_key_dict


def prepare_data(raing, users, movies):
    data = raing.set_index("MovieID").join(
        movies.set_index("MovieID"), 
        how="left"
    )
    data = data.reset_index(drop=False).set_index("UserID").join(
        users.set_index("UserID"), 
        how="left"
    ).reset_index(drop=False)
    
    return data

def label_encoding(sparse_features, embed_key_dict, train, test, unknown_movie_id=0):
    for feat in sparse_features:
        print("embed for %s" % feat)
        lbe = LabelEncoder()
        lbe.fit(embed_key_dict[feat])
        train[feat] = lbe.transform(train[feat])
        if feat == "MovieID":
            test[feat] = test[feat].map(lambda s: unknown_movie_id if s not in lbe.classes_ else s)
        test[feat] = lbe.transform(test[feat])

    return train, test

def get_model_feature_names(train, genre_key2index, train_genre_max_len, sparse_features):
    fixlen_feature_columns = [SparseFeat(
        feat, 
        train[feat].max() + 2, 
        embedding_dim=4) for feat in sparse_features
    ]

    varlen_feature_columns = [VarLenSparseFeat(
        SparseFeat('Genres', 
                vocabulary_size=len(genre_key2index) + 1, 
                embedding_dim=4), 
        maxlen=train_genre_max_len, combiner='mean', weight_name=None)
    ]

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    return feature_names, linear_feature_columns, dnn_feature_columns

def prepare_model_input(data, feature_names, genres_list):
    model_input = {name: data[name] for name in feature_names} 
    model_input["Genres"] = genres_list
    return model_input
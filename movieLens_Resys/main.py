import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.models import DeepFM
from preprocessing import *

if __name__ == "__main__":
    movie_df = pd.read_csv("movies.csv")
    user_df = pd.read_csv("users.csv")
    rating_train = pd.read_csv("rating_train.csv")
    rating_test = pd.read_csv("rating_test.csv")

    # since the movie_id is not clean, we only keep the movie id in the train set
    movie_id_train = pd.DataFrame(rating_train["MovieID"]).drop_duplicates(["MovieID"]).set_index("MovieID")
    movie_df = movie_df.set_index("MovieID").join(movie_id_train, how="inner").reset_index(drop=False)

    # get embedding keys
    embed_key_dict = get_embedding_key(movie_df, user_df)

    # join the three dataframes
    train = prepare_data(rating_train, user_df, movie_df)
    test = prepare_data(rating_test, user_df, movie_df)

    # fill missing Year, Title, and Genres in testset as "unknown" for now
    test["Year"] = test["Year"].fillna("unknown")
    test["Title"] = test["Title"].fillna("unknown")
    test["Genres"] = test["Genres"].fillna("unknown")

    sparse_features = ["MovieID", "UserID", "OccupationID", "Gender", "Age", "Zip-code", "Title", "Year"]
    target = ['Rating']

    # label encoding
    train, test = label_encoding(sparse_features, embed_key_dict, train, test, unknown_movie_id=0)

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in genre_key2index:
                genre_key2index[key] = len(genre_key2index) + 1
        return list(map(lambda x: genre_key2index[x], key_ans))

    # preprocess the sequence feature, add unknown type for missing/unknown Genres
    # set unknown as 1 as the 0 is used for padding
    genre_key2index = {"unknown": 1}
    train_genres_list = list(map(split, train['Genres'].values.tolist()))
    train_genres_length = np.array(list(map(len, train_genres_list)))
    train_genre_max_len = max(train_genres_length)
    train_genres_list = pad_sequences(train_genres_list, maxlen=train_genre_max_len, padding='post')

    test_genres_list = list(map(split, test['Genres'].values.tolist()))
    test_genres_length = np.array(list(map(len, test_genres_list)))
    test_genre_max_len = max(test_genres_length)
    test_genres_list = pad_sequences(test_genres_list, maxlen=test_genre_max_len, padding='post')

    # get feature names and model input
    feature_names, linear_feature_columns, dnn_feature_columns = get_model_feature_names(train, genre_key2index, train_genre_max_len, sparse_features)
    train_model_input = prepare_model_input(train, feature_names, train_genres_list)
    test_model_input = prepare_model_input(test, feature_names, test_genres_list)

    # compile DeepFM model
    mode = "train"
    n_epochs = 20
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression',
                dnn_hidden_units=(256, 256, 256), l2_reg_linear=1e-4, l2_reg_embedding=1e-5, l2_reg_dnn=1e-4)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model/deep_fm_epoch_%d" % n_epochs, 
        save_weights_only=True, verbose=1
    )
    opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.8)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])

    if mode == "train":
        history = model.fit(train_model_input, train[target].values,
                            batch_size=128, epochs=n_epochs, verbose=2, validation_split=0.3, callbacks=[cp_callback])
    else:
        model.load_weights("model/deep_fm_epoch_%d" % n_epochs)
    
    # save output
    y_pred = model.predict(test_model_input)
    y_pred = y_pred.reshape(-1)
    rating_test["Rating"] = y_pred
    rating_test["Rating"] = rating_test["Rating"].round()
    output_df = rating_test[["UserID", "MovieID", "Rating"]]
    output_df.to_csv("Q5_output/Q5_output.csv", index=False, header=True)
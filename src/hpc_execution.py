import pandas as pd
import numpy as np
import nltk
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.linear_model import LogisticRegression
import os
import spacy
nlp = spacy.load('en')


# DATA LOADING FUNCTIONS

# split dataset
def split_dataset(full_data, train_ratio, validation_ratio, test_ratio):
    """
    Function that splits the dataset into train, validation, and test
    """
    random_idx = np.random.permutation(len(full_data))
    train_threshold = int(round(train_ratio * len(full_data)))
    validation_threshold = int(round((train_ratio + validation_ratio) * len(full_data)))

    train_set = full_data.iloc[random_idx[:train_threshold]]
    validation_set = full_data.iloc[random_idx[train_threshold:validation_threshold]]
    test_set = full_data.iloc[random_idx[validation_threshold:]]

    return train_set, validation_set, test_set


# load dataset
def load_datasets(load_dir="../data/kaggle_competition/", prefix="clean_kaggle_", post_fix=""):
    """
    Function that loads the dataset
    """
    train_set = pd.read_csv(os.path.join(load_dir, "{0}train{1}.csv".format(prefix, post_fix)), keep_default_na=False)
    validation_set = pd.read_csv(os.path.join(load_dir, "{0}validation{1}.csv".format(prefix, post_fix)),
                                 keep_default_na=False)
    test_set = pd.read_csv(os.path.join(load_dir, "{0}test{1}.csv".format(prefix, post_fix)), keep_default_na=False)
    return train_set, validation_set, test_set


def xy_split(df, label_col="is_duplicate"):
    """
    Function that splits a data frame into X and y
    """
    return df.drop(label_col, axis=1).as_matrix(), df[label_col]


# DATA CLEANING FUNCTIONS
def clean_str(input_str):
    """
    Helper function that converts string to ASCII
    """
    # trivial case
    if pd.isnull(input_str) or type(input_str)==np.float or type(input_str)==float:
        return ""
    # encoding
    input_str = input_str.decode('ascii', 'ignore').lower()
    return input_str

def clean_dataset(full_dataset):
    """
    Function that cleans the full dataset
    """
    full_dataset["clean_q1"] = full_dataset["question1"].apply(clean_str,1)
    full_dataset["clean_q2"] = full_dataset["question2"].apply(clean_str,1)
    col_need = ["clean_q1", "clean_q2"]
    if "is_duplicate" in full_dataset.columns:
        col_need += ["is_duplicate"]
    return full_dataset[col_need]


# FEATURE ENGINEERING FUNCTIONS
def word_overlap(row):
    """
    Function that calculates the percentage of word overlap
    """
    avg_length = float(len(row['token_1']) + len(row['token_2'])) / 2
    save_token_num = len(set(row['token_1']).intersection(set(row['token_2'])))
    return float(save_token_num) / avg_length


def sentence_similarity(row):
    """
    Function that returns the Spacy sentence similarity
    """
    return row["doc1"].similarity(row["doc2"])


def jaccard_sim(set1, set2):
    """
    Jaccard Similarity
    """
    if len(set1.union(set2)) == 0:
        return 0.0
    else:
        return float(len(set1.intersection(set2))) / len(set1.union(set2))


def jaccard_sim_unhashbale(set1, set2):
    """
    Jaccard Similarity
    """
    count = 0.0
    str_set2 = str(set2)
    for i in set1:
        if str(i) in str_set2:
            count += 1.0
    if (len(set1) + len(set2) - count) == 0:
        return 0.0
    else:
        return count / (len(set1) + len(set2) - count)


def load_embedding(
        glove_file="/scratch/ys1001/ds1003/project/data/glove.6B.50d.txt",
        line_to_load=50000):
    """
    Function that populates a dictionary with word embedding vectors
    """
    ctr = 0
    word_emb = {}
    with open(glove_file, "r") as f:
        for i, line in enumerate(f):
            contents = line.split()
            word_emb[contents[0]] = np.asarray(contents[1:]).astype(float)
            ctr += 1
            if ctr >= line_to_load:
                break
    return word_emb


glove_emb = load_embedding()


def vectorize_tokens(token_list, word_emb, dim=50):
    """
    Function that vectorize phrases from a counter
    """
    ctr = 0.0
    vec = np.zeros(dim)
    for token in token_list:
        if token in word_emb:
            vec += word_emb[token].astype(float)
            ctr += 1
    if ctr == 0:
        return vec
    else:
        return vec / float(ctr)


def emb_dist(row, embedding):
    """
    Function that calculates the euclidean distance among two embeddings
    """
    # embedding
    emb1 = vectorize_tokens(row["token_1"], embedding)
    emb2 = vectorize_tokens(row["token_2"], embedding)
    return np.linalg.norm(emb1 - emb2)


def emb_diff(row, embedding, emb_mat):
    """
    Function that calculates the euclidean distance among two embeddings
    """
    # embedding
    emb1 = vectorize_tokens(row["token_1"], embedding)
    emb2 = vectorize_tokens(row["token_2"], embedding)
    emb_mat.append(np.abs(emb1 - emb2))


def feature_engineering(df, embedding=glove_emb, normalize=False):
    """
    Feature engineering function
    """
    total_begin = time.time()

    # preprocessing #
    # tokenization
    df['token_1'] = df.apply(lambda x: nltk.word_tokenize(x["clean_q1"]), 1)
    df['token_2'] = df.apply(lambda x: nltk.word_tokenize(x["clean_q2"]), 1)
    # spacy rep
    df['doc1'] = df.apply(lambda x: nlp(unicode(x["clean_q1"], "utf-8")), 1)
    df['doc2'] = df.apply(lambda x: nlp(unicode(x["clean_q2"], "utf-8")), 1)
    # capitalized spacy rep
    df['cap_doc1'] = df.apply(lambda x: nlp(unicode(x["clean_q1"].upper(), "utf-8")), 1)
    df['cap_doc2'] = df.apply(lambda x: nlp(unicode(x["clean_q2"].upper(), "utf-8")), 1)
    # entity
    df['entity_set_1'] = df.apply(lambda x: x["cap_doc1"].ents, 1)
    df['entity_set_2'] = df.apply(lambda x: x["cap_doc2"].ents, 1)
    # name chunk
    df['noun_chunks_1'] = df.apply(lambda x: [chunk for chunk in x["cap_doc1"].noun_chunks], 1)
    df['noun_chunks_2'] = df.apply(lambda x: [chunk for chunk in x["cap_doc2"].noun_chunks], 1)

    preprocess_time = time.time()
    print("preprocessed  for {0} seconds".format(preprocess_time - total_begin))

    # length #
    df.loc[:, "len_1"] = df.apply(lambda x: len(x["token_1"]), 1)
    df.loc[:, "len_2"] = df.apply(lambda x: len(x["token_2"]), 1)
    df.loc[:, "len_diff"] = np.abs(df["len_1"] - df["len_2"])
    df.loc[:, "len_diff_percent"] = np.abs(df["len_1"] - df["len_2"]) / ((df["len_1"] + df["len_2"]) / 2)
    after_length = time.time()
    print("length fueature loaded for {0} seconds".format(after_length - preprocess_time))

    # first words match #
    df.loc[:, "first_word_q1"] = df.apply(lambda x: x["clean_q1"].split(" ")[0], 1)
    df.loc[:, "first_word_q2"] = df.apply(lambda x: x["clean_q2"].split(" ")[0], 1)
    df.loc[:, "first_word_match"] = (df["first_word_q1"] == df["first_word_q2"])
    after_first = time.time()
    print("first word feature loaded for {0} seconds".format(after_first - after_length))

    # bag of words #
    #     if tokenizer is None:
    #         bag_of_word_tokenizer = CountVectorizer(stop_words="english", max_features=top_k_word)
    #     else:
    #         bag_of_word_tokenizer = tokenizer
    #     q1_matrix = bag_of_word_tokenizer.fit_transform(df["clean_q1"]).astype(np.float)
    #     q2_matrix = bag_of_word_tokenizer.fit_transform(df["clean_q2"]).astype(np.float)
    #     df["vec_q1"] = [q1_matrix[i] for i in range(len(df))]
    #     df["vec_q2"] = [q2_matrix[i] for i in range(len(df))]
    #     print("question vectorized")


    # similarity measure #
    # cosine_sim = [cosine_similarity(q1_matrix[i], q2_matrix[i])[0][0] for i in range(len(df))]
    # df["cosine_sim"] = cosine_sim
    df.loc[:, "overlap_percent"] = df.apply(word_overlap, 1)
    # Spacy stentence similarity
    df.loc[:, "spacy_sentence_similarity"] = df.apply(sentence_similarity, 1)
    # edit distance
    df.loc[:, "edit_distance"] = df.apply(lambda x: nltk.edit_distance(x["token_1"], x["token_2"]), 1)
    # token Jaccard
    df.loc[:, "token_jaccard"] = df.apply(lambda x: jaccard_sim(set(x["token_1"]), set(x["token_2"])), 1)
    after_sim = time.time()
    print("similarity feature loaded for {0} seconds".format(after_sim - after_first))

    # embedding #
    # embedding diff -- UGLY
    dim_emb = embedding.values()[0].shape[0]
    emb_mat = []
    df.apply(lambda x: emb_diff(x, embedding, emb_mat), 1)
    emb_mat = np.array(emb_mat)
    for dim in range(dim_emb):
        df["emb_diff_dim_{0}".format(dim)] = emb_mat[:, dim]
    # euclidean distance - embedding
    df.loc[:, "emb_dist"] = df.apply(lambda x: emb_dist(x, embedding), 1)
    after_emb = time.time()
    print("embedding feature loaded for {0} seconds".format(after_emb - after_sim))

    # entity features #
    # entity same
    df.loc[:, "entity_same"] = df.apply(lambda x: x["entity_set_1"] == x["entity_set_2"], 1)
    # entity # same
    df.loc[:, "entity_len_same"] = df.apply(lambda x: len(x["entity_set_1"]) == len(x["entity_set_2"]), 1)
    # entity # diff
    df.loc[:, "entity_len_diff"] = df.apply(lambda x: np.abs(len(x["entity_set_1"]) - len(x["entity_set_2"])), 1)
    # entity Jaccard
    df.loc[:, "entity_jaccard"] = df.apply(lambda x: jaccard_sim_unhashbale(x["entity_set_1"], x["entity_set_2"]), 1)

    # noun chunk same
    df.loc[:, "chunk_same"] = df.apply(lambda x: x["noun_chunks_1"] == x["noun_chunks_2"], 1)
    # noun chunk # same
    df.loc[:, "chunk_len_same"] = df.apply(lambda x: len(x["noun_chunks_1"]) == len(x["noun_chunks_2"]), 1)
    # noun chunk # diff
    df.loc[:, "chunk_len_diff"] = df.apply(lambda x: np.abs(len(x["noun_chunks_1"]) - len(x["noun_chunks_2"])), 1)
    # noun chunk Jaccard
    df.loc[:, "chunk_jaccard"] = df.apply(lambda x: jaccard_sim_unhashbale(x["noun_chunks_1"], x["noun_chunks_2"]), 1)
    after_entity = time.time()
    print("entity feature loaded for {0} seconds".format(after_entity - after_emb))

    # filter columns
    ignore_columns = ["first_word_q1", "first_word_q2", "clean_q1", "clean_q2", "token_1", "token_2",
                      "doc1", "doc2", "cap_doc1", "cap_doc2", "noun_chunks_1", "noun_chunks_2",
                      "entity_set_1", "entity_set_2"]
    col_normalize = ['len_1', 'len_2', 'len_diff', 'edit_distance', 'emb_dist', 'entity_len_diff', 'chunk_len_diff']
    # full_feature_df = df
    clean_feature_df = df.drop(ignore_columns, axis=1)
    if normalize:
        for col in clean_feature_df.columns:
            if str(col) in col_normalize:
                col_max = np.max(clean_feature_df[col])
                col_min = np.min(clean_feature_df[col])
                clean_feature_df[col] = (clean_feature_df[col] - col_min) / float(col_max - col_min)
    after_normalize = time.time()
    print("normalization time = {0}".format(time.time() - after_normalize))
    print("total time = {0}".format(time.time() - total_begin))
    return clean_feature_df


train_set, validation_set, test_set = load_datasets(load_dir="/scratch/ys1001/ds1003/project/data/")

# Execution Script
begin_time = time.time()
feature_train = feature_engineering(train_set)
feature_train.to_csv("/scratch/ys1001/ds1003/project/data/feature_train_v2.csv", index=False)
feature_validation = feature_engineering(validation_set)
feature_validation.to_csv("./scratch/ys1001/ds1003/project/data/feature_validation_v2.csv", index=False)
feature_test = feature_engineering(test_set)
feature_test.to_csv("/scratch/ys1001/ds1003/project/data/feature_test_v2.csv", index=False)
print("data featurized, used {0} seconds".format(time.time()-begin_time))
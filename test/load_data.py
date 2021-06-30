import pandas as pd
import scipy
import pickle

# load csv files
"""
name, n_steps, n_ingredients, steps, ingredients
"""
df_train = pd.read_csv("datasets/recipe_train.csv")
df_test = pd.read_csv("datasets/recipe_test.csv")

"""
# load CountVectorizer (pkl) files
"""
#This file contains the CountVectorizer extracted using the text of the recipe "name", "ingr" and "steps" in the training set.
"""
vocab_name_train = pickle.load(open("datasets/recipe_text_features_countvec/train_name_countvectorizer.pkl", "rb"))
vocab_steps_train = pickle.load(open("datasets/recipe_text_features_countvec/train_steps_countvectorizer.pkl", "rb"))
vocab_ingr_train = pickle.load(open("datasets/recipe_text_features_countvec/train_ingr_countvectorizer.pkl", "rb"))
vocab_name_dict_train = vocab_name_train.vocabulary_
vocab_steps_dict_train = vocab_steps_train.vocabulary_
vocab_ingr_dict_train = vocab_ingr_train.vocabulary_


# load sparse matrix (npz) files
"""
#This file contains a sparse matrix of the Bag-of-Word representation of the recipe names for test data. 
"""
## train
### The dense version of this matrix should be [40000 * size of vocabulary], and 
### the element (i,j) in the matrix is the count of each vocabulary term j in instance i. 
### The vocabulary corresponds to the vocabulary_ attribute of vocab (which can be checked as detailed in (1))
arr_name_vec_train = scipy.sparse.load_npz("datasets/recipe_text_features_countvec/train_name_vec.npz").toarray()
arr_steps_vec_train = scipy.sparse.load_npz("datasets/recipe_text_features_countvec/train_steps_vec.npz").toarray()
arr_ingr_vec_train = scipy.sparse.load_npz('datasets/recipe_text_features_countvec/train_ingr_vec.npz').toarray()
## test
### The dense version of this matrix should be [10000 * size of vocabulary]. 
### The vocabulary is the one that has been extracted from training, but 
### the elements in this matrix are the counts for each recipe in the test set.
arr_name_vec_test = scipy.sparse.load_npz("datasets/recipe_text_features_countvec/test_name_vec.npz").toarray()
arr_steps_vec_test = scipy.sparse.load_npz("datasets/recipe_text_features_countvec/test_steps_vec.npz").toarray()
arr_ingr_vec_test = scipy.sparse.load_npz("datasets/recipe_text_features_countvec/test_ingr_vec.npz").toarray()


# load Doc2Vec50 matrix files
""" 
#This file contains a matrix of Doc2Vec representation of the recipe names for training data, with 50 features.
#The element (i,j) in the matrix is a numeric value for feature j of an instance i. 
"""
## train
### The dimension of this matrix is [40000 * 50]
df_name_doc2vec50_train = pd.read_csv(r"datasets/recipe_text_features_doc2vec50/train_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
df_steps_doc2vec50_train = pd.read_csv(r"datasets/recipe_text_features_doc2vec50/train_steps_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
df_ingr_doc2vec50_train = pd.read_csv(r"datasets/recipe_text_features_doc2vec50/train_ingr_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
## test
### The dimension of this matrix is [10000 * 50]
df_name_doc2vec50_test = pd.read_csv(r"datasets/recipe_text_features_doc2vec50/test_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
df_steps_doc2vec50_test = pd.read_csv(r"datasets/recipe_text_features_doc2vec50/test_steps_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
df_ingr_doc2vec50_test = pd.read_csv(r"datasets/recipe_text_features_doc2vec50/test_ingr_doc2vec50.csv", index_col = False, delimiter = ',', header=None)


# load Doc2Vec100 matrix files
"""
#Same as Doc2Vec50 but 100 features are used for each instance
"""
## train
df_name_doc2vec100_train = pd.read_csv("datasets/recipe_text_features_doc2vec100/train_name_doc2vec100.csv", index_col = False, delimiter = ',', header=None)
df_steps_doc2vec100_train = pd.read_csv("datasets/recipe_text_features_doc2vec100/train_steps_doc2vec100.csv", index_col = False, delimiter = ',', header=None)
df_ingr_doc2vec100_train = pd.read_csv("datasets/recipe_text_features_doc2vec100/train_ingr_doc2vec100.csv", index_col = False, delimiter = ',', header=None)
## test
df_name_doc2vec100_test = pd.read_csv("datasets/recipe_text_features_doc2vec100/test_name_doc2vec100.csv", index_col = False, delimiter = ',', header=None)
df_steps_doc2vec100_test = pd.read_csv("datasets/recipe_text_features_doc2vec100/test_steps_doc2vec100.csv", index_col = False, delimiter = ',', header=None)
df_ingr_doc2vec100_test = pd.read_csv("datasets/recipe_text_features_doc2vec100/test_ingr_doc2vec100.csv", index_col = False, delimiter = ',', header=None)

"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skmultiflow.drift_detection import ADWIN, DDM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from tqdm.notebook import tqdm
from random import shuffle
from time import time
import pandas as pd
import numpy as np
import pickle
import sys
import gc
import os
import graphviz



data_path = '/home/maiabboud/Documents/RACHIS/GP_Combiner/GP_combiner_random_feature/data'
code_path = '/home/maiabboud/Documents/RACHIS/GP_Combiner/GP_combiner_random_feature/Shared Codes'
results_path = '/home/maiabboud/Documents/RACHIS/GP_Combiner/GP_combiner_random_feature/Results'
evolving_path = '/home/maiabboud/Documents/RACHIS/GP_Combiner/GP_combiner_random_feature/Evolving'
sys.path.insert(0,code_path)
from genetic_programming import SymbolicRegressor,SymbolicClassifier
from binirizer import CustomLabelBinirizer
from ensemble import Ensemble, Classifier
from oselm import OSELMClassifier,set_use_know


def prepare_data(csv_filename, target_column_name='class'):
    # read csv file
    df = pd.read_csv(csv_filename)
    df = df.iloc[:80000, :]
    column_names = df.columns.tolist()
    if target_column_name not in column_names:
        target_column_name = column_names[-1]
    # get unique value in target column
    unique_vlaues = sorted(df[target_column_name].unique().tolist())
    df[target_column_name] = df[target_column_name].apply(lambda x: 0 if x == unique_vlaues[0] else 1)
    df[target_column_name] = df[target_column_name].astype('int')
    # rename the column of the dataframe
    num_of_columns = len(column_names)
    df.columns = list(range(num_of_columns))
    return df
  
  
  
def train_and_test(model, X_train, y_train, X_test, y_test, unselected_features=None):
    model.fit(X_train, y_train, unselected_features)
    y_pred = model.predict(X_test)
    model.fit(X_test, y_test, unselected_features)
    return model, y_pred
  
def feature_evolving(evolving_matrix):
    """
    evolving_matrix : list of random list
    """
    random_index = np.random.randint(0, len(evolving_matrix), 1)[0]
    return evolving_matrix[random_index]
  
def save_pickle(obj, file_name):
  with open(file_name, 'wb') as f:
    pickle.dump(obj, f)
def load_pickle(file_name):
  with open(file_name, 'rb') as f:
    d = pickle.load(f)
  return d


def save_object(obj, filename,path):
    """
    _ INPUT (obj) THE OBJECT WE NEED SAVW IT (filename) THE NAME OF OBJECT
    """
    filename = os.path.join(path,filename)
    with open(filename+".pkl", 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()
def load_object(filename,path):
    """
    _ INPUT THE NAME OF OBJECT WE NEED LOAD IT
    """
    filename = os.path.join(path,filename)
    with open(filename+".pkl", 'rb') as outp:
        loaded_object = pickle.load(outp)
    outp.close()
    return loaded_object
  
  
def generate_new_samples(buffer, y_values, n=500, y_col='label'):
    if not y_col in buffer.columns.tolist():
      y_col = buffer.columns.tolist()[-1]
    if y_values.sum() == 0:
       return buffer[buffer[y_col] == 1].sample(n, random_state=41)[:, :-1].values, np.array([1] * n)
    else:
      return buffer[buffer[y_col] == 0].sample(n,random_state=41)[:, :-1].values, np.array([0] * n)
    
    
def genetic_programming():
    return SymbolicRegressor(population_size=10,
            generations=5, stopping_criteria=0.85,
            p_crossover=0.7, p_subtree_mutation=0.1,
            p_hoist_mutation=0.05, p_point_mutation=0.1,
            max_samples=0.7, verbose=0,
            parsimony_coefficient=1e-4, random_state=42,
            function_set=['avg2', 'avg3', 'avg5','median3', 'median5', 'maximum2', 'maximum3', 'maximum5'],
            metric='f1-score',
            # n_jobs=-1
            )
    
def create_list_of_oselm_models(number_of_hidden_neurons):
    models= [OSELMClassifier(number_of_hidden_neurons, 'relu', binarizer=CustomLabelBinirizer(), random_state=42),
             OSELMClassifier(number_of_hidden_neurons, 'relu', binarizer=CustomLabelBinirizer(), random_state=42),
             OSELMClassifier(number_of_hidden_neurons, 'relu', binarizer=CustomLabelBinirizer(), random_state=42),
             OSELMClassifier(number_of_hidden_neurons, 'relu', binarizer=CustomLabelBinirizer(), random_state=42)
             ]
    return models
  
  
def generate_oselm_models(number_of_hidden_neurons, apply_model_replacement=False):
    models= [OSELMClassifier(number_of_hidden_neurons, 'relu', binarizer=CustomLabelBinirizer(), random_state=42),
             OSELMClassifier(number_of_hidden_neurons, 'relu', binarizer=CustomLabelBinirizer(), random_state=42),
             OSELMClassifier(number_of_hidden_neurons, 'relu', binarizer=CustomLabelBinirizer(), random_state=42),
             OSELMClassifier(number_of_hidden_neurons, 'relu', binarizer=CustomLabelBinirizer(), random_state=42)
             ]

    ensemble = Ensemble(classifiers=models, program=genetic_programming(), apply_model_replacement=apply_model_replacement)
    return ensemble

def generate_ml_models(number_of_hidden_neurons, apply_model_replacement=False):
    models = [
              KNeighborsClassifier(5),
              KNeighborsClassifier(5),
              LogisticRegression(),
              LogisticRegression(),
              GaussianNB(),
              GaussianNB(),
              GaussianNB(),
              ]
    ensemble = Ensemble(classifiers=models, program=genetic_programming(), apply_model_replacement=apply_model_replacement)
    return ensemble
  
  
def load_best_mask_dfs(chunk_number,DFS_results_path):
  # best_mask_, average_mask_, random_forest_mask_, single_agent_mask_, softmax_mask_
  x = load_object("best_mask__"+str(chunk_number),DFS_results_path)
  if sum(x)==1 or sum(x)==0:x[0]=1
  if sum(x)==len(x) or sum(x)==len(x)-1:x[0]=0
  return x

def concept_drift_detection(drift_detection_obj, sample) -> bool:
    drift_detection_obj.add_element(sample)
    return drift_detection_obj.detected_change()
  
def random_forest_feature_selection(X, y):
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100,random_state=0))
    sel.fit(X, y)
    return sel.get_support()
  
def E2SC4ID(X,y,ensemble,drift_detection_obj):
  y_pred = ensemble.global_support_degree(X)
  if y is not None:
    drift = concept_drift_detection(drift_detection_obj, int(y!=y_pred))
    return drift
  return False

def E2SC4ID_STREAM(ensemble, X, y, unselected_features, drift_detection_obj,
                   chunk_number, result_save_path_data,maxC=4,train_size=0.8,drift=False,transfer_learning=True):
    if chunk_number==1 or not transfer_learning:
      ensemble.fit(X, y)
      return ensemble,drift_detection_obj
    if drift:
      drift_detection_obj.reset()
      X_train,X_valid,y_train,y_valid = train_test_split(X, y, random_state=42, train_size=train_size)

      new_models = ensemble.classifier_induction([model for model in create_list_of_oselm_models(number_of_hidden_neurons=X.shape[1]*3 // 2)]
                                                  ,X_train,y_train,unselected_features)
      if len(ensemble.classifiers) >= maxC:
          ensemble.model_replacement('time')
      ensemble.update_program(X_valid, y_valid)
    else:
      ensemble.fit(X, y)
    return ensemble, drift_detection_obj

def main(f_name, generate_model, train_size=0.8,apply_model_replacement=False,transfer_learning=False,
         feature_selection=[], result_save_path="",DFS_results_path='',ChunkNumber=0,is_synthetic=True,chunk_size = 500, mode_swiching=''):
  datasets,results = {},{}
  d = prepare_data(f_name)
  d = d.sample(frac=1, random_state=42)
  buffer = d.sample(n=5000)
  d.reset_index(inplace=True)
  d.replace([np.inf], 0, inplace=True)
  datasets[f_name.split('/')[-1]] = d
  # drift_locations_in_all_dataset = {}
  drift_locations = np.array([12500, 25000, 37500])//chunk_size
  for key in tqdm(datasets.keys()):
    drift_detection_obj = ADWIN()
    ensemble = None
    result_save_path_data = os.path.join(result_save_path, key)
    drift_location = {}
    prediction_times = {}
    memory_reduction = {}

    results[key] = {'model_result': []}
    data = datasets[key].values
    X, Y = data[:, 0:-1], data[:, -1].astype('int')
    number_of_chunk = X.shape[0]//chunk_size
    if not os.path.exists("{}_evolving_matrix.pkl".format(os.path.join(evolving_path, key))):
      a2 = np.random.randint(low=0, high=X.shape[1], size = X.shape[1] // 6).tolist()
      a3 = np.random.randint(low=0, high=X.shape[1], size = X.shape[1] // 5).tolist()
      a4 = np.random.randint(low=0, high=X.shape[1], size = X.shape[1] // 4).tolist()
      evolving_matrix = [a2, a3, a4]
      save_pickle(evolving_matrix, "{}_evolving_matrix.pkl".format(os.path.join(evolving_path, key)))
    else:
      evolving_matrix = load_pickle("{}_evolving_matrix.pkl".format(os.path.join(evolving_path, key)))
    ensemble = generate_model(number_of_hidden_neurons=X.shape[1]*3 // 2, apply_model_replacement=apply_model_replacement)
    chunks_features = np.array_split(X, number_of_chunk)
    chunks_labels = np.array_split(Y, number_of_chunk)
    print("===================== dataset : {} ======================".format(key))
    chunk_number = 1
    for CN,chunk_X, chunk_Y in tqdm(zip([*range(len(chunks_labels))],chunks_features, chunks_labels)):
      drift = False
      if is_synthetic:
        drift = True if chunk_number in drift_locations else False
      else:
        if chunk_number > 1:
          for i in tqdm(range(len(chunk_X))):
            if drift:
              drift_location[chunk_number] = 'drift'
              break
            x, y_true = chunk_X[i], chunk_Y[i]
            drift = E2SC4ID(x,y_true,ensemble=ensemble,drift_detection_obj=drift_detection_obj)
      try:
        chunk_X, chunk_Y = SMOTE(random_state=0).fit_resample(chunk_X, chunk_Y)
      except:
        if chunk_Y.sum() in [0, 1]:
          new_samples, new_labels = generate_new_samples(buffer, chunk_Y)
          chunk_X = np.concatenate((chunk_X, new_samples))
          chunk_Y = np.concatenate((chunk_Y, new_labels))
      gc.collect()
      unselected_feautres = None
      selected = None
      X_train, X_test, y_train, y_test = train_test_split(chunk_X, chunk_Y, random_state=42, train_size=train_size)
      print("Without Any FS")
      selected = None
      unselected_feautres = None

      selected = [bool(bit) for bit in selected] if not selected is None else None
      if not os.path.exists(result_save_path_data):
        os.mkdir(result_save_path_data)
      if transfer_learning:
        if not selected is None:temp = np.squeeze(X_train[:, selected]) if len(list(X_train[:, selected].shape))>2 else X_train[:, selected]
        else:temp = X_train

        ensemble,drift_detection_obj = E2SC4ID_STREAM(ensemble=ensemble, X=temp, y=y_train, unselected_features=None,drift_detection_obj=drift_detection_obj,
                                  chunk_number=chunk_number, result_save_path_data=result_save_path_data,drift=drift,transfer_learning=transfer_learning)
        if not selected is None:temp = np.squeeze(X_test[:, selected]) if len(list(X_test[:, selected].shape))>2 else X_test[:, selected]
        else:temp = X_test
        ensemble.evaluate(temp, y_test, chunk_number)
      else:
        init_ensemble=generate_model(number_of_hidden_neurons=X.shape[1]*3 // 2,apply_model_replacement=apply_model_replacement)
        if not ensemble is None:
          init_ensemble.set_scores(ensemble.get_scores())
        ensemble,drift_detection_obj = E2SC4ID_STREAM(ensemble=init_ensemble,X=X_train, y=y_train, unselected_features=unselected_feautres,drift_detection_obj=drift_detection_obj,
                                  chunk_number=chunk_number,result_save_path_data=result_save_path_data,drift=drift,transfer_learning=transfer_learning)
        ensemble.evaluate(X_test, y_test, chunk_number)


      if not selected is None and any(selected) is False:selected[0]=True
      if not selected is None:temp = np.squeeze(X_test[:, selected]) if len(list(X_test[:, selected].shape))>2 else X_test[:, selected]
      else:temp = X_test
      start_time = time()
      y_pre = ensemble.predict(temp)
      end_time = time()
      prediction_time = end_time - start_time
      prediction_times[chunk_number] = prediction_time
      memory_reduction[chunk_number] = temp.shape[1]
      
      results[key][chunk_number] = {"y_true" : y_test, "y_pred": y_pre}
      results[key]['model_result'].append(ensemble.get_scores())
      if transfer_learning:
          ensemble.fit(temp, y_test, None)
      chunk_number += 1
      # drift_locations_in_all_dataset[key] = drift_location

      save_pickle(drift_location, os.path.join(result_save_path_data, "{}_drift_location.pkl".format(key)))
      save_pickle(ensemble, os.path.join(result_save_path_data, "{}_ensemble.pkl".format(key)))
      save_pickle(results, os.path.join(result_save_path_data, "{}_results.pkl".format(key)))
      save_pickle(memory_reduction, os.path.join(result_save_path_data, "{}_memory_reduction.pkl".format(key)))
      save_pickle(prediction_times, os.path.join(result_save_path_data, "{}_prediction_times.pkl".format(key)))
      # save_pickle(drift_locations_in_all_dataset, os.path.join(result_save_path_data, "{}_drift_locations_in_all_dataset.pkl".format(key)))

    
    # for i, program in enumerate(ensemble.program_history):
    #   dot_data = program._program.export_graphviz()
    #   graph = graphviz.Source(dot_data)

    #   graph.render(f'images/ex4_tree_{i}', format='png', cleanup=True)
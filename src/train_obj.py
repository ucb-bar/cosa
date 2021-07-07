import pathlib
import time
import pickle

import sklearn
import sklearn.ensemble
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt

import utils
from cosa_input_objs import Arch

def rf_trainer(train_features, train_labels, test_features, test_labels):
    # Instantiate model with 1000 decision trees
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators = 1000, verbose=False, random_state=0, n_jobs = -1)
    # Train the model on training data
    start_time = time.time()
    rf.fit(train_features, train_labels)
    train_time = time.time() - start_time
    # print('random forest training time: {:.1f}s'.format(train_time))
    train_predictions = rf.predict(train_features)
    test_predictions = rf.predict(test_features)

    print('random forest train score: ', rf.score(train_features, train_labels))
    print('random forest train % error: ', '{:.3e}'.format(sklearn.metrics.mean_absolute_percentage_error(train_predictions, train_labels)))
    print('random forest test score: ', rf.score(test_features, test_labels))
    print('random forest test % error: ', '{:.3e}'.format(sklearn.metrics.mean_absolute_percentage_error(test_predictions, test_labels)))

    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

def rf_feature_importances():
    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)

    feature_names = get_feature_names()
    feature_names = np.array(feature_names)

    tree_feature_importances = rf.feature_importances_
    assert len(feature_names) == len(tree_feature_importances)
    sorted_idx = tree_feature_importances.argsort() # Sort in descending order

    num_top_features = 11 # 11 features currently
    top_idx = sorted_idx[-num_top_features:]

    y_ticks = np.arange(0, num_top_features)
    fig, ax = plt.subplots()
    ax.barh(y_ticks, tree_feature_importances[top_idx])
    ax.set_yticklabels(feature_names[top_idx])
    ax.set_yticks(y_ticks)
    ax.set_title('Random Forest Feature Importances (MDI)')
    plt.savefig('features.png', bbox_inches='tight')

def rf_worst_points(test_features, test_labels):
    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)

    test_predictions = rf.predict(test_features)
    preds = np.array(test_predictions)
    labels = np.array(test_labels)

    errors = np.abs((np.log(preds) - np.log(labels)) / np.log(labels))
    sorted_idx = errors.argsort()
    top_idx = sorted_idx[-1:-11:-1] # Get the top 10 errors

    feature_names = get_feature_names()
    for idx in top_idx:
        point = {feature_names[i]:test_features[idx][i] for i in range(len(test_features[idx]))}
        print(f"Bad test point: {point}, label: {test_labels[idx]:.3e}, prediction:, {test_predictions[idx]:.3e}, index: {idx}")
    
    # Random points, for comparison
    for idx in range(10):
        point = {feature_names[i]:test_features[idx][i] for i in range(len(test_features[idx]))}
        print(f"Random test point: {point}, label: {test_labels[idx]:.3e}, prediction:, {test_predictions[idx]:.3e}, index: {idx}")

def get_feature_names():
    feature_names = [
        "arith_instances",
        "mesh_x",
        "reg_instances",
        "accbuf_instances",
        "weightbuf_instances",
        "inputbuf_instances",
        "globalbuf_instances",
        "dram_instances",
        "reg_entries",
        "accbuf_entries",
        "weightbuf_entries",
        "inputbuf_entries",
        "globalbuf_entries",
    ]
    return feature_names

def get_arch_feat(arch):
    arch_path = pathlib.Path(f'/scratch/charleshong/matchlib/cmod/unittests/HybridRouterTopMeshTCGB/timeloop_configs/gen_arch/{arch}.yaml').resolve()
    arch = Arch(arch_path)

    arch_dict = utils.parse_yaml(arch_path)

    arch_feat = []
    arch_feat.append(arch_dict["arch"]["arithmetic"]["instances"]) # MAC instances
    arch_feat.append(arch_dict["arch"]["arithmetic"]["meshX"]) # meshX (one value for all)
    arch_feat.extend(arch.mem_instances) # mem instances (list)
    arch_feat.extend(arch.mem_entries) # mem entries (list)

    return np.array(arch_feat).flatten()

def trainer(dataset_path, test_ratio=0.1, algo='rf', target='cycle', model='resnet50'):
    db = utils.parse_json(dataset_path)

    features = []
    labels = []
    for arch in db:
        arch_feat_arr = get_arch_feat(arch)

        # Construct X vector
        x = arch_feat_arr

        # Get Y data point
        y = db[arch][model][target]

        features.append(x)
        labels.append(y)

    features, labels = sklearn.utils.shuffle(features, labels, random_state=0)
    
    n_samples = len(features)
    x_len = features[0].shape
    print(f"n_samples: {n_samples}, feat_len: {x_len}")
    num_test_data = int(test_ratio * len(features))

    train_features = features[:-num_test_data]
    train_labels = labels[:-num_test_data]

    test_features = features[-num_test_data:]
    test_labels = labels[-num_test_data:]
    
    # Normalize (min-max scaling)
    # norm = sklearn.preprocessing.MinMaxScaler().fit(train_features)
    # train_features = norm.transform(train_features)
    # test_features = norm.transform(test_features)

    if algo == 'rf':
        rf_trainer(train_features, train_labels, test_features, test_labels)
        rf_feature_importances()
        # rf_worst_points(test_features, test_labels)

if __name__ == "__main__":
    dataset_path = pathlib.Path("all_arch.json").resolve()
    trainer(dataset_path, test_ratio=0.1, algo='rf', target='cycle', model='resnet50')
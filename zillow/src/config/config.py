DATA_PATHS = {
    'train_2016_v2': 'C:/Users/Lenovo/Downloads/all/train_2016_v2.csv',
    'properties': 'C:/Users/Lenovo/Downloads/all/properties_2016.csv',
    'sample': 'C:/Users/Lenovo/Downloads/all/sample_submission.csv'
}

GRID_PARAM_RF = {# Number of trees in random forst
                'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 300, num = 12)],
                # Number of features to consider at every split
                'clf__max_features': ['auto', 'sqrt'],
                # Maximum number of levels in tree
                'clf__max_depth' : [int(x) for x in np.linspace(5, 30, num = 6)],
                #max_depth.append(None)
                # Minimum number of samples required to split a node
                'clf__min_samples_split' : np.arange(2, 5, 10),
                # Minimum number of samples required at each leaf node
                'clf__min_samples_leaf' : np.arange(1, 2, 4)
                # Method of selecting samples for training each tree
                #'bootstrap' : bstrap
}

FEATURES = {
    'rf': {
        'continuous': [
            'bedrm_count',
            'square_footage',
        ],
        'categorical': []
    }
}

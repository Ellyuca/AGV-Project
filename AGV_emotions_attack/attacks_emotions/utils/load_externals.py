import sys, os

external_libs = {"Keras-deep-learning-models": "models/keras_models",
                }

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

for lib_name, lib_path in external_libs.items():
    lib_path = os.path.join(project_path, lib_path)

    if lib_name == 'Keras-deep-learning-models':
        lib_token_fpath = os.path.join(lib_path, '__init__.py')
        if not os.path.isfile(lib_token_fpath):
            open(lib_token_fpath, 'a').close()

    
    sys.path.append(lib_path)
    print("Located %s" % lib_name)
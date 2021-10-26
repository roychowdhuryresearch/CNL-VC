##### Data preparision

* Put the units file in the data forder
* change working dir in `data/__init__.py`
* run `script/clean_data_loader.m`
* run `script/load_prepare_clean_data.py` to using clean_data and clean_data2 to genretate `features_mat_clean`and`features_mat_clean2` etc
* run `neural_correlation/split_cnn_feature.py` to generate data for training LSTM and CNN

##### Train LSTM  and CNN

run `neural_correlation/cnn_train.py` using `multiple_train()` --> Kfold LSTM or CNN training

##### Important region

run `generate_channel_data.m`

run `neural_correlation/knockout_refactor.py` multirun by changing the folderdir

##### Memtest

run `neural_correlation/split_neuron_image_only.py` to generate mentest data

run `neural_correlation/memtest_analysis.py`generate meninference

run `neural_correlation/memtest_analysis.py`

##### Important retrain

##### Figure generate

run `experiment_scripts/parse_neural_mircowire_label.py`

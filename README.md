# NeuroIDS
idsdb.py handles the logic of preprocessing and seperaring the data into val train and test. This uses a .h5 db that is only included. The data in the .h5 is preprocessed packets already split into the 31 input vector and truncated to attack sequence length with padding. 

generate_numpy_dataset.py runs the logic of idsdb.py

ids_driver.py contains code to work with TennLab framework and save files as they are run through framework

neuroids.py contains the training and evaluating logic builing upon an established TennLab class

In order to run user would need the .h5 and TennLab framework with Capsian config file

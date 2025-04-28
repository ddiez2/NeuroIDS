from idsdb import IDSDataset


p = "../hd5f/consolidated_data.h5"
idsdb_dataset = IDSDataset(p, debug=True)
idsdb_dataset.save_pickle('../data')

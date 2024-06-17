# Trans2Graph

Source codes of paper _Trans2Graph: Mining Ethereum Phishers with Graph on Dynamic Heterogeneous Transaction Data_
1. Download the data (https://www.kaggle.com/competitions/forta-protect-web3/data) to the folder './kaggle_data/', decompress the data files.
2. Run log_decoder.py with parameter 'train' and 'test' to obtain the token transfer events.
3. Run data_process.ipynb to obtain the processed transaction records with transaction types.
4. Create a folder './graph_data/raw/'
5. Run train.py to train and test the model, with the randomly divided data split in 'train_index'. The final model will be saved in 'models/split{}/trans2graph/'.




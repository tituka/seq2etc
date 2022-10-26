# seq2etc

After installing requirements, seq_run will process the data file and train the model with either default hyperparameters and in train mode, but this can be modified with argument
'-max_len', type=int, default=200)
'-vocab_size', type=int, default=20000
'-batch_size', type=int, default=100
'-layer_num', type=int, default=3
'-hidden_dim', type=int, default=1000
'-nb_epoch', type=int, default=20
'-mode'=default = 'train'

Selecting 'test' mode instead of 'train' will evailaute the model on the validation set, 

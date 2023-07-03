load dataset: 
  python load_RBNS_binary.py --each_class_train_size [number of one class for training] --data_dir [save dataset to]

test psam model:
  python test_psam_motif.py --bs [bs] --data_dir [path to dataset]

train raw data model:
  python train_binary_motif.py --bs 500  --model_path binary_model_window_10  --data_dir tiny_binary_dataset --n_epochs 10 --lr 0.1 
      --l 64 --protein_test 47 --window_size 10 --save False

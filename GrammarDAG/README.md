```
python main_new.py --dataset ../datasets/dataset_dummy.csv --target dynamic_viscosity --feat_arch MPN
python main_new.py --dataset ../datasets/dataset_dummy.csv --target dynamic_viscosity --feat_arch GNN
```
Optional args: `--with_condition`, `--condition_names` (can be multiple), `--testdata_idx_file`
```
python main_new.py --dataset ../datasets/dataset_dummy.csv --target dynamic_viscosity --with_condition --condition_names temperature --feat_arch GNN
python main_new.py --dataset ../datasets/dataset_dummy.csv --target dynamic_viscosity --with_condition --condition_names temperature --feat_arch MPN
python main_new.py --dataset ../datasets/dataset_dummy.csv --target dynamic_viscosity --with_condition --condition_names temperature pressure --feat_arch MPN
python main_new.py --dataset ../datasets/dataset_dummy.csv --target dynamic_viscosity --with_condition --condition_names temperature pressure --feat_arch GNN
python main_new.py --dataset ../datasets/dataset_dummy.csv --target dynamic_viscosity --with_condition --condition_names temperature pressure --feat_arch GNN --testdata_idx_file ../datasets/testdata_idx_dummy.txt
```

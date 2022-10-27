
# structured_noise

## Experiments

### Single clients.
```main.py --nb_clients 1 --dataset_size 10000000 --use_ortho_matrix False```
```main.py --nb_clients 1 --dataset_size 10000000 --use_ortho_matrix True```

### Multi clients.
```main.py --nb_clients 1 --dataset_size 10000000 --use_ortho_matrix True --heterogeneity="homog```
```main.py --nb_clients 1 --dataset_size 10000000 --use_ortho_matrix True --heterogeneity="sigma```
```main.py --nb_clients 1 --dataset_size 10000000 --use_ortho_matrix True --heterogeneity="wstar```

With Artemis:

TODO
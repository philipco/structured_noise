
# structured_noise

## Experiments

### Single clients.
300 min by run, in overall 25 hours.
```main.py --nb_clients 1 --dataset_size 10000000 --use_ortho_matrix False```
```main.py --nb_clients 1 --dataset_size 10000000 --use_ortho_matrix True```

### Multi clients.
Around 10h.
```main.py --nb_clients 10 --dataset_size 10000000 --use_ortho_matrix True --heterogeneity="homog"```
```main.py --nb_clients 10 --dataset_size 10000000 --use_ortho_matrix True --heterogeneity="sigma"```
```main.py --nb_clients 10 --dataset_size 10000000 --use_ortho_matrix True --heterogeneity="wstar"```

With Artemis:
Full: 24h
Stochastic: 14h.
```run_wstar_experiments.py --nb_clients 10 --use_ortho_matrix True --stochastic True```

TODO :
- Comprendre le full batch avec Artemis - size = 130MB.
- faire tourner le code sur le serveur avec 5 runs
- explorer la quantization décorrélé/anticorrélé, notamment dans les cas limites.
- calculer la matrice de covariance pour le sketching
- finir d'écrire la section sur les wstars différents.
- commenter tous les théorèmes.
- problème de référencement des th/lem/prop ...
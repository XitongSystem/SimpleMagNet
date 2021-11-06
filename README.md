# SimpleMagNet
MagNet for node classification

```
# test on WebKB/Cornell with lr=1e-2 and 32 filters
cd code/src
python sparse_Magnet.py --lr=1e-2 --num_filter=32 --dataset=WebKB/Cornell --q=0.25
```

MagNet for link prediction (direction prediction)
```
# test on WebKB/Cornell with lr=1e-3 and 32 filters
cd code/src
python Edge_sparseMagnet.py --lr=1e-3 --num_filter=32 --dataset=WebKB/Cornell --q=0.25
```

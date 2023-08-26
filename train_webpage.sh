CUDA_VISIBLE_DEVICES=0 python main.py -mode train -dd dataset/webpage -pm bert -cd save/webpage -rd resource -knn True -k 16 -beta 0.2 -sp 0.7 -hd 128 -bs 12

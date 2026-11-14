









# python train.py --checkpoint_dir CHECKPOINT_PATH/  --batch_size 128 --w_cla 0.1 --w_con 0 --epochs 60 --detloss giou --claloss focal --uncer True --lr 0.00005 --num_TPCA 10
# python train.py --checkpoint_dir CHECKPOINT_PATH/  --batch_size 128 --w_cla 0 --w_con 1 --epochs 60 --detloss giou --claloss focal --uncer True --lr 0.00005 --num_TPCA 10
# python train.py --checkpoint_dir CHECKPOINT_PATH/  --batch_size 128 --w_cla 0 --w_con 0.1 --epochs 60 --detloss giou --claloss focal --uncer True --lr 0.00005 --num_TPCA 10
# python train.py --checkpoint_dir CHECKPOINT_PATH/  --batch_size 128 --w_cla 0.001 --w_con 0 --epochs 60 --detloss giou --claloss focal --uncer True --lr 0.00005 --num_TPCA 10
# python train.py --checkpoint_dir CHECKPOINT_PATH/  --batch_size 128 --w_cla 1 --w_con 1 --epochs 30 --detloss giou --claloss focal --uncer True --lr 0.00005 --num_TPCA 10
python train.py --checkpoint_dir CHECKPOINT_PATH/  --batch_size 128 --w_cla 0 --w_con 0.1 --epochs 60 --detloss giou --claloss focal --uncer True --lr 0.00001 --num_TPCA 10 --validate False
# python train.py --checkpoint_dir CHECKPOINT_PATH/  --batch_size 128 --w_cla 1 --w_con 0.1 --epochs 30 --detloss giou --claloss focal --uncer True --lr 0.00005 --num_TPCA 10









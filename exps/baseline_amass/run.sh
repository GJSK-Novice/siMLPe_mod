CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_mod.py --seed 888 --exp-name baseline.txt --layer-norm-axis spatial --with-normalization --num 48

#Test on AMASS
# python custom_test.py --model-pth /log/snapshot/model-iter-1000.pth
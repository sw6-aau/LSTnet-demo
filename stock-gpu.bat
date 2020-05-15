
#!/bin/bash
python main.py --gpu 0 --data data/exchange_rate.txt --save save/TAE.pt --hidCNN 50 --hidRNN 50 --output_fun None --epochs 1 --L1Loss False --model AELST2D --hypertune True --hyperepoch 200 --hypercnn 800 --hyperrnn 800 --hyperskip 10 --hyperkernel 32 --evals 1 --horizon 6 --results AEL2DH6.csv

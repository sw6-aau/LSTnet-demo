
#!/bin/bash
python main.py --gpu 0 --data data/exchange_rate.txt --save save/exchange_rate.pt --hidCNN 50 --hidRNN 50 --L1Loss False --output_fun None --model AENet2 --hyperepoch 200 --hypercnn 800 --hyperrnn 800 --hyperskip 10 --hyperkernel 32 --evals 5 --horizon 6
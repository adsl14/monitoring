#!/bin/bash

percentageGPU=0.2

lrs=(1e-3 1e-4 1e-5 1e-6)
dropouts=(0.0 0.1 0.2 0.3 0.4 0.5)
batch_sizes=(8 16 32)
epochs=(100 200)
patiences=(20 40 50)
patiences_reduce_lr=(5 8 12)
nNeurons=("64,32,16,8,4" "32,16,8,4" "16,8,4" "8,4" "4" "")

tam_lrs=${#lrs[*]}
tam_dropouts=${#dropouts[*]}
tam_batch_sizes=${#batch_sizes[*]}
tam_epochs=${#epochs[*]}
tam_patiences=${#patiences[*]}
tam_patiences_reduce_lr=${#patiences_reduce_lr[*]}
tam_nNeurons=${#nNeurons[*]}

for ((n=0; n<tam_nNeurons; ++n))
do

	for ((pr_lr=0; pr_lr<tam_patiences_reduce_lr; ++pr_lr))
	do

		for ((p=0; pr_lr<tam_patiences; ++p))
		do

			for ((e=0; e<tam_epochs; ++e))
			do

				for ((bs=0; bs<tam_batch_sizes; ++bs))
				do

					for ((dr=0; dr<tam_dropouts; ++dr))
					do

						for ((lr=0; lr<tam_lrs; ++lr))
						do
							python train.py --network="CNN+LSTM" --percentageGPU=$percentageGPU --learning_rate=${lrs[lr]} --batch_size=${batch_sizes[bs]} --epochs=${epochs[e]} --percentageDropout=${dropouts[dr]} --patience=${patiences[p]} --patience_reduce_lr=${patiences_reduce_lr[pr_lr]} --nNeurons=${nNeurons[n]} &
							wait
						done

					done

				done

			done

		done

	done

done
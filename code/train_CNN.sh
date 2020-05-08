#!/bin/bash
source ./models_parameters.sh
network="CNN"

for ((nsec=0; nsec<tam_nNeuronsSequence; ++nsec))
do
	for ((nconv1d=0; nconv1d<tam_nNeuronsConv1D; ++nconv1d))
	do

		for ((n=0; n<tam_nNeurons; ++n))
		do

			for ((pr_lr=0; pr_lr<tam_patiences_reduce_lr; ++pr_lr))
			do

				for ((p=0; p<tam_patiences; ++p))
				do

					for ((e=0; e<tam_epochs; ++e))
					do

						for ((bs=0; bs<tam_batch_sizes; ++bs))
						do

							for ((dr=0; dr<tam_dropouts; ++dr))
							do

								for ((lr=0; lr<tam_lrs; ++lr))
								do
									for ((kS=0; kS<tam_kernelSize; ++kS))
									do
										for ((sentinel_index=0; sentinel_index<tam_indexes_sentinel1; ++sentinel_index))
										do
											for ((sentinel=0; sentinel<tam_sentinels; ++sentinel))
											do
												for ((orbit=0; orbit<tam_orbits; ++orbit))
												do
													python train.py --nameExperiment=$nameExperiment  --sentinels=${sentinels[sentinel]} --orbits=${orbits[orbit]} --indexes_sentinel1=${indexes_sentinel1[sentile1_index]} --labels=$labels --colors_label=$colors_label --campaings=$campaings --tags_name=$tags_name --network=$network --percentageGPU=$percentageGPU --learning_rate=${lrs[lr]} --batch_size=${batch_sizes[bs]} --epochs=${epochs[e]} --percentageDropout=${dropouts[dr]} --patience=${patiences[p]} --patience_reduce_lr=${patiences_reduce_lr[pr_lr]} --kernelSize=${kernelSize[kS]} --nNeuronsSequence=${nNeuronsSequence[nsec]} --nNeuronsConv1D=${nNeuronsConv1D[nconv1d]} --nNeurons=${nNeurons[n]} &
													wait
												done

											done

										done

									done

								done

							done

						done

					done

				done

			done

		done

	done

done
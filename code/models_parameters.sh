#!/bin/bash

nameExperiment="rice_t1_balanced_inversed"
sentinels=("A" "B" "A,B")
orbits=("ASC" "DESC" "DESC,ASC")
indexes_sentinel1=("VH_Sum_VV")
labels="cumple,no_cumple"
colors_label="cyan,orange"
campaings="rice_t1_A,B_DESC,ASC_2018-11-15_2018-12-01|rice_t1_A,B_DESC,ASC_2017-11-15_2017-12-01"
tags_name="tags_balanced.csv"
percentageGPU=0.14

tam_sentinels=${#sentinels[*]}
tam_orbits=${#orbits[*]}
tam_indexes_sentinel1=${#indexes_sentinel1[*]}

lrs=(1e-3 1e-4)
batch_sizes=(16 32)
epochs=(200)
dropouts=(0.0 0.4)
patiences=(40)
patiences_reduce_lr=(7)
nNeuronsSequence=("128")
nNeuronsConv1D=("64,64")
nNeurons=("128" "")
kernelSize=(1)

tam_lrs=${#lrs[*]}
tam_batch_sizes=${#batch_sizes[*]}
tam_epochs=${#epochs[*]}
tam_dropouts=${#dropouts[*]}
tam_patiences=${#patiences[*]}
tam_patiences_reduce_lr=${#patiences_reduce_lr[*]}
tam_nNeuronsSequence=${#nNeuronsSequence[*]}
tam_nNeuronsConv1D=${#nNeuronsConv1D[*]}
tam_nNeurons=${#nNeurons[*]}
tam_kernelSize=${#kernelSize[*]}
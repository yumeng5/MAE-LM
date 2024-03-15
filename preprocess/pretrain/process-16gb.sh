#!/usr/bin/env bash

# fail fast
set -e

DATA_DIR=~/disk2/data-16gb-0207/

dict_name=128k

OUT_NAWE=bert-16g-0219-$dict_name

python get_vocab.py $DATA_DIR/sp_$dict_name.model > $DATA_DIR/dict_$dict_name.txt

pv $DATA_DIR/corpus.valid.tok.clean | \
   python ./multiprocessing_sp_encoder.py \
   --sentencepiece-model $DATA_DIR/sp_$dict_name.model | \
   --vocab $DATA_DIR/dict_$dict_name.txt \
   python handle_newline.py > $DATA_DIR/corpus.valid.sp.$dict_name

pv $DATA_DIR/corpus.train.tok.clean | \
   python ./multiprocessing_sp_encoder.py \
   --sentencepiece-model $DATA_DIR/sp_$dict_name.model | \
   --vocab $DATA_DIR/dict_$dict_name.txt \
   python handle_newline.py > $DATA_DIR/corpus.train.sp.$dict_name

python ../../fairseq_cli/preprocess.py \
   --only-source \
   --srcdict $DATA_DIR/dict_$dict_name.txt \
   --trainpref $DATA_DIR/corpus.train.sp.$dict_name \
   --validpref $DATA_DIR/corpus.valid.sp.$dict_name \
   --destdir $DATA_DIR/data-bin/$OUT_NAWE \
   --workers 40

cp $DATA_DIR/sp_$dict_name.model $DATA_DIR/data-bin/$OUT_NAWE/sp.model

# gule
bash ../glue/process.sh ../glue/glue_data ALL  $DATA_DIR/data-bin/$OUT_NAWE/ ../glue/glue-0219-$dict_name

#squad1
squad_output=../squad/squad1-0219-$dict_name
mkdir -p $squad_output
python ../squad/squad_process.py --input ../squad/train-v1.1.json --output $squad_output/train --sentencepiece-model $DATA_DIR/data-bin/$OUT_NAWE/sp.model --vocab $DATA_DIR/data-bin/$OUT_NAWE/dict.txt --is-training
python ../squad/squad_process.py --input ../squad/dev-v1.1.json --output $squad_output/valid --sentencepiece-model $DATA_DIR/data-bin/$OUT_NAWE/sp.model --vocab $DATA_DIR/data-bin/$OUT_NAWE/dict.txt
cp $DATA_DIR/data-bin/$OUT_NAWE/dict.txt $squad_output
cp $DATA_DIR/data-bin/$OUT_NAWE/sp.model $squad_output

#squad2
squad_output=../squad/squad2-0219-$dict_name
mkdir -p $squad_output
python ../squad/squad_process.py --input ../squad/train-v2.0.json --output $squad_output/train --sentencepiece-model $DATA_DIR/data-bin/$OUT_NAWE/sp.model --vocab $DATA_DIR/data-bin/$OUT_NAWE/dict.txt --is-training --version-2-with-negative
python ../squad/squad_process.py --input ../squad/dev-v2.0.json --output $squad_output/valid --sentencepiece-model $DATA_DIR/data-bin/$OUT_NAWE/sp.model --vocab $DATA_DIR/data-bin/$OUT_NAWE/dict.txt --version-2-with-negative
cp $DATA_DIR/data-bin/$OUT_NAWE/dict.txt $squad_output
cp $DATA_DIR/data-bin/$OUT_NAWE/sp.model $squad_output

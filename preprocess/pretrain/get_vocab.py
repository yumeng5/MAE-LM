import sentencepiece as spm
import sys
sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
tokens = [sp.id_to_piece(i) for i in range(len(sp))]
for token in tokens:
    sys.stdout.write(token+'\n')
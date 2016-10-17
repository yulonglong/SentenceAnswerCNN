THEANO_FLAGS="device=gpu0,mode=FAST_RUN,floatX=float32" python keras_lstm.py \
-tr data/WikiQASent-train.txt -tu data/WikiQASent-dev.txt -ts data/WikiQASent-test.txt \
-o output-test --emb vectors/glove.6B.50d.txt

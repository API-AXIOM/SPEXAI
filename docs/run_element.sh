
ELEMENT=7

DATADIR=/home/dhuppenkot2/data/spexai_data/data/element_$ELEMENT
CACHE=/home/dhuppenkot2/data/spexai_data/processed/element$ELEMENT
RUNS=/home/dhuppenkot2/data/spexai_data/runs/element$ELEMENT/recipe_t04

#mkdir /home/dhuppenkot2/data/spexai_data/runs/element$ELEMENT/
#mkdir $CACHE
#mkdir $RUNS

python scripts/preprocess_spectra.py \
    --datadir $DATADIR \
    --outdir  $CACHE \
    --skip-bad

DATAROOT=/home/dhuppenkot2/data/spexai_data
RUNS=$DATAROOT/runs
nohup python scripts/run_all_elements.py \
    --dataroot $DATAROOT --runroot $RUNS \
    --elements 2 3 4 5 6\
    > $RUNS/element2.log 2>&1 &
 
#nohup python -m spexai.train.train_adaptive \
#    --mode reweight --n_train 0 --pr_mix 0.4 --use_linehead off \
#    --schedule wsd --steps 100000 --eval_every 2000 --tag reweight_full \
#    --cachedir $CACHE \
#    --outdir  $RUNS \
#    > $RUNS/recipe_t04.log 2>&1 &

python benchmark_operator.py --rundir $RUNS/recipe_t04 --cachedir $CACHE

python benchmark_instruments.py --linehead_ckpt $RUNS/reweight_full.pt --cachedir $CACHE --rundir $RUNS --responses_dir /home/dhuppenkot2/data/spexai_data/responses

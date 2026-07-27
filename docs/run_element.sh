
ELEMENT=30

DATADIR=/home/dhuppenkot2/data/spexai_data/data/element_$ELEMENT
CACHE=/home/dhuppenkot2/data/spexai_data/processed/element$ELEMENT
RUNS=/home/dhuppenkot2/data/spexai_data/runs/element$ELEMENT/recipe_t04

mkdir /home/dhuppenkot2/data/spexai_data/runs/element$ELEMENT/
mkdir $CACHE
mkdir $RUNS

python scripts/preprocess_spectra.py \
    --datadir $DATADIR \
    --outdir  $CACHE \
    --skip-bad

DATAROOT=/home/dhuppenkot2/data/spexai_data
RUNS=$DATAROOT/runs
nohup python scripts/run_all_elements.py \
    --dataroot $DATAROOT --runroot $RUNS \
    --elements 22 23 24 25 26 27 28 29 30 \
    --train_flags "--batch 256 --lr 1e-4 " \
    > $RUNS/elements22to30.log 2>&1 &
 


# real continuation from previous model
nohup python -m spexai.train.train_adaptive \
  --mode reweight --n_train 0 --tag reweight_full \
  --init_from ~/data/spexai_data/runs/element11/tier1/reweight_full.pt \
  --finetune 1 --resume_optimizer 1 \
  --lr 1e-4 --lr_min_frac 0.01 --steps 40000 \
  --f_max 2000 --curriculum_frac 0.6 \
  --points 2048 --points_final 4096 \
  --signal_frac 0.15 --signal_frac_final 0.05 \
  --batch 256 --hidden 384 --layers 5 --n_freqs 512 --use_linehead auto \
  --eval_every 2000 \
  --cachedir ~/data/spexai_data/processed/element11 \
  --outdir ~/data/spexai_data/runs_finetune/element11/tier1 \ 
  > ~/data/spexai_data/runs_finetune/element11/finetune_testrun.log 2>&1 & 

nohup python scripts/run_all_elements.py \
  --dataroot ~/data/spexai_data --runroot ~/data/spexai_data/runs \
  --elements 11 \
  --train_flags "--tag reweight_full --init_from ~/data/spexai_data/runs/element11/tier1/reweight_full.pt --finetune 0 --resume_optimizer 0 --lr 1e-4 --lr_min_frac 0.01 --steps 100000 --f_max 2000 --curriculum_frac 0.6 --points 2048 --batch 256 --points_final 4096 --signal_frac 0.15 --signal_frac_final 0.05" > ~/data/spexai_data/runs_finetune/element11.log 2>&1 &


# with warmup
nohup bash -c '
for z in 11 13 16; do
  python scripts/run_all_elements.py \
    --dataroot ~/data/spexai_data \
    --runroot  ~/data/spexai_data/runs_warmstart \
    --elements $z \
    --train_flags "--init_from ~/data/spexai_data/runs/element${z}/tier1/reweight_full.pt --batch 256 --steps 30000 --lr 5e-4"
done
' > ~/data/spexai_data/runs_warmstart/warmstart_run2.log 2>&1 &
###### 


# Low-signal elements (e.g. 3, 4, 5) where the 
# sampling didn't work otherwise
nohup python scripts/run_all_elements.py \
    --dataroot $DATAROOT \
    --runroot $RUNS \
    --elements 3 \
    --train_flags "--lr 1e-3 --signal_frac 0.25" \
    > $RUNS/element3.log 2>&1 &

 
#nohup python -m spexai.train.train_adaptive \
#    --mode reweight --n_train 0 --pr_mix 0.4 --use_linehead off \
#    --schedule wsd --steps 100000 --eval_every 2000 --tag reweight_full \
#    --cachedir $CACHE \
#    --outdir  $RUNS \
#    > $RUNS/recipe_t04.log 2>&1 &

python benchmark_operator.py --rundir $RUNS/recipe_t04 --cachedir $CACHE

python benchmark_instruments.py --linehead_ckpt $RUNS/reweight_full.pt --cachedir $CACHE --rundir $RUNS --responses_dir /home/dhuppenkot2/data/spexai_data/responses

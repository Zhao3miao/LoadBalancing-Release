#b!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"

# 1. test BS1_right_BS2_loiter_BS3_left
python train_mopo.py --results_dir experiment3/results_mopo --test_scenario BS1_right_BS2_loiter_BS3_left --ratio 0.2 --seed 3407
python train_ppo.py --results_dir experiment3/results_ppo --test_scenario BS1_right_BS2_loiter_BS3_left --seed 3407
python train_sac_online.py --results_dir experiment3/results_sac_online --test_scenario BS1_right_BS2_loiter_BS3_left --seed 3407
python train_sac_offline.py --results_dir experiment3/results_sac+0 --test_scenario BS1_right_BS2_loiter_BS3_left --ratio 1 0.0 0.0 --seed 3407
python train_sac_offline.py --results_dir experiment3/results_sac+20 --test_scenario BS1_right_BS2_loiter_BS3_left --ratio 1 0.2 0.0 --seed 3407
python train_sac_offline.py --results_dir experiment3/results_sac+100 --test_scenario BS1_right_BS2_loiter_BS3_left --ratio 1 1.0 0.0 --seed 3407
python train_cf.py --results_dir experiment3/results_sac+20+cf --test_scenario BS1_right_BS2_loiter_BS3_left --ratio 1.0 0.2 0.0 --seed 3407 --pred_steps 15
python generate_cf_data.py --results_dir experiment3/results_sac+20+cf --test_scenario BS1_right_BS2_loiter_BS3_left --ratio 1.0 0.2 0.0 --seed 3407 --rollout_steps 15
python train_sac_offline.py --results_dir experiment3/results_sac+20+cf --test_scenario BS1_right_BS2_loiter_BS3_left --ratio 1.0 0.2 1.0 --seed 3407

# 2. test BS1_right_BS2_right_BS3_left
python train_mopo.py --results_dir experiment3/results_mopo --test_scenario BS1_right_BS2_right_BS3_left --ratio 0.2 --seed 3407
python train_ppo.py --results_dir experiment3/results_ppo --test_scenario BS1_right_BS2_right_BS3_left --seed 3407
python train_sac_online.py --results_dir experiment3/results_sac_online --test_scenario BS1_right_BS2_right_BS3_left --seed 3407
python train_sac_offline.py --results_dir experiment3/results_sac+0 --test_scenario BS1_right_BS2_right_BS3_left --ratio 1 0.0 0.0 --seed 3407
python train_sac_offline.py --results_dir experiment3/results_sac+20 --test_scenario BS1_right_BS2_right_BS3_left --ratio 1 0.2 0.0 --seed 3407
python train_sac_offline.py --results_dir experiment3/results_sac+100 --test_scenario BS1_right_BS2_right_BS3_left --ratio 1 1.0 0.0 --seed 3407
python train_cf.py --results_dir experiment3/results_sac+20+cf --test_scenario BS1_right_BS2_right_BS3_left --ratio 1.0 0.2 0.0 --seed 3407 --pred_steps 15
python generate_cf_data.py --results_dir experiment3/results_sac+20+cf --test_scenario BS1_right_BS2_right_BS3_left --ratio 1.0 0.2 0.0 --seed 3407  --rollout_steps 15
python train_sac_offline.py --results_dir experiment3/results_sac+20+cf --test_scenario BS1_right_BS2_right_BS3_left --ratio 1.0 0.2 1.0 --seed 3407 

# 3. test BS1_right_BS3_left
python train_mopo.py --results_dir experiment3/results_mopo --test_scenario BS1_right_BS3_left --ratio 0.2 --seed 3407
python train_ppo.py --results_dir experiment3/results_ppo --test_scenario BS1_right_BS3_left --seed 3407
python train_sac_online.py --results_dir experiment3/results_sac_online --test_scenario BS1_right_BS3_left --seed 3407
python train_sac_offline.py --results_dir experiment3/results_sac+0 --test_scenario BS1_right_BS3_left --ratio 1 0.0 0.0 --seed 3407
python train_sac_offline.py --results_dir experiment3/results_sac+20 --test_scenario BS1_right_BS3_left --ratio 1 0.2 0.0 --seed 3407
python train_sac_offline.py --results_dir experiment3/results_sac+100 --test_scenario BS1_right_BS3_left --ratio 1 1.0 0.0 --seed 3407
python train_cf.py --results_dir experiment3/results_sac+20+cf --test_scenario BS1_right_BS3_left --ratio 1.0 0.2 0.0 --seed 3407 --pred_steps 15
python generate_cf_data.py --results_dir experiment3/results_sac+20+cf --test_scenario BS1_right_BS3_left --ratio 1.0 0.2 0.0 --seed 3407  --rollout_steps 15
python train_sac_offline.py --results_dir experiment3/results_sac+20+cf --test_scenario BS1_right_BS3_left --ratio 1.0 0.2 1.0 --seed 3407 

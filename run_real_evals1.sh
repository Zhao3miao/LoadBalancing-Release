#b!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"

# Static & Random Baselines
python eval_static.py --results_dir experiment1/results_static --scenario_name BS1_right_BS2_right_BS3_left --env real
python eval_random.py --results_dir experiment1/results_random --scenario_name BS1_right_BS2_right_BS3_left --env real

python eval_static.py --results_dir experiment1/results_static --scenario_name BS1_right_BS2_loiter_BS3_left --env real
python eval_random.py --results_dir experiment1/results_random --scenario_name BS1_right_BS2_loiter_BS3_left --env real

python eval_static.py --results_dir experiment1/results_static --scenario_name BS1_right_BS3_left --env real
python eval_random.py --results_dir experiment1/results_random --scenario_name BS1_right_BS3_left --env real


# Online Algorithms (SAC, PPO)
python eval_sac_online.py --results_dir experiment1/results_sac_online --scenario_name BS1_right_BS2_right_BS3_left --env real
python eval_ppo.py --results_dir experiment1/results_ppo --scenario_name BS1_right_BS2_right_BS3_left --env real

python eval_sac_online.py --results_dir experiment1/results_sac_online --scenario_name BS1_right_BS2_loiter_BS3_left --env real
python eval_ppo.py --results_dir experiment1/results_ppo --scenario_name BS1_right_BS2_loiter_BS3_left --env real

python eval_sac_online.py --results_dir experiment1/results_sac_online --scenario_name BS1_right_BS3_left --env real
python eval_ppo.py --results_dir experiment1/results_ppo --scenario_name BS1_right_BS3_left --env real


# MOPO
python eval_mopo.py --results_dir experiment1/results_mopo --scenario_name BS1_right_BS2_right_BS3_left --env real
python eval_mopo.py --results_dir experiment1/results_mopo --scenario_name BS1_right_BS2_loiter_BS3_left --env real
python eval_mopo.py --results_dir experiment1/results_mopo --scenario_name BS1_right_BS3_left --env real

#Offline SAC Variants (+0, +20, +100, +20+cf)
# BS1_right_BS2_right_BS3_left
python eval_sac_offline.py --results_dir experiment1/results_sac+0 --scenario_name BS1_right_BS2_right_BS3_left --env real
python eval_sac_offline.py --results_dir experiment1/results_sac+20 --scenario_name BS1_right_BS2_right_BS3_left --env real
python eval_sac_offline.py --results_dir experiment1/results_sac+100 --scenario_name BS1_right_BS2_right_BS3_left --env real
python eval_sac_offline.py --results_dir experiment1/results_sac+20+cf --scenario_name BS1_right_BS2_right_BS3_left --env real

# BS1_right_BS2_loiter_BS3_left
python eval_sac_offline.py --results_dir experiment1/results_sac+0 --scenario_name BS1_right_BS2_loiter_BS3_left --env real
python eval_sac_offline.py --results_dir experiment1/results_sac+20 --scenario_name BS1_right_BS2_loiter_BS3_left --env real
python eval_sac_offline.py --results_dir experiment1/results_sac+100 --scenario_name BS1_right_BS2_loiter_BS3_left --env real
python eval_sac_offline.py --results_dir experiment1/results_sac+20+cf --scenario_name BS1_right_BS2_loiter_BS3_left --env real

# BS1_right_BS3_left
python eval_sac_offline.py --results_dir experiment1/results_sac+0 --scenario_name BS1_right_BS3_left --env real
python eval_sac_offline.py --results_dir experiment1/results_sac+20 --scenario_name BS1_right_BS3_left --env real
python eval_sac_offline.py --results_dir experiment1/results_sac+100 --scenario_name BS1_right_BS3_left --env real
python eval_sac_offline.py --results_dir experiment1/results_sac+20+cf --scenario_name BS1_right_BS3_left --env real



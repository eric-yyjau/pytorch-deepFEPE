""" batch evaluation script
Organized and documented by You-Yi on 07/07/2020.
This file can check and run evaluation with given checkpoints.


"""

import subprocess
import glob
import yaml
import logging
import os
from pathlib import Path
import argparse
from utils.logging import *

class sequence_info(object):
    def __init__(self, dataset='kitti'):
        self.dataset = dataset
        pass

    @staticmethod
    def get_data_from_a_seq(seq):
        """
        # ['exp_name', 'model_deepF', 'model_SP', mode, 'new_eval_name',]
        """
        mode, exp_name, pretrained, pretrained_SP = seq[3], seq[0], seq[1], seq[2]
        new_eval_name = seq[4]
        return {
            'mode': mode, 'exp_name': exp_name, 'pretrained': pretrained, 'pretrained_SP': pretrained_SP,
            'new_eval_name': new_eval_name,
        }

    @staticmethod
    def update_config(config, mode=1, exp_path='logs/', exp_name='', pretrained='', pretrained_SP='',if_print=True):
        if mode == 0:
            config['training']['pretrained'] = pretrained
            config['training']['pretrained_SP'] = pretrained_SP
        elif mode == 1: # use superpoint
            config['model']['if_SP'] = True;
            config['training']['pretrained'] = f'{exp_path}/{exp_name}/checkpoints/deepFNet_{pretrained}_checkpoint.pth.tar'
            config['training']['pretrained_SP'] = f'{exp_path}/{exp_name}/checkpoints/superPointNet_{pretrained}_checkpoint.pth.tar'
            batch_size = 12
            config['data']['batch_size'] = batch_size; logging.info("let's descrease the batch size to {batch_size}")
        elif mode == 2: # use sift
            config['training']['pretrained'] = f'{exp_path}/{exp_name}/checkpoints/deepFNet_{pretrained}_checkpoint.pth.tar'
            config['training']['pretrained_SP'] = ''
            config['model']['if_SP'] = False; logging.warning('turn off if_SP, make sure your using sift')
            batch_size = 30
            config['data']['batch_size'] = batch_size; logging.info("let's increase the batch size to {batch_size}")

        elif mode == 6: ## eval correspondences, only change superpoint
            logging.info(f"+++++ looks like youre only using Superpoint +++++")
            config['model']['pretrained'] = f'{exp_path}/{exp_name}/checkpoints/superPointNet_{pretrained}_checkpoint.pth.tar'
            batch_size = 1
            config['data']['batch_size'] = batch_size; logging.info("let's keep the batch size to {batch_size}")

        if if_print and mode <= 5:
            logging.info(f"update pretrained: {config['training']['pretrained']}")
            logging.info(f"update pretrained: {config['training']['pretrained_SP']}")
            files_list = [config['training']['pretrained'], config['training']['pretrained_SP']]
        elif if_print and mode <= 10:
            logging.info(f"update pretrained: {config['model']['pretrained']}")
            files_list = [config['model']['pretrained']]

        return config, files_list

    @staticmethod
    def export_sequences(sequences, style='table', dataset='kitti', dump_name=None):
        export_seq = {}
        postfix = 'k' if dataset == 'kitti' else 'a'
        for i, en in enumerate(sequences):
            export_seq[f"{en}.{postfix}"] = [f"{sequences[en][-1]}", 'DeepF_err_ratio.npz'] + sequences[en]
        print(f"sequences:")
        print(f"{sequences}")
        if dump_name is not None:
            file = f"configs/{dump_name}"
            with open(os.path.join(file), "a") as f:
                yaml.dump(export_seq, f, default_flow_style=True)
            logging.info(f"export sequences into {file}")


        pass

    def get_sequences(self, name='', date='1107'):
        """
        sequences: 
        # ['exp_name', 'model_deepF', 'model_SP', mode, 'new_eval_name',]
        """
        gen_eval_name = lambda exp_name, iter, date: f"{exp_name}_{iter/1000}k_{self.dataset}Testall_{date}"
        kitti_ablation = {
            # 'Si-D-k-f': ['baselineTrain_deepF_kitti_fLoss_v1', 30000, 30000, 2],  # 
            # 'Si-D-k-p': ['baselineTrain_sift_deepF_kittiPoseLoss_v1', 36000, 36000, 2],
            # 'Si-D-k-f-p': ['baselineTrain_sift_deepF_poseLoss_v0', 40000, 40000, 2],
            'Sp-D-k-f': ['baselineTrain_kittiSp_deepF_kittiFLoss_v0', 18000, 18000, 1],
            # 'Sp-D-k-p': ['baselineTrain_kittiSp_deepF_kittiPoseLoss_v1', 16000, 16000, 1],
            # 'Sp-D-end-k-f': ['baselineTrain_kittiSp_deepF_end_kittiFLoss_freezeSp_v1', 45000, 45000, 1],
            # 'Sp-D-end-k-p': ['baselineTrain_kittiSp_kittiDeepF_end_kittiPoseLoss_v0', 8000, 8000, 1],
            # 'Sp-D-end-k-f-p': ['baselineTrain_kittiSp_deepF_end_kittiFLossPoseLoss_v1_freezeSp', 38000, 38000, 1],
        }

        apollo_ablation = {  ## new apollo

            ##### current working 
            # 'Si-D-a-p-b4': ['baselineTrain_sift_deepF_poseLoss_apolloseq2_v1', 19600, 19600, 2],
            # 'Sp-D-end-a-p-b12': ['baselineTrain_apolloSp_deepF_poseLoss_apolloseq2_end_v1', 2600, 2600, 1],
            # 'Sp-D-a-p-b16': ['baselineTrain_apolloSp_deepF_poseLoss_apolloseq2_v1', 7800, 7800, 1],

            # 'Sp-D-f-b12': ['baselineTrain_apolloSp_deepF_fLoss_apolloseq2_v1', 12400, 12400, 1],
            # 'Sp-D-f-end-b8': ['baselineTrain_apolloSp_deepF_fLoss_apolloseq2_v0_end_fLoss_v1', 8200, 8200, 1],

            # 'Sp-D-end-a-f-p-b8': ['baselineTrain_apolloSp_deepF_fLoss_apolloseq2_v0_end_fLoss_v1_poseLoss', 11600, 11600, 1],
            # 'Sp-D-end-a-f-p-b8-f': ['baselineTrain_apolloSp_deepF_fLoss_apolloseq2_v0_end_fLoss_v1_poseLoss_freezeSp_fLoss', 5400, 5400, 1],
            # 'Sp-D-end-a-f-p-b8-p': ['baselineTrain_apolloSp_deepF_fLoss_apolloseq2_v0_end_fLoss_v1_poseLoss_freezeSp_poseLoss', 3800, 3800, 1],


            ##### exported
            'Si-D-a-f': ['baselineTrain_sift_deepF_fLoss_apolloseq2_v1', 13200, 13200, 2],  # 
            'Si-D-a-p': ['baselineTrain_sift_deepF_poseLoss_apolloseq2_v0', 21000, 21000, 2],
            'Sp-D-a-f': ['baselineTrain_apolloSp_deepF_fLoss_apolloseq2_v0', 20600, 20600, 1],
            'Sp-D-a-p': ['baselineTrain_apolloSp_deepF_poseLoss_apolloseq2_v0', 12600, 12600, 1],
            'Si-D-a-f-p': ['baselineTrain_sift_deepF_apolloFLossPoseLoss_v0', 35000, 35000, 2],            
            'Sp-D-end-a-f': ['baselineTrain_apolloSp_deepF_fLoss_apolloseq2_end_v0_freezeSp_fLoss', 6000, 6000, 1],
            'Sp-D-end-a-p': ['baselineTrain_apolloSp_deepF_poseLoss_apolloseq2_end_v0', 4600, 4600, 1],
            'Sp-D-end-a-f-p': ['baselineTrain_apolloSp_deepF_fLossPoseLoss_apolloseq2_end_v0_freezeSp_fLoss', 8000, 8000, 1],

            ##### end exported


        }

        # apollo_ablation = {
        #     ##### current working
        #     # 'Sp-D-end-a-p': ['baselineTrain_apolloSp_deepF_end_apolloPoseLoss_v5', 4200, 4200, 1],
        #     # 'Sp-D-end-a-f': ['baselineTrain_cocoSp_deepF_apolloFLoss_end_v0', 12200, 12200, 1],
        #     'Sp-D-end-a-p': ['baselineTrain_cocoSp_deepF_apolloFLossPoseLoss_end_v0', 11200, 11200, 1],
            
        #     # 'Sp-D-a-f': ['baselineTrain_cocoSp_deepF_apolloFLoss_v1', 7200, 7200, 1],

        #     ##### current working end
        #     # 'Si-D-a-f': ['baselineTrain_deepF_apolloFLoss_v3', 60000, 60000, 2],  # 
        #     # 'Si-D-a-p': ['baselineTrain_sift_deepF_apolloPoseLoss_v1', 32800, 32800, 2],
        #     # 'Si-D-a-f-p': ['baselineTrain_sift_deepF_apolloFLossPoseLoss_v0', 35000, 35000, 2],            ##### exported
        #     # 'Sp-D-a-f': ['baselineTrain_apolloSp_deepF_apolloFLoss_v0', 24000, 24000, 1],
        #     # 'Sp-D-a-p': ['baselineTrain_apolloSp_deepF_apolloPoseLoss_v0', 19600, 19600, 1],
        #     # 'Sp-D-end-a-f': ['baselineTrain_apolloSp_apolloDeepF_end_apolloFLoss_v0', 6600, 6600, 1],
        #     # 'Sp-D-end-a-p': ['baselineTrain_apolloSp_deepF_end_apolloPoseLoss_v1', 1800, 1800, 1],
        #     # 'Sp-D-end-a-f-p': ['baselineTrain_apolloSp_apolloDeepF_end_apolloFLossPoseLoss_v1_freezeSp_poseLoss_cont', 22000, 22000, 1],
        #     ##### end exported
        # }

        corr_ablation = {
            ##### superpoint correspondences
            'Sp-k': ['superpoint_kitti_heat2_0', 50000, 50000, 6],
            # 'Sp-D-end-k-f': ['baselineTrain_kittiSp_deepF_end_kittiFLoss_freezeSp_v1', 45000, 45000, 6],
            # 'Sp-D-end-k-p': ['baselineTrain_kittiSp_kittiDeepF_end_kittiPoseLoss_v0', 8000, 8000, 6],
            # 'Sp-D-end-k-f-p': ['baselineTrain_kittiSp_deepF_end_kittiFLossPoseLoss_v1_freezeSp', 38000, 38000, 6],
            # 'Sp-a': ['superpoint_apollo_v1', 40000, 40000, 6],
        }

        all_sequences = {'kitti_ablation': kitti_ablation, 'apollo_ablation': apollo_ablation,
                        'corr_ablation': corr_ablation}
        sequence = all_sequences.get(name, None)
        if sequence is None:
            logging.error(f"sequence name: {name} doesn't exist")
        else:
            idx_exp_name = 0
            idx_iter = 1
            for i, en in enumerate(sequence):
                eval_name = gen_eval_name(sequence[en][idx_exp_name], sequence[en][idx_iter], date)
                sequence[en].extend([eval_name])
        return sequence
            # 1, 'superpoint_kitti_heat2_0', 50000,  ## no need to run

        pass



if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # level=logging.INFO,
        level=logging.INFO,
    )

    # add parser
    parser = argparse.ArgumentParser()
    # Training command
    parser.add_argument("-d", "--dataset", type=str, choices=["kitti", "apollo"], required=True, help='kitti or apollo, select the dataset your about to eval')
    parser.add_argument("-m", "--model_base", type=str, default='kitti', help='[kitti, apollo, corr], select the dataset your about to eval')
    parser.add_argument("-e", "--exper_path", type=str, default='./logs', help='the folder to logs and other checkpoints.')
    parser.add_argument("-t", "--test_date", type=str, default='test_0708', help='test date will be appended in the end of the created folder')
    parser.add_argument("-s", "--scp", type=str, default=None, help="[cephfs, KITTI, APOLLO, local], scp checkpoints from/to your current position")
    parser.add_argument("-python", "--scp_folder", type=str, default=None, help="name to copy to local folder")
    parser.add_argument("-sf", "--scp_from_server", action="store_true", help="send data from server to here")
    

    parser.add_argument("-ce", "--check_exist", action="store_true", help="check if checkpoints exist under 'exper_path'")
    parser.add_argument("-co", "--check_output", action="store_true", help="check if already ran the sequence")
    parser.add_argument("--runEval", action="store_true", help="run deepFEPE evaluation")
    parser.add_argument("--runCorr", action="store_true", help="run superpoint correspondences evaluation")
    parser.add_argument("-es", "--export_sequences",  type=str, default=None, help="The name of dumped yaml")

    args = parser.parse_args()
    print(args)

    # set parameters
    dataset = args.dataset
    assert dataset == 'kitti' or dataset == 'apollo', 'your dataset is not supported'
    if_scp = True if args.scp is not None else False
    scp_location = args.scp
    scp_folder = args.scp_folder
    if_runEval = args.runEval
    if_runCorr = args.runCorr
    if_check_exist = args.check_exist
    if_check_output = args.check_output
    exp_path = args.exper_path
    model_base = args.model_base
    scp_from_server = args.scp_from_server
    TEST_DATE = args.test_date # 
    
    # load base config
    base_config = f'configs/kitti_corr_baselineEval.yaml'
    if if_runCorr:
        base_config = f'configs/superpoint_{dataset}_epiDist.yaml'
    else:
        if dataset == 'kitti':
            base_config = 'configs/kitti_corr_baselineEval.yaml'
        elif dataset == 'apollo':
            base_config = 'configs/apollo_train_corr_baselineEval.yaml'

    def read_base_config(base_config):
        with open(base_config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    config = read_base_config(base_config)

    # assert some settings (for evaluation)
    assert config['model']['if_SP'] == True, "sp is not used"
    assert config['training']['reproduce'] == True, "reproduce should be 'true'"
    assert config['training']['val_batches'] == -1 or config['training']['val_interval'] == -1, "val_batches should be -1 to use all test sequences"
    assert config['training']['retrain'] == False or config['training']['retrain_SP'] == False
    assert config['training']['train'] == False and config['training']['train_SP'] == False

    if config['model']['if_SP'] == False:
        logging.warning("sp is not used")

    # get experiment settings
    seq_manager = sequence_info(dataset=dataset)
    sequence_dict = seq_manager.get_sequences(name=f'{model_base}_ablation', date=TEST_DATE)  # Gamma1.5_1114
    logging.info(f"get sequence_dict: {sequence_dict}")

    # export the sequence for further evaluation
    if args.export_sequences is not None:
        seq_manager.export_sequences(sequence_dict, dataset=dataset, dump_name=args.export_sequences)

        # ['baselineEval_kittiSp_deepF_kittiPoseLoss_v1_16k_apolloTestall', 1, 'baselineTrain_kittiSp_deepF_kittiPoseLoss_v1', 16000, 16000],
        # ['baselineEval_kittiSp_deepF_end_kittiFLossPoseLoss_v1_freezeSp_38k_kittiTestall', 1, 'baselineTrain_kittiSp_deepF_end_kittiFLossPoseLoss_v1_freezeSp', 38000, 38000],
        # ['', 1, 'baselineTrain_apolloSp_deepF_apolloFLoss_v0', 24000, 24000],

    def scp_to_server(file, end_name='theia', from_server=True, location='test'):
        def form_command(remote_file, file, from_server=True, base_command='rsync -r '):
            if from_server:
                command = f"{base_command} {remote_file} {file}"
            else:
                command = f"{base_command} {file} {remote_file}"
            return command

        if end_name == 'theia':
            logging.info('just for testing')
            raise NotImplementedError
            command = form_command(remote_file, file, from_server=from_server)
        elif end_name == 'hyperion':
            raise NotImplementedError
            command = form_command(remote_file, file, from_server=from_server)
        elif end_name == 'cephfs':
            logging.info("test sending to cephfs server")
            ### be sure to use 'Path' to convert into valid path
            raise NotImplementedError
            command = f'kubectl cp {file} {remote_file}'
            command = form_command(remote_file, file, from_server=from_server, base_command='kubectl cp ')
        elif end_name == 'local':
            logging.info("sort data locally")
            # remote_file = f'exp_1108/{file}'
            # remote_file = f'exp_1109/{file}'
            remote_file = f'{location}/{file}'
            Path(remote_file).parent.mkdir(parents=True, exist_ok=True)
            command = form_command(f'{remote_file}', file, from_server=from_server)
            pass
        # elif end_name == 'output':    
        
        logging.info(f"[run command] {command}")
        subprocess.run(f"{command}", shell=True, check=True)

    def check_exit(file, entry='', should_exist=True):
        exist = Path(file).exists()
        msg = f"{entry}: {file} exist? {exist}"
        if (exist and should_exist) or (not exist and not should_exist):
            logging.info(msg)
        else:
            logging.warning(msg)
        return exist
    
    # check if the exported folders exist (check TEST_DATE)
    if if_check_output:
        check_files = "predictions/result_dict_all.npz" if model_base == "corr" else "DeepF_err_ratio.npz" 
        logging.info(f"++++++++ check_output ++++++++")
        for i, en in enumerate(sequence_dict):
            seq = sequence_dict[en]        
            data = seq_manager.get_data_from_a_seq(seq)
            new_eval_name = data['new_eval_name']
            check_exit(f"{exp_path}/{new_eval_name}/{check_files}", entry=en, should_exist=True)       
        logging.info(f"++++++++ end check_output ++++++++")
        
    # check if the checkpoints exist or scp the checkpoints to another place
    if if_scp or if_check_exist:
        logging.info(f"++++++++ if_scp or if_check_exist ++++++++")
        for i, en in enumerate(sequence_dict):
            seq = sequence_dict[en]
            data = seq_manager.get_data_from_a_seq(seq)
            mode, exp_name, pretrained, pretrained_SP = data['mode'], data['exp_name'], data['pretrained'], data['pretrained_SP']
            temp_config, files = seq_manager.update_config(config, mode, exp_path, exp_name, pretrained, pretrained_SP)
            
            # also check config
            if_config = True
            if if_config:
                files = files + [str(Path(files[0]).parent.parent/"config.yml")]

            # mkdir
            exp_dir = Path(f'{exp_path}/{exp_name}')
            exp_dir_checkpoint = exp_dir/'checkpoints'
            exp_dir_checkpoint.mkdir(parents=True, exist_ok=True)
            if if_check_output:
                new_eval_name = data['new_eval_name']
                # check_exit(f"{exp_path}/{new_eval_name}/DeepF_err_ratio.npz", entry=en, should_exist=True)
                files = [f"{exp_path}/{new_eval_name}"]
            

            for file in files:
                exist = Path(file).exists()
                if if_check_exist:
                    if exist:
                        logging.info(f"{en}: {file} exist? {exist}")
                    else:
                        logging.warning(f"{en}: {file} exist? {exist}")
                if if_scp:
                    from_server = scp_from_server
                    if from_server:
                        assert exist == False, f'{file} already exists, stoped.'
                    else:
                        assert exist == True, f'{file} not exists to scp, stoped.'

                    if len(file) > 0:
                        scp_to_server(file, end_name=scp_location, from_server=from_server, location=scp_folder)
        logging.info(f"++++++++ end if_scp or if_check_exist ++++++++")
        pass

    # run evaluation
    if if_runEval or if_runCorr:
        # run sequences
        for i, en in enumerate(sequence_dict):
            seq = sequence_dict[en]
            data = seq_manager.get_data_from_a_seq(seq)
            mode, exp_name, pretrained, pretrained_SP = data['mode'], data['exp_name'], data['pretrained'], data['pretrained_SP']
            new_eval_name = data['new_eval_name']
            # update config
            config = read_base_config(base_config)
            temp_config, _ = seq_manager.update_config(config, mode, exp_path, exp_name, pretrained, pretrained_SP)
            logging.info(f"temp_config: {temp_config}")
            temp_config_file = "temp_config_apo.yaml"
            # dump config
            with open(os.path.join("configs", temp_config_file), "w") as f:
                yaml.dump(temp_config, f, default_flow_style=False)
            if if_runEval and check_exit(f'{exp_path}/{new_eval_name}'):
                logging.error(f'{exp_path}/{new_eval_name} should not exist. Stopped!')
            if if_runEval:
                command = f"python train_good.py eval_good configs/{temp_config_file} \
                        {new_eval_name}  --eval --test"
            elif if_runCorr:
                command = f"python evaluation_epiDist.py val_feature configs/{temp_config_file} \
                        {new_eval_name}  --eval --test"
            logging.info(f"running command: {command}")
            subprocess.run(f"{command}", shell=True, check=True)



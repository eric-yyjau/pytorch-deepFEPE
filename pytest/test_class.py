
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestClass:
    def test_import(self):
        import superpoint

    def test_DeepFNet(self):
        from deepFEPE.models.DeepFNet import main
        main()

    def test_ErrorEstimators(self):
        from deepFEPE.models.ErrorEstimators import main
        main()

    def test_dataloader(self):
        import torch
        import logging
        import yaml

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info('train on device: %s', device)
        config_file = 'deepFEPE/configs/kitti_corr_baseline.yaml'
        with open(config_file, 'r') as f:
            config = yaml.load(f)
        val = 'val' 
        from deepFEPE.utils.loader import dataLoader
        data = dataLoader(config, dataset='kitti_odo_corr', val=val, warp_input=True, val_shuffle=False)
        train_loader, val_loader = data['train_loader'], data['val_loader']
        logging.info('+++[Dataset]+++ train split size %d in %d batches, val split size %d in %d batches'%\
            (len(train_loader)*config['data']['batch_size'], len(train_loader), len(val_loader)*config['data']['batch_size'], len(val_loader)))

    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        # assert hasattr(x, "check")



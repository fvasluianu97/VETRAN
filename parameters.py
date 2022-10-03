import os


# set global parameters
params = {
          "loss": 'chrb',
          "alpha": 1.0,
          "beta": 5.0,
          "gamma": 2.0,
          "delta": 3e-3,
          
          "alpha_perc": 3e-3,
          "alpha_chrb": 1.0,
          
          "lr": 5e-5,
          "sched": False, 
          "lambda init": 0.5,
          "lambda step": 30000,

          "bs": 1,
          "margin": 1,
          "crop": True,
          "crop size h": 360,
          "crop size w": 420,
          "crop factor": 1,
          
          "grad_clipping": True,
          "clip value": 0.1,
          
          "sequence length": 8,  #10,
          "eval sequence length": 10,
          "test sequence length": 10,
          "number of workers": 8,
          "eval number of workers": 4,
          "test number of workers": 4,
          
          "generator layers": 2,
          "num_blocks": 4, 
          "block activation": 'prelu',
          "block mode": 'of',
          "sd mode": 'add',
          
          "att type": 'self', 
          "riam": "diff",
          "lerp": True,
          "forget": False,

          "kernel size": 3,
          "filters_s": 64,
          "filters_d": 128,
          "state dimension": 128,
          "shuffle_factor": 4,
          
          "save interval": 500000,
          "eval interval": 5000,
          "test interval": 5000,
          "full test interval": 20000,
          "num_epochs": 10,
          "challenge": False,
          "verbose": True, 
          "model_name": "place_model_name_here",
          "suffix": "",
          "bitrate": "",
          "type": "experiment",
          "device": "cuda:0",
          "videos_train": 60,
          "train_startswith": 0,
          "videos_val": 16,
          "val_startswith": 60,
          "videos_test": 16,
          "test_startswith": 76,
          "dataset root": "place/data/dir/here",
          "results dir": "./results",
          "server": "local"
          }

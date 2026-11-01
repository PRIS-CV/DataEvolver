import json
import struct
import pytest
from dataevolver.adapters.lora_training import validate_training_config, checkpoint_inventory


def config():
    return {'schema_version':'dataevolver.lora_training.v1','unique_samples':500,'repeats':10,
        'num_train_epochs':1,'num_processes':1,'train_batch_size':1,'gradient_accumulation_steps':4,
        'max_train_steps':1250,'resolution':1024,'rank':16,'lora_alpha':16,'mixed_precision':'bf16',
        'image_column':'image','caption_column':'caption','random_flip':False,'caption_dropout':0,
        'destructive_crop':False,'dataloader_num_workers':0,'learning_rate':1e-4}


def test_exposure_catches_accidental_ddp_or_double_repetition():
    assert validate_training_config(config())['logical_exposures'] == 5000
    for patch in ({'num_processes':2}, {'num_train_epochs':10}, {'caption_column':None},
                  {'random_flip':True}, {'dataloader_num_workers':4}):
        with pytest.raises(ValueError): validate_training_config({**config(),**patch})


def test_real_tensor_header_not_file_size_proves_checkpoint_shape(tmp_path):
    path=tmp_path/'adapter.safetensors'
    header=json.dumps({'transformer.block.to_q.lora_A.weight':{
        'dtype':'F32','shape':[1,2],'data_offsets':[0,8]}}).encode()
    path.write_bytes(len(header).to_bytes(8,'little')+header+struct.pack('<ff',.1,.2))
    assert checkpoint_inventory(path)['tensors']
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):checkpoint_inventory(path)

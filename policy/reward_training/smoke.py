"""One real ResNet18 optimizer epoch using generated inputs, never robot data."""
import argparse
import json
from pathlib import Path
import numpy as np
from .train import configuration, train

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cache = a.output / 'synthetic_cache'
    cache.mkdir()
    rows = []
    for episode in ['train', 'validation']:
        np.save(cache / f'{episode}.npy', np.random.default_rng(42).integers(0,256,(4,3,3,224,224),dtype=np.uint8))
        rows.extend({'sample_id':episode, 'aligned_frame_index':i, 'label':i % 2} for i in range(4))
    frames = a.output / 'frames.jsonl'
    frames.write_text('\n'.join(json.dumps(r) for r in rows))
    split = a.output / 'split.json'
    split.write_text(json.dumps({'train_episodes':['train'], 'val_test_episodes':['validation']}))
    config = a.output / 'smoke_config.json'
    config.write_text(json.dumps({'frames':str(frames.resolve()),'cache':str(cache.resolve()),
        'split':str(split.resolve()),'output':str((a.output/'trained').resolve()),
        'epochs':1,'global_batch_size':4,'workers':0,'pretrained':False}))
    train(configuration(config))
    import torch
    from .core import ThreeViewResNet18
    state = torch.load(a.output/'trained/best.pt', map_location='cpu', weights_only=False)
    model = ThreeViewResNet18(pretrained=False).eval()
    model.load_state_dict(state['model_state_dict'], strict=True)
    with torch.no_grad():
        predictions = model.predict_video(torch.zeros(1,16,3,3,224,224,dtype=torch.uint8))
    assert predictions.shape == (1,16)
    print('RM_SMOKE_OK: optimizer + checkpoint strict reload + 16-frame inference')

if __name__ == '__main__':
    main()

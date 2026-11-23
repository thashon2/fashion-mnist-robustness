"""Simple local test: verifies that model output sizes and masked pattern labels work."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from train import MultiTaskCNN, pattern_from_text

def test_shapes():
    net=MultiTaskCNN({'category':5,'colour':4,'pattern':3},['category','colour','pattern'])
    with torch.no_grad():
        output=net(torch.randn(2,3,128,128))
    assert output['category'].shape == (2,5)
    assert output['colour'].shape == (2,4)
    assert output['pattern'].shape == (2,3)
    assert pattern_from_text('Striped Blue Shirt') == 'striped'
    assert pattern_from_text('Blue Shirt') is None
    assert pattern_from_text('Plain Striped Shirt') is None

if __name__=='__main__':
    test_shapes()
    print('Model shape and pattern-label tests passed')

from . import models
from . import utils
from pathlib import Path

def get_encoder_spaf():
    model = models.__dict__['vit_base'](
                patch_size=16, 
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                use_mean_pooling=0,
            )
    pth_path = Path(__file__).resolve().parent / "spafoundation_1m_vitb16.pth"
    utils.load_pretrained_weights(model, 
                                str(pth_path),
                                "state_dict", 
                                'vit_base', 
                                16)
    
    return model
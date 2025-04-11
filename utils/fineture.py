import torch
import re
import models as n_models

def load_checkpoint_finetune(model,cogfig=None,logger=None, model_ema=None,path=None):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)['model']
    converted_weights = {}
    keys = list(checkpoint.keys())
    for key in keys:
        if re.match(r'cls.*', key):
            continue
        else:
            converted_weights[key] = checkpoint[key]
    model.load_state_dict(converted_weights, strict=False)

    del checkpoint
    torch.cuda.empty_cache()

if __name__=='__main__':
    path= r"../.."
    model=n_models.revcol_tiny(save_memory=True).cuda()
    load_checkpoint_finetune(path=path,model=model)

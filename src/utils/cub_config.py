class Config(object):
    DATA_DIR = "/kaggle/input/cub-200-2011/images" #its expected to have 2 folders: train and test
    SAVE_MODEL_DIR = None
    MODEL = "../model/best_resnet50_cub_fullfinetune.pth" 
    MODEL_CBAM = "../model/best_resnet50_CBAM_cub.pth"
    OUTPUT_DIR = "../out"
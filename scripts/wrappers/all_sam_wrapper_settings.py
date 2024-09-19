class MobileSamModelSettings(object):

    def __init__(self):
        self.image_size : int = 1024
        self.prompt_embed_dim : int= 256
        self.vit_patch_size : int = 16
        self.CHECKPOINT : str = "./EfficientSAM/MobileSAM/weights/mobile_sam.pt"

    def get(self):
        pass

class EdgeSamModelSettings(object):

    def __init__(self):
        self.image_size : int = 1024
        self.prompt_embed_dim : int= 256
        self.vit_patch_size : int = 16
        self.CHECKPOINT : str = "./EfficientSAM/EdgeSAM/weights/edge_sam_3x.pth"


    def get(self):
        pass


class LightHQSamModelSettings(object):

    def __init__(self):
        self.image_size : int = 1024
        self.prompt_embed_dim : int= 256
        self.vit_patch_size : int = 16
        self.CHECKPOINT : str = "./EfficientSAM/LightHQSAM/weights/sam_hq_vit_tiny.pth"

    def get(self):
        pass

class RepViTSamModelSettings(object):

    def __init__(self):
        self.image_size : int = 1024
        self.prompt_embed_dim : int= 256
        self.vit_patch_size : int = 16
        self.CHECKPOINT : str = "./EfficientSAM/RepViTSAM/weights/repvit_sam.pt"
    
    def get(self):
        pass



class DINOCustomParams(object):

    def __init__(self):
        # Define default variables
        self.CAPTION : str  = 'thumb'
        self.BOX_THRESHOLD : float = 0.35
        self.TEXT_THRESHOLD : float = 0.25
        self.NMS_THRESHOLD : float = 0.6

        # GroundingDINO config and checkpoint
        self.GROUNDING_DINO_CONFIG_PATH : str = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT_PATH: str = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"

        self.EVERYTHING_MODE: bool = True

    def get(self):
        pass

    def set_everything_mode(self, mode : bool):
        self.EVERYTHING_MODE = mode

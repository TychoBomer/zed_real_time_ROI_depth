# ./scripts/wrappers/mobile_sam_wrapper.py

import torch
import cv2
import numpy as np
import supervision as sv
import torchvision

from . import all_sam_wrapper_settings as asws

from groundingdino.util.inference import Model
from segment_anything import SamPredictor, SamAutomaticMaskGenerator


class AllSAMWrapper:

    def __init__(self,sam_type : str = 'MobileSAM'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_params = asws.DINOCustomParams()
        
        self.setup_sam_model(sam_type)
        self.grounding_dino_model = Model(model_config_path=self.dino_params.GROUNDING_DINO_CONFIG_PATH,
                                           model_checkpoint_path=self.dino_params.GROUNDING_DINO_CHECKPOINT_PATH)
        
        self.sam_predictor = SamPredictor(self.sam_model)
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()
        self.mask_annotator_2 = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=32,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=400
        )

    def setup_sam_model(self, sam_type):
        if sam_type == 'MobileSAM':
            from EfficientSAM.MobileSAM.setup_mobile_sam import setup_model
            self.model_params = asws.MobileSamModelSettings()
            self.sam_model = setup_model(
                prompt_embed_dim=self.model_params.prompt_embed_dim,
                image_size=self.model_params.image_size,
                vit_patch_size=self.model_params.vit_patch_size
            )
            self.sam_model.load_state_dict(torch.load(self.model_params.CHECKPOINT),
                                            strict=True)
            self.sam_model.to(device=self.device)

        elif sam_type == 'EdgeSAM':
            from EfficientSAM.EdgeSAM.setup_edge_sam import build_edge_sam
            self.model_params = asws.EdgeSamModelSettings()
            self.sam_model = build_edge_sam(
                prompt_embed_dim=self.model_params.prompt_embed_dim,
                image_size=self.model_params.image_size,
                vit_patch_size=self.model_params.vit_patch_size,
                checkpoint=self.model_params.CHECKPOINT
            )
            self.sam_model.to(device=self.device)

            
        elif sam_type == 'LightHQSAM':
            from EfficientSAM.LightHQSAM.setup_light_hqsam import setup_model
            self.model_params = asws.LightHQSamModelSettings()
            self.sam_model = setup_model(
                    prompt_embed_dim=self.model_params.prompt_embed_dim,
                    image_size=self.model_params.image_size,
                    vit_patch_size=self.model_params.vit_patch_size
                )
            self.sam_model.load_state_dict(torch.load(self.model_params.CHECKPOINT), strict=True)
            self.sam_model.to(device=self.device)
        
        elif sam_type == 'RepViTSAM':
            from EfficientSAM.RepViTSAM.setup_repvit_sam import build_sam_repvit
            self.model_params = asws.RepViTSamModelSettings()
            self.sam_model = build_sam_repvit(
                prompt_embed_dim=self.model_params.prompt_embed_dim,
                image_size=self.model_params.image_size,
                vit_patch_size=self.model_params.vit_patch_size,
                checkpoint=self.model_params.CHECKPOINT
            )
            self.sam_model.to(device=self.device)

        elif sam_type == 'FastSAM':
            # TODO Fastsam implementation
            pass


        else:
            raise ValueError(f"Unknown SAM type: {sam_type}")
        

    # Prompting SAM with detected boxes
    def segment(self, image, xyxy):
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def predict(self, image):
        try:
            # Detect objects
                   # Join the text prompts with a period
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=[self.dino_params.CAPTION],
                box_threshold=self.dino_params.BOX_THRESHOLD,
                text_threshold=self.dino_params.TEXT_THRESHOLD
            )

            if not detections or len(detections.xyxy) == 0:
                raise ValueError("No detections found")

            # Annotate image with detections
            labels = [
                f"{self.dino_params.CAPTION} {confidence:0.2f}" 
                for _, _, confidence, class_id, _, _ 
                in detections]
            annotated_frame = self.box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

            # NMS post-process
            print(f"Before NMS: {len(detections.xyxy)} boxes")
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                self.dino_params.NMS_THRESHOLD
            ).numpy().tolist()

            if len(nms_idx) == 0:
                raise ValueError("No boxes left after NMS")

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            print(f"After NMS: {len(detections.xyxy)} boxes")

            # Convert detections to masks
            detections.mask = self.segment(
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )

            if len(detections.mask) == 0:
                raise ValueError("No masks generated")

            binary_mask = detections.mask[0].astype(np.uint8) * 255

            # Annotate image with detections
            labels = [
                f"{self.dino_params.CAPTION} {confidence:0.2f}" 
                for _, _, confidence, class_id, _, _ 
                in detections]
            annotated_image = self.mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            return annotated_image, binary_mask, annotated_frame
        

        except ValueError as e:
            print(f"An error occurred: {e}")
            empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return image, empty_mask, image.copy()

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return image, empty_mask, image.copy()

    def predict_everything(self, image, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95):
        # Convert image to RGB (SAM expects RGB input)
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks for everything in the image
        masks = self.mask_generator.generate(image)
        detections = sv.Detections.from_sam(masks)
        annotated_image = self.mask_annotator_2.annotate(image, detections)
        
        
        
        return annotated_image
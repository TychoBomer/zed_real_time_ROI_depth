import numpy as np

class Sam2PromptType:
    valid_prompt_types = {"g_dino_bbox", "bbox", "point", "mask"} # all types SAM2 could handle

    def __init__(self, prompt_type,user_caption=None, **kwargs) -> None:
        self._prompt_type = None
        self.user_caption = user_caption
        self.prompt_type = prompt_type # attempts the set function @prompt_type.setter
        self.params = kwargs
        self.validate_prompt()

    def validate_prompt(self)->None:
        if self.prompt_type == "g_dino_bbox":
            if not self.user_caption or not isinstance(self.user_caption, str):
                raise ValueError("For 'g_dino_bbox' prompt, 'user_caption' must be provided as a non-empty string.")
            
            # prepare uswer caption to use dots as separators (advised to use in GroundinDINO)
            self.user_caption = self.format_user_caption(self.user_caption)


        elif self.prompt_type == "point":
            if "point_coords" not in self.params or not isinstance(self.params["point_coords"], (list,tuple)):
                raise ValueError("For sam2 prompt 'point' prompt, 'point_coords' must be provided as a list or tuple of (x, y) coordinates.")            
            point_coords = np.array(self.params["point_coords"])
            try:
                if point_coords.ndim != 2 or point_coords.shape[1] != 2:
                    raise ValueError("Each point in 'point_coords' must have exactly two values (x, y).")
                self.params["point_coords"] = point_coords

                # Ensure labels are provided for multiple points
                if "labels" not in self.params or not isinstance(self.params["labels"], list) or len(self.params["labels"]) != len(point_coords):
                    raise ValueError("For 'point' prompt, 'labels' must be provided as a list of the same length as 'point_coords'.")
                # Allow convert proper format
                self.params["point_coords"] = point_coords

            except Exception as e:
                raise ValueError(f'Invalid format for point_coords: {e}')
                    

        elif self.prompt_type == "bbox":
            if "bbox_coords" not in self.params or not isinstance(self.params["bbox_coords"], (tuple,list)):
                raise ValueError("For sam2 prompt 'bbox', 'bbox_coords' must be provided as a tuple (x1, y1, x2, y2).")

            try:
                bbox_coords = np.array(self.params["bbox_coords"])
                if bbox_coords.ndim != 2 or bbox_coords.shape[1] != 4:
                    raise ValueError("Each bounding box must have exactly four values (x1, y1, x2, y2).")
                
                # Ensure all coordinates are valid
                for bbox in bbox_coords:
                    x1, y1, x2, y2 = bbox
                    if not (x1 < x2 and y1 < y2):
                        raise ValueError(f"Invalid bbox coordinates: {bbox}. Ensure (x1, y1, x2, y2) with x1 < x2 and y1 < y2.")
                self.params["bbox_coords"] = bbox_coords
            
            except Exception as e:
                raise ValueError(f'Invalid format for bbox_coords: {e}')

                
        elif self.prompt_type == "mask":
            raise NotImplementedError("Not implemented yet and probably will not be used")
        
    @staticmethod
    def format_user_caption(user_cap:str) -> str:
        # Force split with dot
        formatted_caption = user_cap.replace(",", ".").replace("_", ".").replace(" ", ".").replace("+",".").replace("/",".")
        formatted_caption = ".".join(filter(None, formatted_caption.split(".")))
        return formatted_caption

    @property 
    def prompt_type(self) -> str:
        return self._prompt_type

    @prompt_type.setter
    def prompt_type(self, selected_prompt_type) -> None:
        if selected_prompt_type not in self.valid_prompt_types:
            raise ValueError(f"Invalid prompt type for SAM2! Valid promt types are: {self.valid_prompt_types}")
        self._prompt_type = selected_prompt_type

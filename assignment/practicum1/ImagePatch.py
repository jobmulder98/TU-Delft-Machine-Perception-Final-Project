from BoundingBox import BoundingBox

class ImagePatch:
    def __init__(self, image, bbox, score=None):
        assert type(bbox).__name__.endswith("BoundingBox"), type(bbox)
        self.image = image
        self.bbox = bbox
        self.score = score

    def __repr__(self):
        image_str = self.image.shape if self.image is not None else "None"
        return f"ImagePatch(image={image_str}, bbox={self.bbox}, score={self.score})"

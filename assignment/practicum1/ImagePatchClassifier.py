
import tensorflow as tf

class ImagePatchClassifier(tf.keras.Model):
    def __init__(self, feature_extractor, classifier):
        super(ImagePatchClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.flatten1 = tf.keras.layers.Flatten()

    @tf.function
    def call(self, image):
        features = self.feature_extractor(image)
        features_flat = self.flatten1(features)
        estimate = self.classifier(features_flat)
        return estimate


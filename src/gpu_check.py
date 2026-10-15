"""Require CUDA device discovery and a real convolution gradient step."""
import json
from config import initialize_gpu_environment, RESULTS_DIR, write_json
initialize_gpu_environment()
import tensorflow as tf

def configure_gpu(required=True):
    devices=tf.config.list_physical_devices('GPU')
    if required and not devices:
        raise RuntimeError('No TensorFlow GPU. Install requirements-gpu.txt and run scripts/run.sh gpu-check. CPU fallback is disabled.')
    for device in devices:
        try: tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError: pass
    return devices

def main():
    devices=configure_gpu()
    tf.config.set_soft_device_placement(False)
    with tf.device('/GPU:0'):
        kernel=tf.Variable(tf.random.normal((3,3,1,8)))
        inputs=tf.random.normal((2,64,173,1))
        with tf.GradientTape() as tape:
            output=tf.nn.conv2d(inputs,kernel,1,'SAME')
            loss=tf.reduce_mean(output**2)
        gradient=tape.gradient(loss,kernel)
        kernel.assign_sub(.001*gradient)
    result=dict(tensorflow=tf.__version__, devices=[d.name for d in devices],
        computation_device=output.device, gradient_device=gradient.device,
        loss=float(loss), passed='GPU:0' in output.device,
        build_info=tf.sysconfig.get_build_info())
    if not result['passed']: raise RuntimeError('Convolution did not execute on GPU')
    write_json(RESULTS_DIR/'gpu_check.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()

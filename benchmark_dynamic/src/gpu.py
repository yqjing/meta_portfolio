import subprocess

def get_gpu_status(device):
    if device.type != 'cpu':
        try:
            # Run gpustat command and capture its output
            output = subprocess.check_output(["/work/y5jing/.local/bin/gpustat"])
            # output = subprocess.check_output(["nvidia-smi"])
            return output.decode("utf-8")  # Decode byte string to regular string
        except subprocess.CalledProcessError:
            # Handle errors if gpustat command fails
            return "Error: Unable to retrieve GPU status."
    else:
        pass

# Example usage
# gpu_status = get_gpu_status()


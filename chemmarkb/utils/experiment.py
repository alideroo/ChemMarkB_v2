import datetime
import os
import re
import time


def get_config_name(config_path):
    config_path = str(config_path)
    return os.path.basename(config_path)[: os.path.basename(config_path).rfind(".")]


def get_experiment_name(model_name: str, version: str, version_time: datetime.datetime):
    illegal_chars = re.compile(r"[\\/:\"*?<>|]+")
    model_name = illegal_chars.sub("_", model_name)
    version = illegal_chars.sub("_", version)
    time_prefix = version_time.strftime("%y%m%d%H%M")
    return f"{model_name}/{time_prefix}-{version}"


def get_experiment_version():
    return time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

def parse_device_ids(devices: str) -> list[int]:
    if not devices:
        raise ValueError("Device string cannot be empty")
    
    try:
        device_list = [int(d.strip()) for d in devices.split(",") if d.strip()]
        if not device_list:
            raise ValueError("No valid device IDs found")
        return device_list
    except ValueError as e:
        raise ValueError(
            f"Invalid device string: '{devices}'. "
            f"Expected format: '0,1,2' or '0'. Error: {e}"
        )
    
def validate_batch_size(
    batch_size: int,
    num_devices: int,
    require_divisible: bool = True
) -> int:
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
    
    if num_devices <= 0:
        raise ValueError(f"Number of devices must be positive, got {num_devices}")
    
    if require_divisible and batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by "
            f"the number of devices ({num_devices}). "
            f"Suggestion: use batch size {(batch_size // num_devices + 1) * num_devices}"
        )
    
    return (batch_size + num_devices - 1) // num_devices
import subprocess
import csv
import time
from datetime import datetime

def get_power_metrics():
    try:
        result = subprocess.run(
            ['sudo', 'powermetrics', '-n1', '--samplers', 'cpu_power,gpu_power'],
            capture_output=True, text=True
        )
        output = result.stdout
        cpu_power = gpu_power = total_power = None

        for line in output.splitlines():
            if "CPU Power" in line:
                cpu_power = float(line.split(":")[1].strip().split()[0])
            elif "GPU Power" in line:
                gpu_power = float(line.split(":")[1].strip().split()[0])
            elif "Total Power" in line:
                total_power = float(line.split(":")[1].strip().split()[0])

        return cpu_power, gpu_power, total_power
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

# CSV setup
with open("power_metrics.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "CPU Power (mW)", "GPU Power (mW)", "Total Power (mW)"])

    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu, gpu, total = get_power_metrics()
        writer.writerow([timestamp, cpu, gpu, total])
        time.sleep(1)


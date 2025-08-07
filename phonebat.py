import subprocess
import csv
import time
from datetime import datetime

def parse_battery_output(output):
    data = {}
    for line in output.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            data[key] = value
    return data

def log_battery_to_csv(filename, interval=1):
    with open(filename, mode='w', newline='') as file:
        writer = None
        while True:
            try:
                result = subprocess.run(
                    ['adb', 'shell', 'dumpsys', 'battery'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    print("ADB error:", result.stderr)
                    break

                battery_data = parse_battery_output(result.stdout)
                battery_data['timestamp'] = datetime.now().isoformat()

                if writer is None:
                    # Write header once
                    writer = csv.DictWriter(file, fieldnames=battery_data.keys())
                    writer.writeheader()

                writer.writerow(battery_data)
                print("Logged:", battery_data['timestamp'], battery_data.get('battery_level', 'N/A'))

                time.sleep(interval)

            except KeyboardInterrupt:
                print("Logging stopped by user.")
                break

# Run the logger
log_battery_to_csv('battery_log.csv', interval=1)


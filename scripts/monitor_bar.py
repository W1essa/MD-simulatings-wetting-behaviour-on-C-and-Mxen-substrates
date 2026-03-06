import time
import sys
import re
from tqdm import tqdm

# USAGE: python monitor_bar.py <total_steps> <log_file_path>

def get_last_recorded_step(filename):
    """Читає файл повністю, щоб знайти останній записаний крок."""
    last_step = 0
    try:
        with open(filename, "r") as f:
            for line in f:
                # Шукаємо рядки, що починаються з числа
                match = re.match(r'^\s*(\d+)\s+', line)
                if match:
                    last_step = int(match.group(1))
    except FileNotFoundError:
        return 0
    return last_step

def monitor_lammps(total_steps, log_file):
    print(f"--- Monitoring: {log_file} ---")
    
    # 1. Знаходимо поточний стан, щоб не читати старі дані
    current_step = get_last_recorded_step(log_file)
    print(f"--- Current status: Step {current_step} of {total_steps} ---")

    # 2. Ініціалізуємо бар з початковим значенням (initial=current_step)
    pbar = tqdm(total=total_steps, initial=current_step, unit="step", ncols=80, colour="green")
    
    try:
        with open(log_file, "r") as f:
            # 3. Переходимо в кінець файлу, щоб читати ТІЛЬКИ нові дані
            f.seek(0, 2) 
            
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5) # Чекаємо на нові дані
                    continue
                
                match = re.match(r'^\s*(\d+)\s+', line)
                if match:
                    step = int(match.group(1))
                    
                    if step > current_step:
                        increment = step - current_step
                        pbar.update(increment)
                        current_step = step
                    
                    if current_step >= total_steps:
                        break
    except KeyboardInterrupt:
        pbar.close()
        print("\nMonitoring stopped.")
    except FileNotFoundError:
        print(f"Error: Could not find file {log_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_bar.py <total_steps> [log_file]")
        sys.exit(1)
        
    steps = int(sys.argv[1])
    logfile = sys.argv[2] if len(sys.argv) > 2 else "log.lammps"
        
    monitor_lammps(steps, logfile)
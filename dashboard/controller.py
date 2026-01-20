import subprocess
import os
import signal
import json
import time

class SystemController:
    def __init__(self):
        self.processes = {}
        self.logs = {}

    def save_config(self, new_config):
        """Save updated config to config.json"""
        try:
            # Validate types if needed
            with open("config.json", "w") as f:
                json.dump(new_config, f, indent=4)
            return True, "Config Saved"
        except Exception as e:
            return False, str(e)

    def load_config(self):
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except:
            return {}

    def run_process(self, name, command):
        """Run a background process"""
        if name in self.processes and self.processes[name].poll() is None:
            return False, "Process already running"

        # Log file
        log_file = f"{name}.log"
        self.logs[name] = log_file
        
        try:
            # We use subprocess.Popen to run in background
            # Redirect stdout/stderr to log file
            with open(log_file, "w") as out:
                p = subprocess.Popen(
                    command, 
                    shell=True, # Need shell for python -m ...
                    stdout=out, 
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid # Create new session group
                )
            self.processes[name] = p
            return True, f"Started PID: {p.pid}"
        except Exception as e:
            return False, str(e)

    def stop_process(self, name):
        if name not in self.processes:
            return False, "Process not found"
        
        p = self.processes[name]
        if p.poll() is not None:
            return True, "Already stopped"
            
        try:
            # Kill the process group to ensure children die
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(timeout=5)
            return True, "Stopped"
        except Exception as e:
            return False, str(e)

    def get_status(self, name):
        if name not in self.processes:
            return "Stopped"
        if self.processes[name].poll() is None:
            return "Running"
        return "Stopped"

    def get_log_tail(self, name, n=20):
        log_file = f"{name}.log"
        if not os.path.exists(log_file):
            return "No logs yet."
        try:
            with open(log_file, "r") as f:
                return "".join(f.readlines()[-n:])
        except:
            return "Error reading log."

# Singleton instance for Streamlit state? 
# Streamlit reloads classes. Better to keep this stateful? 
# For simple usage, we can't easily persist the Popen object across Streamlit reruns 
# because Streamlit stores picklable state, and Popen is not picklable.
# SOLUTION: Use PID file or check OS processes. 
# But for MVP within a session, we can try to use st.session_state if the object was created outside 
# or use a lock file. 
# Actually, standard Streamlit approach for persistence of non-picklable objects is tricky.
# We will use psutil or pid files to track status if we want robustness.
# For now, let's just create a PID file tracker.

class PersistentController(SystemController):
    PID_DIR = ".pids"
    
    def __init__(self):
        super().__init__()
        if not os.path.exists(self.PID_DIR):
            os.makedirs(self.PID_DIR)

    def _save_pid(self, name, pid):
        with open(os.path.join(self.PID_DIR, name), "w") as f:
            f.write(str(pid))
            
    def _read_pid(self, name):
        try:
            with open(os.path.join(self.PID_DIR, name), "r") as f:
                return int(f.read().strip())
        except:
            return None

    def run_process(self, name, command):
        import subprocess # Local import to be safe
        
        # Check if running by PID
        old_pid = self._read_pid(name)
        if old_pid:
            try:
                os.kill(old_pid, 0) # Check if alive
                return False, "Already running (PID exists)"
            except OSError:
                pass # Dead
        
        log_file = f"logs/{name}.log"
        if not os.path.exists("logs"): os.makedirs("logs")
        
        try:
            with open(log_file, "w") as out:
                # Use setsid to easily kill group
                p = subprocess.Popen(
                    command, 
                    shell=True,
                    stdout=out, 
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    cwd=os.getcwd() 
                )
            self._save_pid(name, p.pid)
            return True, f"Started {name} (PID: {p.pid})"
        except Exception as e:
            return False, str(e)

    def stop_process(self, name):
        pid = self._read_pid(name)
        if not pid: return False, "Not running"
        
        try:
            os.killpg(pid, signal.SIGTERM) # Kill group assuming setsid was used? 
            # Actually setsid makes the PID the group leader. 
            # If we didn't use setsid, we might just kill the shell wrapper.
            # Let's hope setsid coupled with shell=True works as expected on Mac.
            # Alternatively: os.kill(pid, signal.SIGTERM)
            
            # Remove PID file
            if os.path.exists(os.path.join(self.PID_DIR, name)):
                os.remove(os.path.join(self.PID_DIR, name))
            return True, "Stopped"
        except Exception as e:
            return False, str(e)
            
    def get_status(self, name):
        pid = self._read_pid(name)
        if not pid: return "Stopped"
        
        try:
            import psutil
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                return "Stopped"
            return "Running"
        except ImportError:
            # Fallback if psutil not installed
            try:
                os.kill(pid, 0)
                return "Running"
            except OSError:
                return "Stopped"
        except Exception:
            # psutil.NoSuchProcess or other errors
            return "Stopped"

    def get_log_tail(self, name, n=20):
        log_file = f"logs/{name}.log"
        if not os.path.exists(log_file): return "Waiting for logs..."
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                return "".join(lines[-n:])
        except:
            return "Error reading log"

    def get_full_log(self, name, n=1000):
        log_file = f"logs/{name}.log"
        if not os.path.exists(log_file): return "Wait for logs..."
        try:
            with open(log_file, "r") as f:
                # Read last n lines to avoid memory issues with huge logs
                # Simple approach: read all then slice. 
                # For very large files, seek might be better but for now this is fine.
                lines = f.readlines()
                return "".join(lines[-n:])
        except str as e:
            return f"Error reading log: {str(e)}"

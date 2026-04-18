import subprocess
import os
import time
import io
import csv
from settings_folder import settings
from common import utils
import airsim
import msgs
import psutil
import platform

class GameHandler:
    def __init__(self):
        self.game_file = settings.game_file
        press_play_dir = os.path.dirname(os.path.realpath(__file__))
        self.ue4_exe_path = settings.unreal_exec

        # -game: Run in game mode, -WINDOWED: Windowed mode, -VRMode: disable VR
        # -AutomationTest or use PIE (Play In Editor) - but we want standalone game
        self.ue4_params = " -game"+" -ResX="+str(settings.game_resX)+ " -ResY="+str(settings.game_resY)+ \
                          " -WinX="+str(settings.ue4_winX)+ " -WinY="+str(settings.ue4_winY)+ " -Windowed -NOPAUSE"
        self.cmd = str('"'+ self.ue4_exe_path+ '"')+" "+str('"'+ self.game_file+ '"')+ str(self.ue4_params)
        assert(os.path.exists(self.ue4_exe_path)), "Unreal Editor executable:" + self.ue4_exe_path + "doesn't exist"
        assert(os.path.exists(self.game_file)), "game_file: " + self.game_file +  " doesn't exist"

    def _kill_processes_by_name(self, name):
        """Force-kill all processes with the given image name."""
        if not name:
            return
        os.system(f'killall -9 "{name}" > /dev/null 2>&1')

    def _is_port_in_use(self, port):
        """Check whether the given TCP port is currently occupied."""
        try:
            output = subprocess.check_output(
                f"ss -tuln | grep :{port} || netstat -tuln | grep :{port}",
                shell=True, stderr=subprocess.DEVNULL, text=True, errors="ignore", timeout=5
            )
            return bool(output.strip())
        except Exception:
            return False

    def _kill_port_owner(self, port):
        """Try to kill whatever process is holding the given TCP port."""
        os.system(f"fuser -k -n tcp {port} > /dev/null 2>&1")

    def start_game_in_editor(self):
        popen_kwargs = {
            "shell": True,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

        # Kill existing game instance if any
        self.kill_game_in_editor()

        # Wait until the AirSim port is actually free
        airsim_port = getattr(settings, 'port', 41451)
        for i in range(10):
            if not self._is_port_in_use(airsim_port):
                break
            print(f"Port {airsim_port} still in use, waiting... ({i+1}/10)")
            self._kill_port_owner(airsim_port)
            time.sleep(2)
        else:
            print(f"WARNING: Port {airsim_port} is still in use after cleanup. Launch may fail.")

        unreal_pids_before_launch = utils.find_process_id_by_name("UE4Editor")
        subprocess.Popen(self.cmd, **popen_kwargs)
        time.sleep(2)

        diff_proc = []
        max_attempts = 12
        attempts = 0
        while not (len(diff_proc) == 1) and attempts < max_attempts:
            time.sleep(15)
            current_pids = utils.find_process_id_by_name("UE4Editor")
            diff_proc = (utils.list_diff(current_pids, unreal_pids_before_launch))
            attempts += 1

        if len(diff_proc) != 1:
            raise RuntimeError(
                "Failed to detect a new UE process after launch. "
                "This usually means UE failed to start (e.g., 'bind: Address already in use')."
            )

        settings.game_proc_pid = diff_proc[0]
        
        # 等待游戏进程完全启动
        time.sleep(25)
        print("Game process started. Ready for AirSim connection.")

    def kill_game_in_editor(self):
        # 1) Kill by cached PID first
        pid_str = str(getattr(settings, "game_proc_pid", "")).strip()
        if pid_str:
            os.system(f'kill -9 {pid_str} > /dev/null 2>&1')
            settings.game_proc_pid = ''

        # 2) Kill by process names
        process_names = {"UE4Editor", "UnrealEditor"}
        basename = os.path.basename(self.ue4_exe_path)
        if basename:
            process_names.add(basename)

        for _ in range(2):
            for name in list(process_names):
                self._kill_processes_by_name(name)
            self._kill_processes_by_name("CrashReportClient")
            time.sleep(2)
            any_alive = False
            for name in list(process_names):
                if self._is_target_process_alive(name):
                    any_alive = True
                    break
            if not any_alive:
                break

        # 3) Kill whoever is still holding the AirSim port
        airsim_port = getattr(settings, 'port', 41451)
        self._kill_port_owner(airsim_port)

        # 4) Wait for OS to actually release the port
        for i in range(5):
            if not self._is_port_in_use(airsim_port):
                break
            time.sleep(1)
        else:
            print(f"WARNING: Port {airsim_port} is still in use after kill_game_in_editor. Next launch may fail.")

    def restart_game(self):
        msgs.restart_game_count += 1
        self.kill_game_in_editor()
        time.sleep(2)
        self.start_game_in_editor()

    def _is_target_process_alive(self, target_name):
        """Check whether a target UE process exists and is in a valid running state."""
        if not target_name:
            return False

        raw_pids = utils.find_process_id_by_name(target_name)
        if not raw_pids:
            return False

        for pid in raw_pids:
            try:
                proc = psutil.Process(pid)
                if proc.is_running() and proc.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False

    def is_game_process_alive(self):
        """Check whether UE editor process is alive."""
        for target_name in ["UE4Editor", "UnrealEditor"]:
            if self._is_target_process_alive(target_name):
                return True
        return False

    def is_game_window_alive(self):
        """
        Check whether UE editor has a visible main window.
        (Not supported on pure Ubuntu; returns None.)
        """
        return None

    def probe_game_health(self, check_window=True):
        """
        Probe UE runtime health from process perspective.
        Returns a dict for monitoring/debug logs.
        """
        process_alive = self.is_game_process_alive()
        return {
            "process_alive": process_alive,
            "window_alive": None,
        }

    def check_and_recover_game(self, force_restart=False, reason="", check_window=True):
        """
        Checks if the UE process is running (or forced unhealthy), then restarts.
        Returns True if the game was restarted, False otherwise.
        """
        health = self.probe_game_health(check_window=check_window)
        process_alive = health["process_alive"]

        if not process_alive:
            force_restart = True
            reason = "process_missing"

        if not force_restart:
            return False

        print(f"WARNING: UE process marked unhealthy ({reason or 'unknown reason'}). Initiating forced recovery...")
        self.restart_game()
        return True

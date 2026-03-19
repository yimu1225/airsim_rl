import subprocess
import os
import time
import shutil
from settings_folder import settings
from common import utils
import airsim
import msgs
import psutil

class GameHandler:
    def __init__(self):
        # if not (settings.ip == '127.0.0.1'):
        #     return

        self.game_file = settings.game_file
        press_play_dir = os.path.dirname(os.path.realpath(__file__))
        #self.press_play_file = press_play_dir +"\\game_handling\\press_play\\Debug\\Debug"
        #self.press_play_file = press_play_dir +"\\press_play\\Debug\\press_play.exe"
        self.ue4_exe_path = settings.unreal_exec

        # Ubuntu direct-run mode: use native Linux path directly.
        self.game_file_arg = self.game_file

        # -game: Run in game mode, -WINDOWED: Windowed mode, -VRMode: disable VR
        # -AutomationTest or use PIE (Play In Editor) - but we want standalone game
        self.ue4_params = " -game"+" -ResX="+str(settings.game_resX)+ " -ResY="+str(settings.game_resY)+ \
                          " -WinX="+str(settings.ue4_winX)+ " -WinY="+str(settings.ue4_winY)+ " -Windowed -NOPAUSE"
        self.cmd = str('"'+ self.ue4_exe_path+ '"')+" "+str('"'+ self.game_file_arg+ '"')+ str(self.ue4_params)
        ue4_exists = os.path.exists(self.ue4_exe_path) or (shutil.which(self.ue4_exe_path) is not None)
        assert ue4_exists, "Unreal Editor executable:" + self.ue4_exe_path + "doesn't exist"
        assert(os.path.exists(self.game_file)), "game_file: " + self.game_file +  " doesn't exist"
        #assert(os.path.exists(self.press_play_file)), "press_play file: " + self.press_play_file +  " doesn't exist"

    def _editor_process_names(self):
        if os.name == "nt":
            return ("UE4Editor.exe", "UnrealEditor.exe")
        return ("UE4Editor", "UnrealEditor")

    def _find_editor_pids(self):
        pids = []
        for name in self._editor_process_names():
            pids.extend(utils.find_process_id_by_name(name))
        return sorted(set(pids))


    def start_game_in_editor(self):
        # if not (settings.ip == '127.0.0.1'):
        #     print("can not start the game in a remote machine")
        #     exit(0)

        # Kill existing game instance if any, similar to Windows behavior
        self.kill_game_in_editor()
        time.sleep(2)

        if(os.name=="nt"):
            unreal_pids_before_launch = self._find_editor_pids()
            # Ensure game_file is quoted in case of spaces
            self.cmd = str('"'+ self.ue4_exe_path+ '"')+" "+str('"'+ self.game_file+ '"')+ str(self.ue4_params)
            subprocess.Popen(self.cmd, shell=True)
            time.sleep(2)
        else:
            unreal_pids_before_launch = self._find_editor_pids()

            # Ensure game_file is quoted in case of spaces
            arg = getattr(self, "game_file_arg", self.game_file)
            self.cmd = str('"'+ self.ue4_exe_path+ '"')+" "+str('"'+ arg + '"')+ str(self.ue4_params)
            subprocess.Popen(self.cmd, shell=True)
            time.sleep(2)

        diff_proc = []  # a list containing the difference between the previous UE4 processes
        # and the one that is about to be launched

        # wait till there is a UE4Editor process
        while not (len(diff_proc) == 1):
            time.sleep(3)
            current_pids = self._find_editor_pids()
            diff_proc = (utils.list_diff(current_pids, unreal_pids_before_launch))
            if len(diff_proc) >= 1:
                break

        settings.game_proc_pid = diff_proc[0]
        #time.sleep(30)
        client = airsim.MultirotorClient(settings.ip)
        connection_established = False
        connection_ctr = 0  # counting the number of time tried to connect
        # wait till connected to the multi rotor
        time.sleep(5)
        while not (connection_established):
            try:
                #os.system(self.press_play_file)
                # time.sleep(2)
                client.confirmConnection()
                connection_established = True
            except Exception as e:
                if (connection_ctr >= settings.connection_count_threshold and msgs.restart_game_count >= settings.restart_game_from_scratch_count_threshold):
                    print("couldn't connect to the UE4Editor multirotor after multiple tries")
                    print("memory utilization:" + str(psutil.virtual_memory()[2]) + "%")
                    exit(0)
                if (connection_ctr == settings.connection_count_threshold):
                    self.restart_game()
                print("connection not established yet")
                time.sleep(5)
                connection_ctr += 1
                client = airsim.MultirotorClient(settings.ip)
                pass

        # Connection is established, game should be ready
        print("Connection established! Game is ready.")
        
        """ 
		os.system(self.game_file)
		time.sleep(30) 
		os.system(self.press_play_file)
		time.sleep(2)
		"""

    def kill_game_in_editor(self):
        process1_exist = False
        process2_exist = False
        tasklist = []

        for each_proc in psutil.process_iter():
            try:
                tasklist.append(each_proc.name())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        for el in tasklist:
            if ("UE4Editor" in el) or ("UnrealEditor" in el):
                process1_exist = True
            if "CrashReportClient" in el:
                process2_exist = True
            if process1_exist and process2_exist:
                break

        if (settings.game_proc_pid == ''):  # if proc not provided, find any Unreal and kill
            if process1_exist:
                if(os.name=="nt"):
                    os.system("taskkill /f /im  " + "UE4Editor.exe")
                    os.system("taskkill /f /im  " + "UnrealEditor.exe")
                elif(os.name=="posix"):
                    # Only run killall if process was actually found in Linux
                    os.system("killall UE4Editor > /dev/null 2>&1")
                    os.system("killall UnrealEditor > /dev/null 2>&1")
        else:
            if(os.name=="nt"):
                os.system("taskkill /f /pid  " + str(settings.game_proc_pid))
                time.sleep(2)
                settings.game_proc_pid = ''
            elif(os.name=="posix"):
                if str(settings.game_proc_pid).strip():
                    os.system("kill " + str(settings.game_proc_pid))
                time.sleep(2)
                settings.game_proc_pid = ''

        if process2_exist:
            if(os.name=="nt"):
                os.system("taskkill /f /im  " + "CrashReportClient.exe")
            elif(os.name=="posix"):
                os.system("killall " + "CrashReportClient")



    def restart_game(self):
        # if not (settings.ip == '127.0.0.1'):
        #     print("can not restart the game in a remote machine")
        #     exit(0)
        msgs.restart_game_count += 1
        self.kill_game_in_editor()  # kill in case there are any
        time.sleep(2)
        self.start_game_in_editor()

    def _is_target_process_alive(self, target_name):
        """
        Check whether a target UE process exists and is in a valid running state.
        """
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
        """
        Check whether UE editor process is alive (UE4/UE5 compatible names).
        """
        target_names = ["UE4Editor.exe", "UnrealEditor.exe"] if os.name == "nt" else ["UE4Editor", "UnrealEditor"]
        for target_name in target_names:
            if self._is_target_process_alive(target_name):
                return True
        return False

    def is_game_window_alive(self):
        """
        Ubuntu direct-run mode does not use window-health probing.
        Returns None to indicate "not checked".
        """
        return None

    def probe_game_health(self, check_window=True):
        """
        Probe UE runtime health from process/window perspective.
        Returns a dict for monitoring/debug logs.
        """
        process_alive = self.is_game_process_alive()
        window_alive = None
        if process_alive and check_window:
            window_alive = self.is_game_window_alive()
        return {
            "process_alive": process_alive,
            "window_alive": window_alive,
        }

    def check_and_recover_game(self, force_restart=False, reason="", check_window=True):
        """
        Checks if the UE process is running (or forced unhealthy), then restarts.
        Returns True if the game was restarted, False otherwise.
        """
        health = self.probe_game_health(check_window=check_window)
        process_alive = health["process_alive"]
        window_alive = health["window_alive"]

        # Process state has highest priority.
        # If process is gone, always treat as process-missing recovery,
        # regardless of any prior window-related reason.
        if not process_alive:
            force_restart = True
            reason = "process_missing"

        if process_alive and not force_restart and check_window:
            if window_alive is False:
                force_restart = True
                reason = reason or "ue_window_closed"

        if process_alive and not force_restart:
            return False

        if force_restart:
            print(f"WARNING: UE process marked unhealthy ({reason or 'unknown reason'}). Initiating forced recovery...")
        else:
            print("WARNING: UE process not found (or zombie) during monitoring! Initiating recovery...")

        self.restart_game()
        return True

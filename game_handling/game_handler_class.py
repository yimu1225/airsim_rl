import subprocess
import os
import time
from settings_folder import settings
from common import utils
import airsim
import msgs
import psutil
import platform

class GameHandler:
    def __init__(self):
        # if not (settings.ip == '127.0.0.1'):
        #     return

        self.game_file = settings.game_file
        press_play_dir = os.path.dirname(os.path.realpath(__file__))
        #self.press_play_file = press_play_dir +"\\game_handling\\press_play\\Debug\\Debug"
        #self.press_play_file = press_play_dir +"\\press_play\\Debug\\press_play.exe"
        self.ue4_exe_path = settings.unreal_exec

        # WSL2 Compat: Convert game file path to Windows format if we are launching a Windows .exe from Linux
        self.game_file_arg = self.game_file
        if platform.system() == "Linux" and "microsoft" in platform.uname().release.lower() and self.ue4_exe_path.endswith(".exe"):
            try:
                self.game_file_arg = subprocess.check_output(["wslpath", "-w", self.game_file]).decode().strip()
                print(f"WSL2 Mode: Converted game file path to {self.game_file_arg}")
            except Exception as e:
                print(f"Warning: Failed to convert path using wslpath: {e}")

        # -game: Run in game mode, -WINDOWED: Windowed mode, -VRMode: disable VR
        # -AutomationTest or use PIE (Play In Editor) - but we want standalone game
        self.ue4_params = " -game"+" -ResX="+str(settings.game_resX)+ " -ResY="+str(settings.game_resY)+ \
                          " -WinX="+str(settings.ue4_winX)+ " -WinY="+str(settings.ue4_winY)+ " -Windowed -NOPAUSE"
        self.cmd = str('"'+ self.ue4_exe_path+ '"')+" "+str('"'+ self.game_file_arg+ '"')+ str(self.ue4_params)
        assert(os.path.exists(self.ue4_exe_path)), "Unreal Editor executable:" + self.ue4_exe_path + "doesn't exist"
        assert(os.path.exists(self.game_file)), "game_file: " + self.game_file +  " doesn't exist"
        #assert(os.path.exists(self.press_play_file)), "press_play file: " + self.press_play_file +  " doesn't exist"


    def start_game_in_editor(self):
        # if not (settings.ip == '127.0.0.1'):
        #     print("can not start the game in a remote machine")
        #     exit(0)

        # Kill existing game instance if any, similar to Windows behavior
        self.kill_game_in_editor()
        time.sleep(2)

        if(os.name=="nt"):
            unreal_pids_before_launch = utils.find_process_id_by_name("UE4Editor.exe")
            # Ensure game_file is quoted in case of spaces
            self.cmd = str('"'+ self.ue4_exe_path+ '"')+" "+str('"'+ self.game_file+ '"')+ str(self.ue4_params)
            subprocess.Popen(self.cmd, shell=True)
            time.sleep(2)
        else:
            unreal_pids_before_launch = utils.find_process_id_by_name("UE4Editor")
            if not unreal_pids_before_launch and self.ue4_exe_path.endswith(".exe"):
                 unreal_pids_before_launch = utils.find_process_id_by_name("UE4Editor.exe")

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
            target_name = "UE4Editor.exe" if (os.name == "nt" or self.ue4_exe_path.endswith(".exe")) else "UE4Editor"
            current_pids = utils.find_process_id_by_name(target_name)
            diff_proc = (utils.list_diff(current_pids, unreal_pids_before_launch))

        settings.game_proc_pid = diff_proc[0]
        #time.sleep(30)
        client = airsim.MultirotorClient(settings.ip)
        connection_established = False
        connection_ctr = 0  # counting the number of time tried to connect
        # wait till connected to the multi rotor
        time.sleep(1)
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
            if "UE4Editor" in el:
                process1_exist = True
            if "CrashReportClient" in el:
                process2_exist = True
            if process1_exist and process2_exist:
                break

        # WSL2 Special Check: psutil cannot see Windows processes, so we must assume they might exist
        # or rely on the kill command's own error handling.
        wsl_windows_mode = (os.name == "posix" and self.ue4_exe_path.endswith(".exe"))

        if (settings.game_proc_pid == ''):  # if proc not provided, find any Unreal and kill
            if (process1_exist or wsl_windows_mode):
                if(os.name=="nt"):
                    os.system("taskkill /f /im  " + "UE4Editor.exe")
                elif(os.name=="posix"):
                    if self.ue4_exe_path.endswith(".exe"):
                        # Only run taskkill if in WSL/Win mode
                        os.system("taskkill.exe /f /im UE4Editor.exe > /dev/null 2>&1")
                    elif process1_exist: 
                        # Only run killall if process was actually found in Linux
                        os.system("killall "+ "UE4Editor")
        else:
            if(os.name=="nt"):
                os.system("taskkill /f /pid  " + str(settings.game_proc_pid))
                time.sleep(2)
                settings.game_proc_pid = ''
            elif(os.name=="posix"):
                if self.ue4_exe_path.endswith(".exe"):
                    os.system("taskkill.exe /f /pid " + str(settings.game_proc_pid) + " > /dev/null 2>&1")
                else:
                    if str(settings.game_proc_pid).strip():
                        os.system("kill " + str(settings.game_proc_pid))
                time.sleep(2)
                settings.game_proc_pid = ''

        if (process2_exist or wsl_windows_mode):
            if(os.name=="nt"):
                os.system("taskkill /f /im  " + "CrashReportClient.exe")
            elif(os.name=="posix"):
                if self.ue4_exe_path.endswith(".exe"):
                    os.system("taskkill.exe /f /im CrashReportClient.exe > /dev/null 2>&1")
                elif process2_exist:
                    os.system("killall " + "CrashReportClient")



    def restart_game(self):
        # if not (settings.ip == '127.0.0.1'):
        #     print("can not restart the game in a remote machine")
        #     exit(0)
        msgs.restart_game_count += 1
        self.kill_game_in_editor()  # kill in case there are any
        time.sleep(2)
        self.start_game_in_editor()

    def check_and_recover_game(self):
        """
        Checks if the UE4 process is running. If not, restarts the game.
        Returns True if the game was restarted, False otherwise.
        """
        # Determine likely process name
        target_name = "UE4Editor.exe" if (os.name == "nt" or self.ue4_exe_path.endswith(".exe")) else "UE4Editor"

        # Prefer tracking the known PID when available
        if str(settings.game_proc_pid).strip():
            try:
                proc = psutil.Process(int(settings.game_proc_pid))
                if proc.is_running() and proc.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ValueError):
                pass

        # WSL2: psutil cannot see Windows processes, use tasklist.exe
        wsl_windows_mode = (os.name == "posix" and self.ue4_exe_path.endswith(".exe"))
        if wsl_windows_mode:
            try:
                output = subprocess.check_output(
                    ["tasklist.exe", "/fi", f"imagename eq {target_name}"],
                    stderr=subprocess.DEVNULL,
                ).decode(errors="ignore")
                if target_name.lower() in output.lower():
                    return False
            except Exception:
                pass
        else:
            # Check running processes with more strict validation
            # We need to verify that found PIDs are not zombies
            raw_pids = utils.find_process_id_by_name(target_name)
            valid_pids = []

            if raw_pids:
                for pid in raw_pids:
                    try:
                        p = psutil.Process(pid)
                        if p.status() != psutil.STATUS_ZOMBIE and p.status() != psutil.STATUS_DEAD:
                            valid_pids.append(pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue

            if valid_pids:
                return False

        print("WARNING: UE4 process not found (or zombie) during monitoring! Initiating recovery...")
        self.restart_game()
        return True

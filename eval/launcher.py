#!/usr/bin/env python3
"""Launch the UE4 evaluation scene from WSL2 without EnvGenConfig.json."""

from __future__ import annotations

import csv
import io
import os
import platform
import subprocess
import time

from settings_folder import settings


DEFAULT_SCENE_PROJECT = "/mnt/d/Projects/ST/ST.uproject"
DEFAULT_UNREAL_EDITOR = "/mnt/d/SoftWare/Epic Games/Game/UE_4.18/Engine/Binaries/Win64/UE4Editor.exe"


def _to_windows_path(path: str) -> str:
    if platform.system() == "Linux" and "microsoft" in platform.uname().release.lower():
        try:
            return subprocess.check_output(["wslpath", "-w", path], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return path
    return path


class SceneGameHandler:
    """Minimal UE4 launcher for the evaluation scene.

    This deliberately does not read or write EnvGenConfig.json. It only opens the
    .uproject with UE4Editor and manages the editor process for evaluation.
    """

    def __init__(
        self,
        project_file: str = DEFAULT_SCENE_PROJECT,
        unreal_editor: str = DEFAULT_UNREAL_EDITOR,
        *,
        res_x: int | None = None,
        res_y: int | None = None,
        win_x: int | None = None,
        win_y: int | None = None,
    ) -> None:
        self.project_file = os.path.abspath(os.path.expanduser(project_file))
        self.unreal_editor = os.path.abspath(os.path.expanduser(unreal_editor))
        self.project_arg = _to_windows_path(self.project_file)
        self.process_pid = ""
        self.res_x = int(res_x if res_x is not None else settings.game_resX)
        self.res_y = int(res_y if res_y is not None else settings.game_resY)
        self.win_x = int(win_x if win_x is not None else settings.ue4_winX)
        self.win_y = int(win_y if win_y is not None else settings.ue4_winY)

        if not os.path.exists(self.project_file):
            raise FileNotFoundError(f"Evaluation scene project file not found: {self.project_file}")
        if not os.path.exists(self.unreal_editor):
            raise FileNotFoundError(f"UE4Editor executable not found: {self.unreal_editor}")

    def _cmd(self) -> str:
        params = (
            f" -game -ResX={self.res_x} -ResY={self.res_y}"
            f" -WinX={self.win_x} -WinY={self.win_y}"
            " -Windowed -NOPAUSE -NOSOUND"
        )
        return f'"{self.unreal_editor}" "{self.project_arg}"{params}'

    def _find_windows_pids(self, image_name: str) -> list[str]:
        try:
            output = subprocess.check_output(
                ["tasklist.exe", "/fi", f"IMAGENAME eq {image_name}", "/fo", "csv", "/nh"],
                stderr=subprocess.DEVNULL,
            ).decode(errors="ignore")
        except Exception:
            return []
        if "No tasks are running" in output:
            return []
        rows = list(csv.reader(io.StringIO(output)))
        return [row[1].strip() for row in rows if len(row) >= 2 and row[0].strip().lower() == image_name.lower()]

    def _find_linux_pids(self, process_name: str) -> list[str]:
        try:
            output = subprocess.check_output(["pgrep", "-f", process_name], stderr=subprocess.DEVNULL).decode()
        except Exception:
            return []
        return [line.strip() for line in output.splitlines() if line.strip()]

    def _find_editor_pids(self) -> list[str]:
        if self.unreal_editor.endswith(".exe"):
            return self._find_windows_pids("UE4Editor.exe")
        return self._find_linux_pids("UE4Editor")

    def start_game_in_editor(self) -> None:
        self.kill_game_in_editor()
        time.sleep(2)

        before = set(self._find_editor_pids())
        subprocess.Popen(self._cmd(), shell=True)

        deadline = time.time() + 180
        while time.time() < deadline:
            time.sleep(5)
            current = set(self._find_editor_pids())
            diff_proc = sorted(current - before)
            if diff_proc:
                self.process_pid = str(diff_proc[0])
                break
        else:
            raise RuntimeError("Timed out waiting for UE4Editor process to start.")

        time.sleep(35)
        print("UE4 test scene started. Ready for AirSim connection.")

    def restart_game(self) -> None:
        self.kill_game_in_editor()
        time.sleep(2)
        self.start_game_in_editor()

    def kill_game_in_editor(self) -> None:
        if self.unreal_editor.endswith(".exe"):
            if self.process_pid:
                os.system(f"taskkill.exe /f /pid {self.process_pid} > /dev/null 2>&1")
                self.process_pid = ""
            else:
                os.system("taskkill.exe /f /im UE4Editor.exe > /dev/null 2>&1")
            os.system("taskkill.exe /f /im CrashReportClient.exe > /dev/null 2>&1")
            return

        if self.process_pid:
            os.system(f"kill {self.process_pid} > /dev/null 2>&1")
            self.process_pid = ""
        else:
            os.system("killall UE4Editor > /dev/null 2>&1")

    def is_game_process_alive(self) -> bool:
        if self.unreal_editor.endswith(".exe"):
            if self.process_pid:
                try:
                    output = subprocess.check_output(
                        ["tasklist.exe", "/fi", f"PID eq {self.process_pid}", "/fo", "csv", "/nh"],
                        stderr=subprocess.DEVNULL,
                    ).decode(errors="ignore")
                    if "No tasks are running" not in output:
                        rows = list(csv.reader(io.StringIO(output)))
                        return any(len(row) >= 2 and row[1].strip() == self.process_pid for row in rows)
                except Exception:
                    return False
            try:
                output = subprocess.check_output(
                    ["tasklist.exe", "/fi", "IMAGENAME eq UE4Editor.exe", "/fo", "csv", "/nh"],
                    stderr=subprocess.DEVNULL,
                ).decode(errors="ignore")
                return "UE4Editor.exe" in output
            except Exception:
                return False

        return bool(self._find_editor_pids())

# PyQt鼠标连点器

### 🧑‍💻 User

使用pyqt编写一个鼠标连点器，要求可以设置轨迹随机偏移、可编程点击频率等功能

---

### 🤖 Assistant

下面是一个功能完整的 PyQt5 鼠标连点器，包含轨迹随机偏移、可编程频率、热键控制等高级特性。

```python
#!/usr/bin/env python3
"""
PyQt5 鼠标连点器
- 可编程点击频率（CPS + 随机抖动）
- 轨迹随机偏移（模拟人类移动轨迹）
- 全局热键启停
- 多种点击模式
"""

import sys
import random
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QComboBox, QStatusBar, QGridLayout, QFrame, QSlider,
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPoint
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

# ── 平台兼容：全局热键 & 鼠标控制 ──────────────────────────────────
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    # --- 鼠标控制 (Windows) ---
    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]

    INPUT_MOUSE = 0
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_MIDDLEDOWN = 0x0020
    MOUSEEVENTF_MIDDLEUP = 0x0040
    MOUSEEVENTF_ABSOLUTE = 0x8000

    def _send_input(flags, dx=0, dy=0, data=0):
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi.dx = dx
        inp.mi.dy = dy
        inp.mi.mouseData = data
        inp.mi.dwFlags = flags
        inp.mi.time = 0
        inp.mi.dwExtraInfo = None
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def mouse_move(x: int, y: int):
        """绝对移动"""
        _send_input(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, int(x * 65535 // 65535), int(y * 65535 // 65535))

    def mouse_move_relative(dx: int, dy: int):
        """相对移动"""
        _send_input(MOUSEEVENTF_MOVE, dx, dy)

    def mouse_left_down():
        _send_input(MOUSEEVENTF_LEFTDOWN)

    def mouse_left_up():
        _send_input(MOUSEEVENTF_LEFTUP)

    def mouse_right_down():
        _send_input(MOUSEEVENTF_RIGHTDOWN)

    def mouse_right_up():
        _send_input(MOUSEEVENTF_RIGHTUP)

    def mouse_middle_down():
        _send_input(MOUSEEVENTF_MIDDLEDOWN)

    def mouse_middle_up():
        _send_input(MOUSEEVENTF_MIDDLEUP)

    def get_cursor_pos():
        pt = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return (pt.x, pt.y)

    # --- 全局热键 (Windows) ---
    MOD_ALT = 0x0001
    MOD_CONTROL = 0x0002
    MOD_SHIFT = 0x0004
    MOD_NOREPEAT = 0x4000
    VK_F6 = 0x75
    VK_F7 = 0x76
    VK_F8 = 0x77
    WM_HOTKEY = 0x0312

    def register_hotkey(hwnd, id_, modifiers, vk):
        return ctypes.windll.user32.RegisterHotKey(hwnd, id_, modifiers, vk)

    def unregister_hotkey(hwnd, id_):
        ctypes.windll.user32.UnregisterHotKey(hwnd, id_)

else:
    # 非 Windows 平台的占位实现（需要额外的库如 pynput）
    def mouse_move(x, y): pass
    def mouse_move_relative(dx, dy): pass
    def mouse_left_down(): pass
    def mouse_left_up(): pass
    def mouse_right_down(): pass
    def mouse_right_up(): pass
    def mouse_middle_down(): pass
    def mouse_middle_up(): pass
    def get_cursor_pos(): return (0, 0)
    def register_hotkey(hwnd, id_, mod, vk): return False
    def unregister_hotkey(hwnd, id_): pass
    MOD_NOREPEAT = 0
    VK_F6 = VK_F7 = VK_F8 = 0
    WM_HOTKEY = 0


# ── 轨迹偏移配置 ──────────────────────────────────────────────────
@dataclass
class TrajectoryConfig:
    """轨迹偏移配置"""
    enabled: bool = True
    max_offset: int = 8            # 最大偏移像素
    step_count: int = 3            # 偏移步数（步数越多越平滑）
    step_delay_ms: int = 5         # 每步延迟(ms)
    bezier_smoothing: bool = True  # 使用贝塞尔曲线平滑


# ── 点击频率配置 ──────────────────────────────────────────────────
@dataclass
class FrequencyConfig:
    """点击频率配置"""
    cps: float = 10.0              # 基础每秒点击次数
    jitter_enabled: bool = True    # 是否启用频率抖动
    jitter_percent: float = 15.0   # 抖动百分比（0-100）
    hold_ms: int = 15              # 按下持续时间(ms)


# ── 轨迹生成器 ────────────────────────────────────────────────────
class TrajectoryGenerator:
    """生成随机偏移轨迹"""

    @staticmethod
    def _bezier_point(t: float, p0, p1, p2):
        """二次贝塞尔曲线"""
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        return (x, y)

    @staticmethod
    def generate(config: TrajectoryConfig) -> list:
        """生成偏移轨迹点列表 [(dx, dy), ...]"""
        if not config.enabled:
            return [(0, 0)]

        # 随机方向
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(1, config.max_offset)

        # 目标偏移
        target_dx = math.cos(angle) * distance
        target_dy = math.sin(angle) * distance

        steps = config.step_count
        if steps < 1:
            steps = 1

        points = []

        if config.bezier_smoothing:
            # 使用贝塞尔曲线：起点(0,0)、控制点、终点(target)
            cp_x = random.uniform(-config.max_offset * 0.3, target_dx * 1.3)
            cp_y = random.uniform(-config.max_offset * 0.3, target_dy * 1.3)
            control = (cp_x, cp_y)

            for i in range(steps):
                t = (i + 1) / steps
                pt = TrajectoryGenerator._bezier_point(t, (0, 0), control, (target_dx, target_dy))
                points.append(pt)
        else:
            # 线性插值 + 微扰动
            for i in range(steps):
                progress = (i + 1) / steps
                dx = target_dx * progress + random.uniform(-1, 1)
                dy = target_dy * progress + random.uniform(-1, 1)
                points.append((dx, dy))

        return points


# ── 连点工作线程 ──────────────────────────────────────────────────
class ClickerWorker(QThread):
    """连点器工作线程（不阻塞 UI）"""
    status_changed = pyqtSignal(str)
    click_performed = pyqtSignal(int)  # 已点击次数

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._paused = False

        # 配置
        self.freq_config = FrequencyConfig()
        self.traj_config = TrajectoryConfig()

        # 点击设置
        self.click_button = "left"       # left/right/middle
        self.click_mode = "single"       # single/double
        self.target_position = None      # None=跟随光标, (x,y)=固定位置
        self.click_limit = 0             # 0=无限

        self._click_count = 0
        self._traj_gen = TrajectoryGenerator()

    def run(self):
        self._running = True
        self._click_count = 0
        self.status_changed.emit("▶ 运行中...")

        while self._running:
            if self._paused:
                self.msleep(50)
                continue

            # 点击次数限制
            if self.click_limit > 0 and self._click_count >= self.click_limit:
                self._running = False
                self.status_changed.emit(f"✓ 已完成（{self._click_count} 次点击）")
                break

            # 获取目标位置
            if self.target_position:
                tx, ty = self.target_position
            else:
                tx, ty = get_cursor_pos()

            # 执行轨迹偏移
            if self.traj_config.enabled:
                trajectory = self._traj_gen.generate(self.traj_config)
                # 先移动到起始位置（如果不是跟随光标模式）
                if self.target_position:
                    mouse_move(tx, ty)
                    self.msleep(2)

                # 执行偏移轨迹
                for dx, dy in trajectory:
                    mouse_move_relative(int(dx), int(dy))
                    self.msleep(self.traj_config.step_delay_ms)

                # 轨迹完成后的最终位置就是点击位置
                # （相对移动后已在目标附近）

            # 执行点击
            self._perform_click()

            self._click_count += 1
            self.click_performed.emit(self._click_count)

            # 计算下一次点击间隔
            delay = self._calc_delay()
            self.msleep(max(1, int(delay)))

        if not self._paused:
            self.status_changed.emit("⏹ 已停止")

    def _perform_click(self):
        """执行一次点击"""
        btn_map = {
            "left": (mouse_left_down, mouse_left_up),
            "right": (mouse_right_down, mouse_right_up),
            "middle": (mouse_middle_down, mouse_middle_up),
        }
        down, up = btn_map.get(self.click_button, (mouse_left_down, mouse_left_up))

        times = 2 if self.click_mode == "double" else 1
        for i in range(times):
            down()
            self.msleep(self.freq_config.hold_ms)
            up()
            if i == 0 and times > 1:
                self.msleep(max(5, self.freq_config.hold_ms * 2))

    def _calc_delay(self) -> float:
        """计算下一次点击延迟（毫秒）"""
        base_delay = 1000.0 / max(0.1, self.freq_config.cps)

        if self.freq_config.jitter_enabled:
            jitter = self.freq_config.jitter_percent / 100.0
            factor = random.uniform(1 - jitter, 1 + jitter)
            delay = base_delay * factor
        else:
            delay = base_delay

        return max(10, delay)

    def stop(self):
        self._running = False
        self._paused = False

    def pause(self):
        self._paused = True
        self.status_changed.emit("⏸ 已暂停")

    def resume(self):
        self._paused = False
        self.status_changed.emit("▶ 运行中...")

    def toggle_pause(self):
        if self._paused:
            self.resume()
        else:
            self.pause()


# ── 主窗口 UI ─────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    HOTKEY_START = 1
    HOTKEY_STOP = 2
    HOTKEY_PAUSE = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle("鼠标连点器 v2.0")
        self.setFixedSize(480, 620)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)

        # 暗色主题
        self._apply_dark_theme()

        # 连点器工作线程
        self.worker = ClickerWorker()
        self.worker.status_changed.connect(self._on_status)
        self.worker.click_performed.connect(self._on_click)

        self._init_ui()
        self._register_hotkeys()

    # ── 暗色主题 ──────────────────────────────────────────────
    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 48))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
        palette.setColor(QPalette.ToolTipBase, QColor(45, 45, 48))
        palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(60, 60, 65))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 10px;
                color: #ddd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4db8ff;
            }
            QPushButton {
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover { background: #555; }
            QPushButton#startBtn {
                background: #1a8a3f; color: #fff;
            }
            QPushButton#startBtn:hover { background: #20a84d; }
            QPushButton#stopBtn {
                background: #b33; color: #fff;
            }
            QPushButton#stopBtn:hover { background: #d44; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background: #2d2d2d; color: #ddd;
                border: 1px solid #555; padding: 3px;
                border-radius: 3px;
            }
            QLabel { color: #ccc; }
        """)

    # ── 构建 UI ───────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 10, 12, 10)

        # ── 频率设置 ──
        freq_group = QGroupBox("⏱ 点击频率")
        fl = QGridLayout(freq_group)
        fl.setSpacing(6)

        fl.addWidget(QLabel("每秒点击 (CPS):"), 0, 0)
        self.spin_cps = QDoubleSpinBox()
        self.spin_cps.setRange(0.5, 1000)
        self.spin_cps.setValue(10.0)
        self.spin_cps.setDecimals(1)
        self.spin_cps.setSingleStep(1.0)
        self.spin_cps.setSuffix(" cps")
        fl.addWidget(self.spin_cps, 0, 1)

        fl.addWidget(QLabel("按下持续时间:"), 1, 0)
        self.spin_hold = QSpinBox()
        self.spin_hold.setRange(1, 500)
        self.spin_hold.setValue(15)
        self.spin_hold.setSuffix(" ms")
        fl.addWidget(self.spin_hold, 1, 1)

        self.chk_jitter = QCheckBox("启用频率随机抖动")
        self.chk_jitter.setChecked(True)
        fl.addWidget(self.chk_jitter, 2, 0)

        self.spin_jitter = QDoubleSpinBox()
        self.spin_jitter.setRange(1, 50)
        self.spin_jitter.setValue(15)
        self.spin_jitter.setDecimals(1)
        self.spin_jitter.setSuffix(" %")
        fl.addWidget(self.spin_jitter, 2, 1)

        layout.addWidget(freq_group)

        # ── 轨迹偏移设置 ──
        traj_group = QGroupBox("🎯 轨迹随机偏移")
        tl = QGridLayout(traj_group)
        tl.setSpacing(6)

        self.chk_traj = QCheckBox("启用轨迹偏移（模拟人类）")
        self.chk_traj.setChecked(True)
        tl.addWidget(self.chk_traj, 0, 0, 1, 2)

        tl.addWidget(QLabel("最大偏移:"), 1, 0)
        self.spin_offset = QSpinBox()
        self.spin_offset.setRange(1, 50)
        self.spin_offset.setValue(8)
        self.spin_offset.setSuffix(" px")
        tl.addWidget(self.spin_offset, 1, 1)

        tl.addWidget(QLabel("偏移步数:"), 2, 0)
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 20)
        self.spin_steps.setValue(3)
        tl.addWidget(self.spin_steps, 2, 1)

        tl.addWidget(QLabel("每步延迟:"), 3, 0)
        self.spin_step_delay = QSpinBox()
        self.spin_step_delay.setRange(1, 50)
        self.spin_step_delay.setValue(5)
        self.spin_step_delay.setSuffix(" ms")
        tl.addWidget(self.spin_step_delay, 3, 1)

        self.chk_bezier = QCheckBox("贝塞尔曲线平滑")
        self.chk_bezier.setChecked(True)
        tl.addWidget(self.chk_bezier, 4, 0, 1, 2)

        layout.addWidget(traj_group)

        # ── 点击设置 ──
        click_group = QGroupBox("🖱 点击设置")
        cl = QGridLayout(click_group)
        cl.setSpacing(6)

        cl.addWidget(QLabel("鼠标按键:"), 0, 0)
        self.combo_button = QComboBox()
        self.combo_button.addItems(["左键 (Left)", "右键 (Right)", "中键 (Middle)"])
        cl.addWidget(self.combo_button, 0, 1)

        cl.addWidget(QLabel("点击模式:"), 1, 0)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["单击 (Single)", "双击 (Double)"])
        cl.addWidget(self.combo_mode, 1, 1)

        cl.addWidget(QLabel("目标位置:"), 2, 0)
        self.combo_position = QComboBox()
        self.combo_position.addItems(["跟随光标", "固定位置（点击设置）"])
        cl.addWidget(self.combo_position, 2, 1)

        self.btn_set_pos = QPushButton("📍 设置固定位置")
        self.btn_set_pos.clicked.connect(self._set_fixed_position)
        self.btn_set_pos.setEnabled(False)
        cl.addWidget(self.btn_set_pos, 3, 0, 1, 2)

        self.lbl_fixed_pos = QLabel("未设置")
        self.lbl_fixed_pos.setStyleSheet("color: #888;")
        cl.addWidget(self.lbl_fixed_pos, 3, 1)

        cl.addWidget(QLabel("点击次数限制:"), 4, 0)
        self.spin_limit = QSpinBox()
        self.spin_limit.setRange(0, 99999)
        self.spin_limit.setValue(0)
        self.spin_limit.setSpecialValueText("无限")
        cl.addWidget(self.spin_limit, 4, 1)

        layout.addWidget(click_group)

        # ── 控制按钮 ──
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("▶ 开始 (F6)")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.clicked.connect(self.start_clicker)
        btn_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸ 暂停 (F8)")
        self.btn_pause.clicked.connect(self.pause_clicker)
        self.btn_pause.setEnabled(False)
        btn_layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹ 停止 (F7)")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.stop_clicker)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        layout.addLayout(btn_layout)

        # ── 状态栏 ──
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_frame.setStyleSheet("QFrame { background: #1a1a1a; border-radius: 4px; padding: 4px; }")
        sf_layout = QHBoxLayout(status_frame)
        sf_layout.setContentsMargins(8, 4, 8, 4)

        self.lbl_status = QLabel("⏹ 就绪 - 按 F6 开始")
        self.lbl_status.setFont(QFont("Consolas", 10))
        sf_layout.addWidget(self.lbl_status)

        self.lbl_count = QLabel("点击: 0")
        self.lbl_count.setFont(QFont("Consolas", 10))
        self.lbl_count.setStyleSheet("color: #4db8ff;")
        sf_layout.addWidget(self.lbl_count)

        layout.addWidget(status_frame)

        # ── 快捷键提示 ──
        hint_label = QLabel("⌨ 全局热键:  F6=开始  |  F7=停止  |  F8=暂停/继续")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(hint_label)

    # ── 固定位置 ─────────────────────────────────────────────
    def _set_fixed_position(self):
        """3秒后记录当前位置"""
        self.btn_set_pos.setText("⏳ 3秒后记录位置，请移动鼠标...")
        self.btn_set_pos.setEnabled(False)
        self._countdown = 3
        self._countdown_timer = QTimer()
        self._countdown_timer.timeout.connect(self._countdown_tick)
        self._countdown_timer.start(1000)

    def _countdown_tick(self):
        self._countdown -= 1
        if self._countdown > 0:
            self.btn_set_pos.setText(f"⏳ {self._countdown}秒后记录位置...")
        else:
            self._countdown_timer.stop()
            pos = get_cursor_pos()
            self.worker.target_position = pos
            self.lbl_fixed_pos.setText(f"({pos[0]}, {pos[1]})")
            self.lbl_fixed_pos.setStyleSheet("color: #4db8ff;")
            self.btn_set_pos.setText("📍 重新设置位置")
            self.btn_set_pos.setEnabled(True)

    # ── 热键注册 ─────────────────────────────────────────────
    def _register_hotkeys(self):
        if sys.platform == "win32":
            hwnd = int(self.winId())
            register_hotkey(hwnd, self.HOTKEY_START, MOD_NOREPEAT, VK_F6)
            register_hotkey(hwnd, self.HOTKEY_STOP, MOD_NOREPEAT, VK_F7)
            register_hotkey(hwnd, self.HOTKEY_PAUSE, MOD_NOREPEAT, VK_F8)

    def nativeEvent(self, eventType, message):
        """处理 Windows 热键消息"""
        if sys.platform == "win32" and eventType == "windows_generic_MSG":
            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY:
                if msg.wParam == self.HOTKEY_START:
                    self.start_clicker()
                elif msg.wParam == self.HOTKEY_STOP:
                    self.stop_clicker()
                elif msg.wParam == self.HOTKEY_PAUSE:
                    self.pause_clicker()
                return True, 0
        return super().nativeEvent(eventType, message)

    # ── 控制方法 ─────────────────────────────────────────────
    def _gather_config(self):
        """从 UI 收集配置到 worker"""
        self.worker.freq_config.cps = self.spin_cps.value()
        self.worker.freq_config.hold_ms = self.spin_hold.value()
        self.worker.freq_config.jitter_enabled = self.chk_jitter.isChecked()
        self.worker.freq_config.jitter_percent = self.spin_jitter.value()

        self.worker.traj_config.enabled = self.chk_traj.isChecked()
        self.worker.traj_config.max_offset = self.spin_offset.value()
        self.worker.traj_config.step_count = self.spin_steps.value()
        self.worker.traj_config.step_delay_ms = self.spin_step_delay.value()
        self.worker.traj_config.bezier_smoothing = self.chk_bezier.isChecked()

        btn_map = {0: "left", 1: "right", 2: "middle"}
        self.worker.click_button = btn_map[self.combo_button.currentIndex()]
        self.worker.click_mode = "double" if self.combo_mode.currentIndex() == 1 else "single"
        self.worker.click_limit = self.spin_limit.value()

        # 位置模式
        if self.combo_position.currentIndex() == 0:
            self.worker.target_position = None
            self.btn_set_pos.setEnabled(False)
            self.lbl_fixed_pos.setText("跟随光标")
            self.lbl_fixed_pos.setStyleSheet("color: #888;")
        # 固定位置由 _set_fixed_position 设置

    def start_clicker(self):
        if self.worker.isRunning():
            return
        self._gather_config()
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ 暂停 (F8)")

    def stop_clicker(self):
        self.worker.stop()
        self.worker.wait(200)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)

    def pause_clicker(self):
        if not self.worker.isRunning():
            return
        self.worker.toggle_pause()
        if self.worker._paused:
            self.btn_pause.setText("▶ 继续 (F8)")
        else:
            self.btn_pause.setText("⏸ 暂停 (F8)")

    def _on_status(self, msg):
        self.lbl_status.setText(msg)

    def _on_click(self, count):
        self.lbl_count.setText(f"点击: {count}")

    # ── 窗口关闭 ─────────────────────────────────────────────
    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(500)
        # 注销热键
        if sys.platform == "win32":
            hwnd = int(self.winId())
            unregister_hotkey(hwnd, self.HOTKEY_START)
            unregister_hotkey(hwnd, self.HOTKEY_STOP)
            unregister_hotkey(hwnd, self.HOTKEY_PAUSE)
        event.accept()


# ── 入口 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Windows 下设置 DPI 感知
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_PER_MONITOR_DPI_AWARE
        except:
            pass

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
```

## 功能概览

| 模块 | 功能说明 |
|------|----------|
| **点击频率** | 可设 CPS（0.5–1000），支持 ±N% 随机抖动 + 按下持续时间 |
| **轨迹偏移** | 贝塞尔曲线平滑的随机方向偏移，可调最大偏移/步数/每步延迟 |
| **全局热键** | `F6` 开始、`F7` 停止、`F8` 暂停/继续（Windows 下全局生效） |
| **点击模式** | 左/右/中键、单击/双击 |
| **目标位置** | 跟随光标 / 3秒倒计时设置固定位置 |
| **次数限制** | 可设执行 N 次后自动停止 |

## 安装与运行

```bash
pip install PyQt5
python clicker.py
```

> **注意**：轨迹偏移和全局热键目前基于 Windows API (`SendInput` / `RegisterHotKey`)。macOS/Linux 用户需将底层的 `mouse_*` 和 `register_hotkey` 替换为 `pynput` 或 `pyautogui` 实现即可跨平台使用。
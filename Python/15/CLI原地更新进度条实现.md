# CLI原地更新进度条实现

### 🧑‍💻 User

如何实现tqdm/rich progressbar那种在cli中原地更新log而不是大量log刷屏？

---

### 🤖 Assistant

要在 CLI 中实现原地更新的 progress bar，核心原理是**使用 ANSI 转义序列控制光标位置**，而不是不断打印新行。以下是几种实现方式：

## 1. 最简实现：使用 `\r` 回车符

```python
import time
import sys

def simple_progress():
    for i in range(101):
        # \r 将光标移回行首，end='' 不换行
        sys.stdout.write(f'\rProgress: {i}% [{"#" * i}{"." * (100 - i)}]')
        sys.stdout.flush()  # 立即刷新缓冲区
        time.sleep(0.05)
    print()  # 最后换行

simple_progress()
```

## 2. 使用 ANSI 转义序列（更灵活）

```python
import sys
import time

def ansi_progress():
    # 隐藏光标
    sys.stdout.write('\033[?25l')
  
    for i in range(101):
        # \033[K 清除当前行
        # \033[G 移动到行首
        bar = '#' * (i // 2) + '-' * (50 - i // 2)
        sys.stdout.write(f'\033[G\033[KProgress: [{bar}] {i:3d}%')
        sys.stdout.flush()
        time.sleep(0.05)
  
    sys.stdout.write('\n')
    # 显示光标
    sys.stdout.write('\033[?25h')

ansi_progress()
```

### 常用 ANSI 转义序列速查

```python
# 光标的移动
CURSOR_UP = '\033[A'       # 上移 n 行：\033[nA
CURSOR_DOWN = '\033[B'     # 下移 n 行
CURSOR_FORWARD = '\033[C'  # 右移 n 列
CURSOR_BACK = '\033[D'     # 左移 n 列
CURSOR_TO_LINE_START = '\033[G'  # 移至行首
CURSOR_TO_COL = '\033[{n}G'      # 移至第 n 列

# 清除屏幕内容
CLEAR_LINE = '\033[K'      # 清除从光标到行尾
CLEAR_LINE_ALL = '\033[2K' # 清除整行
CLEAR_SCREEN_DOWN = '\033[J'     # 清除从光标到屏幕底
CLEAR_SCREEN_ALL = '\033[2J'     # 清除整个屏幕

# 光标的显示/隐藏
HIDE_CURSOR = '\033[?25l'
SHOW_CURSOR = '\033[?25h'

# 保存/恢复光标位置
SAVE_CURSOR = '\033[s'
RESTORE_CURSOR = '\033[u'
```

## 3. 多行进度条实现

```python
import sys
import time
import threading

class MultiProgress:
    def __init__(self, num_lines=3):
        self.num_lines = num_lines
        self.lines = [[] for _ in range(num_lines)]
        self.lock = threading.Lock()
        # 预留空间
        sys.stdout.write('\n' * num_lines)
        sys.stdout.flush()
  
    def update(self, line_idx, text):
        with self.lock:
            # 保存光标位置
            sys.stdout.write('\033[s')
            # 向上移动到目标行
            sys.stdout.write(f'\033[{self.num_lines - line_idx}A')
            # 清除并重写该行
            sys.stdout.write(f'\033[G\033[2K{text}')
            # 恢复光标位置
            sys.stdout.write('\033[u')
            sys.stdout.flush()

def demo():
    mp = MultiProgress(3)
    for i in range(101):
        mp.update(0, f'Task 1: [{"#" * (i//5)}{"." * (20-i//5)}] {i}%')
        mp.update(1, f'Task 2: [{"#" * ((i//3))}{"." * (33-i//3)}] {i}%')
        mp.update(2, f'Status: Processing item {i}/100')
        time.sleep(0.03)
    print('\nDone!')

demo()
```

## 4. 封装成可复用的 ProgressBar 类

```python
import sys
import time
from typing import Optional

class ProgressBar:
    def __init__(
        self,
        total: int,
        prefix: str = '',
        suffix: str = '',
        length: int = 50,
        fill: str = '█',
        empty: str = '░',
    ):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.empty = empty
        self.current = 0
      
    def update(self, n: int = 1, suffix: Optional[str] = None):
        self.current += n
        percent = self.current / self.total
        filled = int(self.length * percent)
        bar = self.fill * filled + self.empty * (self.length - filled)
      
        if suffix is not None:
            self.suffix = suffix
          
        line = f'\r{self.prefix} |{bar}| {percent:.1%} {self.suffix}'
        sys.stdout.write(line)
        sys.stdout.flush()
      
        if self.current >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()
  
    def __enter__(self):
        sys.stdout.write('\033[?25l')  # 隐藏光标
        return self
  
    def __exit__(self, *args):
        sys.stdout.write('\033[?25h')  # 显示光标

# 使用示例
with ProgressBar(100, prefix='Downloading', suffix='Complete', length=30) as pb:
    for _ in range(100):
        time.sleep(0.03)
        pb.update(1)
```

## 5. 关键技术点总结

| 技术 | 说明 |
|------|------|
| `\r` | 最简单，只能在同一行更新 |
| `sys.stdout.write()` + `flush()` | 避免 print 的自动换行 |
| ANSI `\033[A` | 多行场景，上移光标 |
| ANSI `\033[2K` | 清除整行再重写 |
| ANSI `\033[s` / `\033[u` | 保存/恢复光标位置 |
| ANSI `\033[?25l/h` | 隐藏/显示光标（消除闪烁） |

**核心思路**：移动光标 → 清除旧内容 → 写入新内容 → 恢复光标 → 刷新缓冲区。这就是 tqdm 和 rich 底层的工作方式。
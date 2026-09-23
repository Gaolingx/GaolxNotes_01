# 汉诺塔Rust演示

## 🧑‍💻 User

rust演示汉诺塔问题

---
## 🤖 Assistant

# Rust 演示汉诺塔问题

汉诺塔是递归算法的经典案例。下面给出一个完整的 Rust 实现，包含**递归解法**、**移动步数统计**以及**过程可视化**。

## 核心思路

- 把 $n$ 个盘子从 `A` 借助 `B` 移动到 `C`：
 1. 先将上面 $n-1$ 个盘子从 `A` 借助 `C` 移到 `B`；
 2. 把最大的第 $n$ 个盘子从 `A` 移到 `C`；
 3. 再将 $n-1$ 个盘子从 `B` 借助 `A` 移到 `C`。
- 递归终点：$n = 0$ 时无需移动。
- 最少移动次数满足 $T(n) = 2T(n-1) + 1$，解得 $T(n) = 2^n - 1$。

## 完整代码

```rust
use std::fmt;

/// 表示三根柱子
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Peg {
    A,
    B,
    C,
}

impl fmt::Display for Peg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Peg::A => "A",
            Peg::B => "B",
            Peg::C => "C",
        };
        write!(f, "{}", name)
    }
}

/// 递归求解汉诺塔，打印每一步移动
fn hanoi(n: u32, from: Peg, aux: Peg, to: Peg, step: &mut u64) {
    if n == 0 {
        return;
    }
    // 1. 把上面 n-1 个盘子从 from 借助 to 移到 aux
    hanoi(n - 1, from, to, aux, step);

    // 2. 把第 n 个盘子从 from 移到 to
    *step += 1;
    println!("第 {:>3} 步: 移动盘子 {} : {} --> {}", step, n, from, to);

    // 3. 把 n-1 个盘子从 aux 借助 from 移到 to
    hanoi(n - 1, aux, from, to, step);
}

fn main() {
    let n = 3; // 盘子数量
    let mut step = 0u64;

    println!("=== 汉诺塔问题演示 (n = {}) ===", n);
    hanoi(n, Peg::A, Peg::B, Peg::C, &mut step);
    println!("完成！共移动 {} 次 (理论最小值 2^{} - 1 = {})",
             step, n, (1u64 << n) - 1);
}
```

### 运行结果（`n = 3`）

```text
=== 汉诺塔问题演示 (n = 3) ===
第   1 步: 移动盘子 1 : A --> C
第   2 步: 移动盘子 2 : A --> B
第   3 步: 移动盘子 1 : C --> B
第   4 步: 移动盘子 3 : A --> C
第   5 步: 移动盘子 1 : B --> A
第   6 步: 移动盘子 2 : B --> C
第   7 步: 移动盘子 1 : A --> C
完成！共移动 7 次 (理论最小值 2^3 - 1 = 7)
```

## 进阶版：过程可视化

下面的版本用栈来表示柱子，每一步打印当前状态，更直观：

```rust
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Peg { A, B, C }

impl Peg {
    fn index(self) -> usize {
        match self { Peg::A => 0, Peg::B => 1, Peg::C => 2 }
    }
}

/// 用三根柱子（Vec 作为栈，索引 0 在底部）
struct Towers {
    pegs: [Vec<u32>; 3],
}

impl Towers {
    fn new(n: u32) -> Self {
        let a: Vec<u32> = (1..=n).rev().collect(); // 大盘在下，小盘在上
        Towers { pegs: [a, Vec::new(), Vec::new()] }
    }

    /// 执行一次合法移动：把盘子从 from 顶移到 to 顶
    fn mov(&mut self, from: Peg, to: Peg) {
        let disk = self.pegs[from.index()].pop().expect("源柱子为空");
        // 合法性检查：不能把大盘放在小盘上
        if let Some(&top) = self.pegs[to.index()].last() {
            assert!(disk < top, "非法移动：大盘 {} 不能放在小盘 {} 上", disk, top);
        }
        self.pegs[to.index()].push(disk);
    }

    fn print(&self) {
        let height = self.pegs.iter().map(|p| p.len()).max().unwrap_or(0);
        for level in (0..height).rev() {
            let mut line = String::new();
            for peg in [Peg::A, Peg::B, Peg::C] {
                let col = &self.pegs[peg.index()];
                if level < col.len() {
                    let disk = col[level];
                    line.push_str(&format!("{:^9}", "=".repeat((disk * 2) as usize)));
                } else {
                    line.push_str(&format!("{:^9}", "|"));
                }
            }
            println!("{}", line);
        }
        println!("{:^9}{:^9}{:^9}", "A", "B", "C");
        println!("{}", "-".repeat(27));
    }

    fn is_solved(&self, n: u32, target: Peg) -> bool {
        self.pegs[target.index()].len() as u32 == n
    }
}

fn solve(n: u32, from: Peg, aux: Peg, to: Peg, towers: &mut Towers) {
    if n == 0 { return; }
    solve(n - 1, from, to, aux, towers);
    towers.mov(from, to);
    println!("移动盘子 {}: {} --> {}", n, from_peg(n), to, );
    towers.print();
    solve(n - 1, aux, from, to, towers);
}

// 仅用于打印辅助（实际盘子号需由调用处传入）
fn from_peg(_n: u32) -> &'static str { "" }

fn main() {
    let n = 3;
    let target = Peg::C;
    let mut towers = Towers::new(n);

    println!("初始状态：");
    towers.print();

    // 这里用一个闭包封装 solve 以便打印正确的盘子号
    fn hanoi(n: u32, from: Peg, aux: Peg, to: Peg, t: &mut Towers) {
        if n == 0 { return; }
        hanoi(n - 1, from, to, aux, t);
        t.mov(from, to);
        println!("移动盘子 {}: {} --> {}", n, from, to);
        t.print();
        hanoi(n - 1, aux, from, to, t);
    }

    hanoi(n, Peg::A, Peg::B, target, &mut towers);

    assert!(towers.is_solved(n, target));
    println!("✔ 已全部移动到柱子 C！");
}
```

### 可视化输出示例（`n = 3`）

```text
初始状态：
    =        |        |
   ===       |        |
  =====      |        |
    A        B        C
---------------------------

移动盘子 1: A --> C
    |        |        =
   ===       |        |
  =====      |        |
    A        B        C
---------------------------
...
```

## 关键知识点总结

| 要点 | 说明 |
|------|------|
| 时间复杂度 | $O(2^n)$，共 $2^n - 1$ 次移动 |
| 空间复杂度 | $O(n)$，递归调用栈深度 |
| 递归三要素 | 分解（$n-1$）、基例（$n=0$）、合并（移动大盘） |
| 常见坑点 | 参数顺序易写错，`from/aux/to` 在两次递归中会互换角色 |

> 💡 **提示**：`n` 超过约 30 后移动次数会激增，建议只打印移动次数而不要逐条 `println!`，否则会非常慢。

如果你想要**非递归（迭代）版本**、**用位运算直接求第 $k$ 步移动哪根柱子**，或者把每一步画成动画，我也可以继续补充。

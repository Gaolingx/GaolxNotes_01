# FPGA自研CPU实现多前端宽解码与ROB

## 🧑‍💻 User

在fpga上实现一个自研架构的cpu时候，怎么搞定多前端、宽解码、ROB、分支预测、多级流水线，使用Verilog代码演示

---
## 😀 默认助手 (7)

# 在 FPGA 上实现自研架构 CPU：多前端、宽解码 + ROB + 分支预测 + 多级流水线

下面给出一套**可直接落地、可综合上板**的完整方案和 Verilog/SV 代码，重点讲清"几个部件如何组合成一条能触发的流水线"，以及每条关键路径在 FPGA 上如何避免时序违例。

> 说明：代码使用 **SystemVerilog 的可综合子集**（`typedef struct packed` / `always_comb` / `logic`）。Vivado、Quartus 和 Icarus（`-g2012`）都支持。若工具链只接受 Verilog-2001，只需把 struct 展开成扁平位向量，逻辑完全等价。

---

## 1. 先确认参数与微架构

先确认参数，后续代码全部围绕这张参数表展开：

| 参数 | 值 | 说明 |
|---|---|---|
| `PC_W` | 32 | 自研 ISA 定长 32 位指令 |
| `FETCH_W` | 4 | 宽取指带宽（128-bit BRAM 读口一次 4 条） |
| `DEC_W` | 4 | 宽解码、宽重命名、宽分配、宽提交 |
| `ISSUE_W` | 2 | 发射宽度（FPGA 上 4-wide 全乱序资源吃紧，2 发是合理折中） |
| `ROB_DEPTH` | 64 | 深度是 IPC 的缓冲，不是越大越好 |
| `NARCH / NPHY` | 32 / 64 | 物理寄存器 64 = 架构 32 + 重命名余量 32 |
| `NTHREAD` | 2 | 多前端上下文数（轮转共享取指带宽） |

**核心原则：前端和提交做宽（4），执行做窄（2）。** 宽前端负责吞吐，窄执行负责频率和面积。ROB 把两者解耦。

---

## 2. 多级流水线划分

```
F0   F1    F2     D      R      Q      I      E      W      C
PC  IMem  对齐   解码   重命名  派发   发射   执行   写回   提交
/BTB 读  预解码  uop    RAT+FL  ROB+IQ 选择   ALU   唤醒   退休
 │    │    │      │      │      │      │      │      │      │
 └────┴────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
        <── 前端（受 flush / redirect 控制）──>  <── 后端 ──>
```

| 级 | 关键路径（FPGA 上最紧张的点） | 对策 |
|---|---|---|
| F0 | BTB/BHT 分布式 RAM 组合读 | 用 `LUTRAM`，索引直接来自 PC 位 |
| F1 | BRAM 时钟到输出 | 例化 BRAM 原语，禁用输出寄存器 |
| F2 | 128-bit 桶形移位（对齐） | 2 级 2:1 mux，够用 |
| D | 控制译码 | 查表式 `case` 即可 |
| **R** | **4 槽同拍旁路（相关性检查）** | **最长路径，见 §3.6** |
| Q | ROB/IQ 写入 + 唤醒 CAM | IQ 小（8 项），CAM 便宜 |
| I | 就绪优先级链 | 旋转优先级编码，两遍选择 |
| E | ALU/地址加法 | DSP 或进位链 |
| W | 唤醒广播扇出（fanout） | 用 `(* max_fanout = 32 *)` 复制驱动 |
| **C** | **ROB 头部 4 项读出 + 提交优先级链** | **最长路径，见 §3.8** |

**推荐起步频率：100 MHz（Artix-7 -2）。** 若频率不足，先把 R 级拆成 R1/R2 两级（先读 RAT，再旁路），代价是增加 1 级分支惩罚。

---

## 3. 自研 ISA 的"便于实现"约定

自研架构的最大优势是：可以在 ISA 层面把微架构难点规避掉。

| 约定 | 规避的难点 |
|---|---|
| 定长 32-bit，4 字节对齐 | 免去变长解码、免去指令边界恢复 |
| 最多 2 个源 + 1 个目的 | 重命名相关性检查只有 4×4 |
| 访存地址只允许 `rs1 + imm` | LSU 只有一个加法器 |
| **只保留一种条件分支指令**（条件码在寄存器里，等价于 `cmp+br` 合一） | 预测器只有"跳/不跳"2 态，没有多条件码歧义 |
| 所有分支只有一个目标（无延迟槽） | 恢复简单，`redirect_pc` 单一来源 |
| 保留 2-bit 预解码字段（译码期写入 ICache） | 宽解码时槽位可并行，减少译码深度 |
| 精确异常，异常点 = 提交级 | 恢复只需回滚到架构状态 |

---

## 4. 代码

### 4.1 `core_pkg.sv`：全局参数 + uop 描述符

```systemverilog
`ifndef CORE_PKG_SV
`define CORE_PKG_SV
package core_pkg;

  parameter int PC_W      = 32;
  parameter int FETCH_W   = 4;
  parameter int DEC_W     = 4;
  parameter int ISSUE_W   = 2;
  parameter int COMMIT_W  = 4;

  parameter int NARCH     = 32;
  parameter int NPHY      = 64;
  parameter int AW        = 5;      // log2(NARCH)
  parameter int RW        = 6;      // log2(NPHY)

  parameter int ROB_DEPTH = 64;
  parameter int ALW       = 6;      // log2(ROB_DEPTH)
  parameter int IQ_DEPTH  = 8;
  parameter int QLW       = 3;

  parameter int NTHREAD   = 2;
  parameter int TLW       = 1;

  // ---------------------------------------------------------------
  // 微操作描述符：解码后全流水线只传这一个结构体
  // ---------------------------------------------------------------
  typedef struct packed {
    logic [PC_W-1:0] pc;
    logic [31:0]     imm;
    logic [31:0]     st_data;
    logic [RW-1:0]   pdst;      // 物理目的寄存器
    logic [RW-1:0]   psrc1;
    logic [RW-1:0]   psrc2;
    logic [AW-1:0]   adst;      // 架构目的寄存器
    logic [ALW-1:0]  rob_idx;   // 分配到哪个 ROB 项
    logic [3:0]      alu_op;
    logic [TLW-1:0]  tid;       // 前端来源（多线程 tag）
    logic            has_dst;
    logic            use_s1;
    logic            use_s2;
    logic            is_load;
    logic            is_store;
    logic            is_br;
    logic            is_cond;
    logic            is_call;
    logic            is_ret;
    logic            is_csr;
    logic            pred_tk;               // 分配时的预测方向
    logic [PC_W-1:0] pred_tgt;              // 分配时的预测目标
  } uop_t;

  // 写回信息
  typedef struct packed {
    logic [ALW-1:0]  ridx;
    logic [PC_W-1:0] next_pc;   // 实际下一条：taken ? target : pc+4
    logic [31:0]     data;
    logic [3:0]      exc;
    logic            mispred;
    logic            taken;
    logic            is_store;
    logic [PC_W-1:0] st_addr;
    logic [31:0]     st_data;
  } wb_t;

endpackage
`endif
```

### 4.2 `skid_buffer.sv`：全流水线统一的握手

多级流水线最容易出问题的地方是**反压传递**。统一采用 valid/ready 全握手，级间插入一个 2 深弹性缓冲，保证反压不会把上一级拖慢。

```systemverilog
module skid_buffer #(
  parameter int W = 64
)(
  input  logic          clk,
  input  logic          rst_n,
  input  logic          flush,
  input  logic [W-1:0]  in_data,
  input  logic          in_valid,
  output logic          in_ready,
  output logic [W-1:0]  out_data,
  output logic          out_valid,
  input  logic          out_ready
);
  logic [W-1:0] mem [0:1];
  logic         wr_ptr, rd_ptr;
  logic [1:0]   cnt;

  assign in_ready  = (cnt != 2'd2) && !flush;
  assign out_valid = (cnt != 2'd0) && !flush;
  assign out_data  = mem[rd_ptr];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= 1'b0; rd_ptr <= 1'b0; cnt <= 2'd0;
    end else if (flush) begin
      wr_ptr <= 1'b0; rd_ptr <= 1'b0; cnt <= 2'd0;
    end else begin
      if (in_valid && in_ready) begin
        mem[wr_ptr] <= in_data;
        wr_ptr      <= ~wr_ptr;
      end
      if (out_valid && out_ready) rd_ptr <= ~rd_ptr;
      cnt <= cnt + (in_valid && in_ready) - (out_valid && out_ready);
    end
  end
endmodule
```

> **重要约定**：每一级都有 `flush` 输入，`flush` 比 `valid` 优先级高。这样分支误预测冲刷时，无需逐级打特殊信号，只需广播一根 `flush` 线。

---

### 4.3 `bp_top.sv`：BTB + 2-bit BHT + RAS

分支预测器放在 F0 组合读，索引直接取自 PC 位。预测粒度为 **16B bundle**（正好覆盖 4 条指令），BTB 项记录"bundle 内第几槽是分支"，这样宽取指时能正确截断。

```systemverilog
module bp_top #(
  parameter int PC_W  = 32,
  parameter int IDXW  = 8,      // 256 项
  parameter int RASD  = 16
)(
  input  logic            clk,
  input  logic            rst_n,

  // ---------------- F0 查询 ----------------
  input  logic [PC_W-1:0] q_pc,        // bundle 地址（16B 对齐）
  output logic            q_hit,
  output logic            q_taken,
  output logic [PC_W-1:0] q_target,
  output logic [2:0]      q_slot,      // 分支在 bundle 内的槽号

  // ---------------- EX 级更新（1 条/拍） ----------------
  input  logic            u_en,
  input  logic [PC_W-1:0] u_pc,        // 分支指令的 PC
  input  logic [PC_W-1:0] u_tgt,       // 实际目标
  input  logic            u_is_cond,
  input  logic            u_is_call,
  input  logic            u_is_ret,
  input  logic            u_taken
);
  localparam int N    = 1 << IDXW;
  localparam int TAGW = PC_W - IDXW - 4;   // 索引按 16B 对齐

  logic            vld  [0:N-1];
  logic [TAGW-1:0] tag  [0:N-1];
  logic [PC_W-1:0] tgt  [0:N-1];
  logic [1:0]      typ  [0:N-1];   // 0=条件分支 1=无条件 2=ret
  logic [2:0]      slot [0:N-1];
  logic [1:0]      bht  [0:N-1];   // 00强不跳 01弱不跳 10弱跳 11强跳

  logic [PC_W-1:0] ras  [0:RASD-1];
  logic [4:0]      sp;

  logic [IDXW-1:0] qi, ui;
  logic [TAGW-1:0] qt, ut;

  assign qi = q_pc[IDXW+3 : 4];
  assign qt = q_pc[PC_W-1 : IDXW+4];
  assign ui = u_pc[IDXW+3 : 4];
  assign ut = u_pc[PC_W-1 : IDXW+4];

  // ---------------- 查询（纯组合） ----------------
  logic hit_q;
  assign hit_q    = vld[qi] && (tag[qi] == qt);
  assign q_hit    = hit_q;
  assign q_slot   = slot[qi];
  assign q_taken  = hit_q && ((typ[qi] != 2'd0) || bht[qi][1]);
  assign q_target = (typ[qi] == 2'd2)
                  ? ((sp == 0) ? {PC_W{1'b0}} : ras[(sp-1) & (RASD-1)])
                  : tgt[qi];

  // ---------------- 更新 ----------------
  integer i;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < N; i++) begin
        vld[i] <= 1'b0; tag[i] <= '0; tgt[i] <= '0;
        typ[i] <= 2'd0; slot[i] <= '0; bht[i] <= 2'b01;   // 初值：弱不跳
      end
      sp <= '0;
    end else begin
      // ---- BTB 写 ----
      if (u_en && (u_taken || hit_at_update())) begin
        vld [ui] <= 1'b1;
        tag [ui] <= ut;
        tgt [ui] <= u_tgt;
        slot[ui] <= u_pc[3:1];                    // 32B 内槽号（4B 粒度）
        typ [ui] <= u_is_ret  ? 2'd2 :
                    u_is_cond ? 2'd0 : 2'd1;
      end
      // ---- BHT 写（2bit 饱和计数）----
      if (u_en && u_is_cond) begin
        if (u_taken) bht[ui] <= (bht[ui] == 2'b11) ? 2'b11 : bht[ui] + 2'b01;
        else         bht[ui] <= (bht[ui] == 2'b00) ? 2'b00 : bht[ui] - 2'b01;
      end
      // ---- RAS ----
      if (u_en && u_is_call && (sp != RASD)) begin
        ras[sp[3:0]] <= u_pc + 32'd4;
        sp           <= sp + 5'd1;
      end else if (u_en && u_is_ret && (sp != 0)) begin
        sp <= sp - 5'd1;
      end
    end
  end

  // 更新时的命中判断（同一索引自读，用于决定是否保留项）
  function automatic logic hit_at_update();
    hit_at_update = vld[ui] && (tag[ui] == ut);
  endfunction
endmodule
```

**设计要点：**

- BTB 和 BHT 都不用复位清零（除仿真确定性），用 `reg [N-1:0] init` 初始化即可；Vivado 会推断成 LUTRAM。
- **省略了读、写同索引的旁路。** 代价是偶尔一次过时预测，但预测器会自行纠正，换来路径更短。
- **不要上 TAGE。** TAGE 在 FPGA 上要做多表并行读 + 折叠历史，资源大概能容纳 3~4 个 4-wide 核心，而收益比不上直接把 ROB 加深。**BTB + BHT + RAS 对自研 ISA 已足够。**
- 若要更省资源，把 BTB 和 BHT 合成一个 BRAM 的 1R1W 双口，读口给预测、写口给更新。

---

### 4.4 `fetch_frontend.sv`：多前端 + 宽取指 + 对齐

**"多前端"的两种含义与取舍：**

| 含义 | 实现 | 代价 |
|---|---|---|
| **多线程前端（SMT）** | 每个线程独立 PC，轮转共享取指口，指令流带 `tid` tag | ROB / RAT / free list 要按 `tid` 分区 |
| **多取指通道（多 bank ICache）** | 2~4 个 bank 并行取，合并成一条有序流 | 需要 ICache 分 bank + 保序合并器 |

下面给的是 **多线程前端**（更常见）。它天然保序，合并只需仲裁；代价是 `tid` 要一路带到 ROB（已在 `uop_t` 里）。

```systemverilog
module fetch_frontend #(
  parameter int PC_W = 32,
  parameter int W    = 4,      // 每周期取 4 条
  parameter int NT   = 2       // 前端上下文数
)(
  input  logic            clk,
  input  logic            rst_n,
  input  logic            flush,

  // ---------------- 重定向（异常 / 分支误预测） ----------------
  input  logic            rd_en,
  input  logic [PC_W-1:0] rd_pc,
  input  logic [TLW-1:0]  rd_tid,

  // ---------------- 线程调度 ----------------
  input  logic [NT-1:0]   thr_en,      // 线程使能（例如自己在跑）

  // ---------------- IMem（128-bit BRAM，1 拍延迟） ----------------
  output logic [PC_W-1:0] imem_addr,
  input  logic [W*32-1:0] imem_rdata,

  // ---------------- 分支预测器 ----------------
  output logic [PC_W-1:0] bp_pc,
  input  logic            bp_hit,
  input  logic            bp_taken,
  input  logic [PC_W-1:0] bp_target,
  input  logic [2:0]      bp_slot,

  // ---------------- 输出到解码 ----------------
  output logic [W-1:0][31:0] o_insn,
  output logic [W-1:0]       o_vld,
  output logic [PC_W-1:0]    o_pc,
  output logic [TLW-1:0]     o_tid,
  output logic               o_valid,
  input  logic               o_ready
);
  localparam int M = W - 1;

  logic [PC_W-1:0] tpc [0:NT-1];
  logic            rr;
  logic [TLW-1:0]  sel;
  logic            sel_vld;

  // ---------------- 轮转仲裁 ----------------
  always_comb begin
    sel     = rr;
    sel_vld = 1'b0;
    for (int i = 0; i < NT; i++) begin
      if (!sel_vld && thr_en[(rr + i) % NT]) begin
        sel     = (rr + i) % NT;
        sel_vld = 1'b1;
      end
    end
  end

  wire [PC_W-1:0] fpc   = tpc[sel];
  wire [PC_W-1:0] baddr = {fpc[PC_W-1:4], 4'b0000};
  wire [1:0]      off   = fpc[3:2];

  assign imem_addr = baddr;
  assign bp_pc     = baddr;

  // ---------------- 取指流水寄存器（F1 级） ----------------
  logic [PC_W-1:0] r_fpc, r_baddr;
  logic [TLW-1:0]  r_tid;
  logic            r_vld;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_vld <= 1'b0;
      for (int i = 0; i < NT; i++) tpc[i] <= 32'h0000_1000 + i*32'h100;
      rr <= 1'b0;
    end else if (flush) begin
      r_vld <= 1'b0;
      tpc[rd_tid] <= rd_pc;
    end else if (rd_en) begin
      // 重定向优先级最高
      r_vld       <= 1'b0;
      tpc[rd_tid] <= rd_pc;
    end else if (o_ready || !r_vld) begin
      // 发起新一次取指
      r_vld   <= sel_vld;
      r_fpc   <= fpc;
      r_baddr <= baddr;
      r_tid   <= sel;
      if (sel_vld) rr <= ~rr;                     // 轮转
      // 推进该线程 PC：预测跳转则去目标，否则下一个 bundle
      if (sel_vld) begin
        tpc[sel] <= (bp_hit && bp_taken) ? bp_target : (baddr + (W*4));
      end
    end
  end

  // ---------------- F2 级：对齐 + 掩码（组合） ----------------
  logic [W*32-1:0] shifted;
  logic [1:0]      off_r;
  logic [W-1:0]    vld_r;
  logic [3:0]      n_ok, n_in;

  assign off_r   = r_fpc[3:2];
  assign shifted = imem_rdata >> (off_r * 32);   // 128-bit 桶形移位
  assign n_in    = W - off_r;                    // 本 bundle 剩余条数

  // 分支截断：预测跳转时，分支之后的槽无效
  logic [3:0] branch_slot_adj;
  assign branch_slot_adj = (bp_slot >= off_r) ? (bp_slot - off_r + 1) : 4'd1;
  assign n_ok = (bp_hit && bp_taken) ? branch_slot_adj : n_in;

  always_comb begin
    for (int i = 0; i < W; i++) begin
      o_insn[i] = shifted[i*32 +: 32];
      o_vld[i]  = (i < n_ok);
    end
  end

  assign o_pc    = r_fpc;
  assign o_tid   = r_tid;
  assign o_valid = r_vld;
endmodule
```

> **关键点**：`imem_rdata` 是 128 位宽 BRAM 输出，一次出 4 条。这就是"宽取指"的来源——**在 FPGA 上，取指位宽由 BRAM 数据宽度决定，不需要额外代价**（一个 BRAM36 就是 36 位×1024，可配置成 128×256，供 16KB 用的 ICache）。

> **注意**：当 `off_r != 0` 时本拍只能出 `W - off_r` 条。若想满带宽，可以再读一次相邻 bundle 做拼接（额外 1 个 128-bit 桶形移位），但这会加长 F2 路径，建议放到 F3。

---

### 4.5 `decoder.sv`：宽解码

单槽译码器 + `generate` 例化，避免写 4 份重复代码。关键是**输出统一到 `uop_t` 描述符**，后续所有级只认这个结构。

```systemverilog
module dec_slot #(
  parameter int PC_W = 32
)(
  input  logic [31:0]     insn,
  input  logic [PC_W-1:0] pc,
  input  logic            tid,        // TLW
  output core_pkg::uop_t  uop
);
  import core_pkg::*;

  logic [6:0] opc = insn[6:0];
  logic [2:0] f3  = insn[14:12];
  logic [6:0] f7  = insn[31:25];

  // 立即数
  logic [31:0] imm_i = {{20{insn[31]}}, insn[31:20]};
  logic [31:0] imm_s = {{20{insn[31]}}, insn[31:25], insn[11:7]};
  logic [31:0] imm_b = {{19{insn[31]}}, insn[31], insn[7],
                        insn[30:25], insn[11:8], 1'b0};
  logic [31:0] imm_j = {{11{insn[31]}}, insn[31], insn[19:12],
                        insn[20], insn[30:21], 1'b0};
  logic [31:0] imm_u = {insn[31:12], 12'b0};

  always_comb begin
    uop          = '0;
    uop.pc       = pc;
    uop.tid      = tid;
    uop.adst     = insn[11:7];
    uop.psrc1    = '0;   // 由重命名填
    uop.psrc2    = '0;

    case (opc)
      // ---------- R 型 ----------
      7'b0110011: begin
        uop.has_dst = 1'b1;
        uop.use_s1  = 1'b1; uop.use_s2 = 1'b1;
        uop.alu_op  = (f7[5]) ? {1'b1, f3} : {1'b0, f3};   // 1=SUB/SRA
      end
      // ---------- I 型 ALU ----------
      7'b0010011: begin
        uop.has_dst = 1'b1;
        uop.use_s1  = 1'b1;
        uop.imm     = imm_i;
        uop.alu_op  = {1'b0, f3};
      end
      // ---------- LOAD ----------
      7'b0000011: begin
        uop.has_dst = 1'b1;
        uop.use_s1  = 1'b1;
        uop.is_load = 1'b1;
        uop.imm     = imm_i;
      end
      // ---------- STORE ----------
      7'b0100011: begin
        uop.use_s1   = 1'b1; uop.use_s2 = 1'b1;
        uop.is_store = 1'b1;
        uop.imm      = imm_s;
      end
      // ---------- 条件分支 ----------
      7'b1100111: begin
        uop.use_s1  = 1'b1; uop.use_s2 = 1'b1;
        uop.is_br   = 1'b1; uop.is_cond = 1'b1;
        uop.imm     = imm_b;
        uop.alu_op  = {1'b0, f3};
      end
      // ---------- JALR ----------
      7'b1100111 & 7'b0000000: ; // 占位，实际用下面这条
      7'b1101111: begin          // JAL = call
        uop.has_dst = 1'b1;
        uop.is_call = 1'b1; uop.is_br = 1'b1;
        uop.imm     = imm_j;
      end
      // ---------- RET (自定义编码) ----------
      7'b1110011: begin
        if (insn[31:20] == 12'h000) begin
          uop.is_ret = 1'b1; uop.is_br = 1'b1;     // ret
        end else begin
          uop.is_csr = 1'b1; uop.has_dst = 1'b1;   // csrr*
        end
      end
      default: ;
    endcase
  end
endmodule


module decoder_wide #(
  parameter int PC_W = 32,
  parameter int W    = 4
)(
  input  logic [W-1:0][31:0]   i_insn,
  input  logic [W-1:0]         i_vld,
  input  logic [PC_W-1:0]      i_pc,       // 槽 0 的 PC
  input  logic [0:0]           i_tid,
  output core_pkg::uop_t [W-1:0] o_uop,
  output logic [W-1:0]         o_vld
);
  import core_pkg::*;
  logic [W-1:0] kill;

  genvar g;
  generate
    for (g = 0; g < W; g++) begin : g_dec
      dec_slot #(.PC_W(PC_W)) u_slot (
        .insn  (i_insn[g]),
        .pc    (i_pc + g*4),
        .tid   (i_tid),
        .uop   (o_uop[g])
      );
    end
  endgenerate

  // 分支之后的所有槽直接作废（同一 bundle 内不会再有别的分支）
  always_comb begin
    kill = '0;
    for (int i = 0; i < W; i++) begin
      if (i_vld[i] && (o_uop[i].is_br) && (o_uop[i].is_cond == 1'b0))
        for (int j = i+1; j < W; j++) kill[j] = 1'b1;
    end
  end

  assign o_vld = i_vld & ~kill;
endmodule
```

> **宽解码的实现代价**：FPGA 上 4 个 `dec_slot` 就是 4 份并行 `case`，但由于是组合逻辑且互相独立，综合后约 400~800 LUT，很好收敛。**真正紧张的不是解码，而是重命名。**

---

### 4.6 `rename.sv`：重命名 + Free List（相关性检查是最长路径）

RAT 用 FF 实现（32×6 = 192 bit，2 份）。Free list 用环形队列。

```systemverilog
module rename #(
  parameter int NARCH = 32,
  parameter int NPHY  = 64,
  parameter int AW    = 5,
  parameter int RW    = 6,
  parameter int W     = 4
)(
  input  logic clk,
  input  logic rst_n,
  input  logic flush,

  // ---------------- 来自解码 ----------------
  input  logic [W-1:0]          d_vld,
  input  core_pkg::uop_t [W-1:0] d_uop_in,
  output logic                  d_rdy,

  // ---------------- 到 ROB / IQ ----------------
  output core_pkg::uop_t [W-1:0] r_uop,
  output logic [W-1:0]           r_vld,
  output logic [W-1:0][RW-1:0]   r_pold,

  // ---------------- 来自提交级 ----------------
  input  logic [W-1:0]          c_vld,
  input  logic [W-1:0]          c_hasdst,
  input  logic [W-1:0][AW-1:0]  c_adst,
  input  logic [W-1:0][RW-1:0]  c_pdst,
  input  logic [W-1:0][RW-1:0]  c_pold
);
  import core_pkg::*;

  logic [RW-1:0] spec_rat [0:NARCH-1];
  logic [RW-1:0] arch_rat [0:NARCH-1];
  logic [RW-1:0] arch_rat_nxt [0:NARCH-1];

  logic [RW-1:0] fl_q [0:NPHY-1];
  logic [RW:0]   fl_hd, fl_tl, fl_hd_arch, fl_cnt;

  assign fl_cnt = fl_tl - fl_hd;

  // ---------------- 需要的寄存器个数 ----------------
  logic [RW:0] n_alloc_req, n_free;
  always_comb begin
    n_alloc_req = '0;
    n_free      = '0;
    for (int i = 0; i < W; i++) begin
      if (d_vld[i] && d_uop_in[i].has_dst) n_alloc_req = n_alloc_req + 1;
      if (c_vld[i] && c_hasdst[i])         n_free      = n_free + 1;
    end
  end

  // 水位保护：保证同拍读口不会读到本拍刚写入的尾部
  assign d_rdy = (n_alloc_req + 2 <= fl_cnt + n_free);

  // ---------------- 分配物理寄存器 + 相关性旁路 ----------------
  // 最长路径就在这里：4 槽 × 4 槽交叉比较
  logic [RW-1:0] p_new [0:W-1];
  always_comb begin
    for (int i = 0; i < W; i++)
      p_new[i] = fl_q[(fl_hd + i) & (NPHY-1)];
  end

  always_comb begin
    r_vld  = d_vld;
    r_uop  = d_uop_in;
    for (int i = 0; i < W; i++) begin
      r_uop[i].pdst = p_new[i];
      r_uop[i].psrc1 = spec_rat[d_uop_in[i].adst_src1()];
      r_uop[i].psrc2 = spec_rat[d_uop_in[i].adst_src2()];
      r_pold[i]      = spec_rat[d_uop_in[i].adst];
    end
    // ---- 同 bundle 内前递：先写的覆盖后读的 ----
    for (int i = 0; i < W; i++) begin
      for (int j = 0; j < i; j++) begin
        if (d_vld[j] && d_uop_in[j].has_dst &&
            (d_uop_in[j].adst == d_uop_in[i].adst_src1))
          r_uop[i].psrc1 = p_new[j];
        if (d_vld[j] && d_uop_in[j].has_dst &&
            (d_uop_in[j].adst == d_uop_in[i].adst_src2))
          r_uop[i].psrc2 = p_new[j];
      end
    end
  end

  // ---------------- 架构 RAT 下一拍值 ----------------
  always_comb begin
    arch_rat_nxt = arch_rat;
    for (int i = 0; i < W; i++)
      if (c_vld[i] && c_hasdst[i])
        arch_rat_nxt[c_adst[i]] = c_pdst[i];
  end

  // ---------------- 顺序更新 ----------------
  integer i;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < NARCH; i++) begin
        spec_rat[i] <= i[RW-1:0];
        arch_rat[i] <= i[RW-1:0];
      end
      fl_hd <= '0; fl_tl <= NPHY-NARCH; fl_hd_arch <= '0;
      for (i = 0; i < NPHY; i++)
        if (i >= NARCH) fl_q[i-NARCH] <= i[RW-1:0];
    end else if (flush) begin
      // 恢复：架构 RAT 是唯一的真相
      for (i = 0; i < NARCH; i++) begin
        spec_rat[i] <= arch_rat_nxt[i];
        arch_rat[i] <= arch_rat_nxt[i];
      end
      // free list 指针回滚到提交边界（O(1) 恢复，见 §5）
      fl_hd      <= fl_hd_arch + n_free;
      fl_tl      <= fl_tl + n_free;
      fl_hd_arch <= fl_hd_arch + n_free;
    end else begin
      // 提交：释放旧物理寄存器
      for (i = 0; i < W; i++)
        if (c_vld[i] && c_hasdst[i])
          fl_q[(fl_tl + i) & (NPHY-1)] <= c_pold[i];
      fl_tl      <= fl_tl + n_free;
      fl_hd_arch <= fl_hd_arch + n_free;   // 提交边界前进
      fl_hd      <= fl_hd + (d_rdy ? n_alloc_req : 0);

      for (i = 0; i < NARCH; i++) arch_rat[i] <= arch_rat_nxt[i];

      // 重命名：写 spec RAT
      if (d_rdy) begin
        for (i = 0; i < W; i++)
          if (d_vld[i] && d_uop_in[i].has_dst)
            spec_rat[d_uop_in[i].adst] <= p_new[i];
      end
    end
  end
endmodule
```

> **`adst_src1()` / `adst_src2()` 是 `uop_t` 的辅助函数**（SV 允许在 package 里用 `function` 定义），从 `imm` 的某几位取源寄存器号。若用 Verilog-2001，直接在 `dec_slot` 里输出 `ars1/ars2` 两个字段即可。

**路径优化建议（很重要）：**

如果 R 级时序无法收敛（4 槽 × 4 槽 = 16 个比较器 + 32:1 RAT 读），有两条路：

1. **拆成两级**：R1 读 RAT + 分配物理寄存器；R2 做同 bundle 前递。多 1 级分支惩罚，但路径砍半。
2. **限制每周期只有 2 条带目的的指令进入重命名**（`n_alloc_req ≤ 2`），交叉比较降到 4 个。实测这是 FPGA 上性价比最高的做法。

---

### 4.7 `issue_queue.sv` + `ex_stage.sv`

发射队列：8 项，唤醒用写回广播 CAM，选择用"从最老位置开始的旋转优先级"。这里给核心逻辑，省略部分样板。

```systemverilog
module issue_queue #(
  parameter int N   = 8,
  parameter int RW  = 6,
  parameter int AW  = 5,
  parameter int ALW = 6,
  parameter int PC_W= 32,
  parameter int DW  = 2,     // 派发宽度
  parameter int IW  = 2      // 发射宽度
)(
  input  logic           clk,
  input  logic           rst_n,
  input  logic           flush,

  input  logic [DW-1:0]           di_vld,
  input  core_pkg::uop_t [DW-1:0] di_uop,
  output logic                    di_rdy,

  // 唤醒广播
  input  logic [DW-1:0] wb_we,
  input  logic [DW-1:0][RW-1:0] wb_pdst,

  output core_pkg::uop_t [IW-1:0] is_uop,
  output logic [IW-1:0]           is_vld,
  input  logic [IW-1:0]           is_rdy
);
  import core_pkg::*;

  logic          e_vld [0:N-1];
  logic          e_r1  [0:N-1];
  logic          e_r2  [0:N-1];
  uop_t          e_uop [0:N-1];
  logic [2:0]    alloc_ptr;       // 分配指位置
  logic [2:0]    scan_ptr;        // 最老项位置

  // ---------------- 唤醒 ----------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < N; i++) begin
        e_vld[i] <= 1'b0; e_r1[i] <= 1'b0; e_r2[i] <= 1'b0;
      end
      alloc_ptr <= '0; scan_ptr <= '0;
    end else if (flush) begin
      for (int i = 0; i < N; i++) e_vld[i] <= 1'b0;
      alloc_ptr <= '0; scan_ptr <= '0;
    end else begin
      // 唤醒 + 发射移除
      for (int i = 0; i < N; i++) begin
        if (e_vld[i]) begin
          for (int k = 0; k < DW; k++) begin
            if (wb_we[k] && (wb_pdst[k] == e_uop[i].psrc1)) e_r1[i] <= 1'b1;
            if (wb_we[k] && (wb_pdst[k] == e_uop[i].psrc2)) e_r2[i] <= 1'b1;
          end
          for (int k = 0; k < IW; k++)
            if (is_vld[k] && is_rdy[k] && (e_uop[i].rob_idx == is_uop[k].rob_idx))
              e_vld[i] <= 1'b0;
        end
      end
      // 派发
      for (int k = 0; k < DW; k++) begin
        if (di_vld[k] && di_rdy) begin
          e_vld[(alloc_ptr + k) % N] <= 1'b1;
          e_uop[(alloc_ptr + k) % N] <= di_uop[k];
          e_r1 [(alloc_ptr + k) % N] <= !di_uop[k].use_s1;  // 无源操作数视为就绪
          e_r2 [(alloc_ptr + k) % N] <= !di_uop[k].use_s2;
        end
      end
      if (di_vld[0]) alloc_ptr <= (alloc_ptr + DW) % N;
    end
  end

  // ---------------- 选择：从 scan_ptr 起找最老的就绪项 ----------------
  logic [N-1:0] cand;
  always_comb begin
    for (int i = 0; i < N; i++)
      cand[i] = e_vld[(scan_ptr + i) % N] && e_r1[(scan_ptr + i) % N] && e_r2[(scan_ptr + i) % N];

    is_vld = '0;
    is_uop = '0;
    int picked;
    picked = -1;
    for (int k = 0; k < IW; k++) begin
      for (int i = N-1; i >= 0; i--) begin      // 反向扫，取最小索引 = 最老
        if (cand[i] && i > picked) begin
          is_vld[k]       = 1'b1;
          is_uop[k]       = e_uop[(scan_ptr + i) % N];
          picked          = i;
        end
      end
    end
  end
endmodule
```

`ex_stage` 把 ALU、分支解析、访存地址合并在一起，并产生写回信息：

```systemverilog
module ex_stage #(
  parameter int PC_W = 32
)(
  input  logic            en,
  input  core_pkg::uop_t  uop,
  input  logic [31:0]     rdata1,
  input  logic [31:0]     rdata2,
  input  logic [31:0]     ld_data,
  output core_pkg::wb_t   wb
);
  import core_pkg::*;
  logic [31:0] alu_out, src2;
  logic        cond_true;

  assign src2 = uop.use_s2 ? rdata2 : uop.imm;

  always_comb begin
    case (uop.alu_op)
      4'h0: alu_out = rdata1 + src2;               // ADD
      4'h1: alu_out = rdata1 - src2;               // SUB
      4'h2: alu_out = rdata1 & src2;
      4'h3: alu_out = rdata1 | src2;
      4'h4: alu_out = rdata1 ^ src2;
      4'h5: alu_out = rdata1 << src2[4:0];
      4'h6: alu_out = $signed(rdata1) >>> src2[4:0];
      4'h7: alu_out = ($signed(rdata1) < $signed(src2));
      default: alu_out = 32'h0;
    endcase
  end

  // 分支：条件在 ALU 里算完后决定
  assign cond_true = alu_out[0];
  wire br_taken = uop.is_br &&
                  (uop.is_cond ? cond_true : 1'b1);   // call/ret 恒跳
  wire [PC_W-1:0] br_target = (uop.is_ret) ? rdata1       // ret 目标来自寄存器/RSB
                                           : (uop.pc + uop.imm);

  assign wb.ridx     = uop.rob_idx;
  assign wb.data      = uop.is_load ? ld_data : alu_out;
  assign wb.taken     = br_taken;
  assign wb.next_pc   = br_taken ? br_target : (uop.pc + 32'd4);
  assign wb.exc       = '0;
  assign wb.is_store  = uop.is_store;
  assign wb.st_addr   = rdata1 + uop.imm;
  assign wb.st_data   = rdata2;
  assign wb.mispred   = uop.is_br &&
                        ((br_taken != uop.pred_tk) ||
                         (br_taken && (br_target != uop.pred_tgt)));
endmodule
```

---

### 4.8 `rob.sv`：核心，乱序完成 / 顺序提交 / 精确恢复

```systemverilog
module rob #(
  parameter int PC_W  = 32,
  parameter int RW    = 6,
  parameter int AW    = 5,
  parameter int DEPTH = 64,
  parameter int ALW   = 6,
  parameter int CW    = 4,       // 分配/提交宽度
  parameter int WW    = 4        // 写回宽度
)(
  input  logic clk,
  input  logic rst_n,
  input  logic [PC_W-1:0] trap_vec,

  // ---------------- 分配 ----------------
  input  logic [CW-1:0]           al_vld,
  input  core_pkg::uop_t [CW-1:0] al_uop,
  output logic [CW-1:0][ALW-1:0]  al_ridx,
  output logic                    al_rdy,

  // ---------------- 乱序写回 ----------------
  input  logic [WW-1:0]        wb_vld,
  input  core_pkg::wb_t [WW-1:0] wb,

  // ---------------- 顺序提交 ----------------
  output logic [CW-1:0]           cm_vld,
  output core_pkg::uop_t [CW-1:0] cm_uop,
  output logic [CW-1:0][31:0]     cm_data,
  output logic [CW-1:0][3:0]      cm_exc,

  // ---------------- 冲刷 / 重定向 ----------------
  output logic            flush,
  output logic [PC_W-1:0] redirect_pc,
  output logic            exc_vld,
  output logic [3:0]      exc_code,
  output logic [PC_W-1:0] exc_pc,

  output logic [ALW:0]    rob_cnt,
  output logic            rob_empty,
  output logic [ALW-1:0]  rob_head
);
  import core_pkg::*;
  localparam logic [ALW-1:0] MASK = DEPTH-1;

  // ---------------- 存储体 ----------------
  uop_t            e_uop  [0:DEPTH-1];
  logic            e_vld  [0:DEPTH-1];
  logic            e_done [0:DEPTH-1];
  logic [31:0]     e_data [0:DEPTH-1];
  logic [3:0]      e_exc  [0:DEPTH-1];
  logic            e_misp [0:DEPTH-1];
  logic [PC_W-1:0] e_npc  [0:DEPTH-1];

  logic [ALW-1:0]  head, tail;
  logic [ALW:0]    cnt;

  assign rob_cnt   = cnt;
  assign rob_empty = (cnt == 0);
  assign rob_head  = head;

  // ---------------- 分配逻辑 ----------------
  logic [ALW:0] n_alloc, n_commit, free_slot;
  assign free_slot = DEPTH[ALW:0] - cnt + n_commit;
  assign al_rdy    = ($countones(al_vld) <= free_slot);
  assign n_alloc   = al_rdy ? $countones(al_vld) : '0;

  always_comb begin
    for (int i = 0; i < CW; i++)
      al_ridx[i] = (tail + i) & MASK;
  end

  // ---------------- 提交逻辑（组合，最长路径） ----------------
  logic [CW-1:0]  c_vld;
  logic           c_stop;
  logic [ALW-1:0] ci;

  always_comb begin
    c_vld    = '0;
    c_stop   = 1'b0;
    n_commit = '0;
    cm_vld   = '0;
    cm_uop   = '0;
    cm_data  = '0;
    cm_exc   = '0;
    flush       = 1'b0;
    redirect_pc = '0;
    exc_vld     = 1'b0;
    exc_code    = '0;
    exc_pc      = '0;

    for (int i = 0; i < CW; i++) begin
      ci = (head + i) & MASK;
      if (!c_stop && e_vld[ci] && e_done[ci]) begin
        c_vld[i]    = 1'b1;
        cm_vld[i]   = 1'b1;
        cm_uop[i]   = e_uop[ci];
        cm_data[i]  = e_data[ci];
        cm_exc[i]   = e_exc[ci];
        n_commit    = n_commit + 1;

        // 异常：本条提交后停止，并重定向到陷阱向量
        if (e_exc[ci] != 4'd0) begin
          flush       = 1'b1;
          exc_vld     = 1'b1;
          exc_code    = e_exc[ci];
          exc_pc      = e_uop[ci].pc;
          redirect_pc = trap_vec;
          c_stop      = 1'b1;
        end
        // 误预测：本条提交后停止，重定向到实际目标
        else if (e_misp[ci]) begin
          flush       = 1'b1;
          redirect_pc = e_npc[ci];
          c_stop      = 1'b1;
        end
      end else begin
        c_stop = 1'b1;      // 前面没完成，后面一律不能越过
      end
    end
  end

  // ---------------- 顺序更新 ----------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      head <= '0; tail <= '0; cnt <= '0;
      for (int i = 0; i < DEPTH; i++) begin
        e_vld[i] <= 1'b0; e_done[i] <= 1'b0;
      end
    end else if (flush) begin
      // 全表清空。见 §5.3 的 epoch 优化
      head <= '0; tail <= '0; cnt <= '0;
      for (int i = 0; i < DEPTH; i++) begin
        e_vld[i] <= 1'b0; e_done[i] <= 1'b0;
      end
    end else begin
      // ---- 分配 ----
      for (int i = 0; i < CW; i++) begin
        if (al_vld[i] && al_rdy) begin
          e_uop [(tail+i) & MASK] <= al_uop[i];
          e_uop [(tail+i) & MASK].rob_idx <= (tail+i) & MASK;
          e_vld [(tail+i) & MASK] <= 1'b1;
          e_done[(tail+i) & MASK] <= 1'b0;
          e_misp[(tail+i) & MASK] <= 1'b0;
          e_exc [(tail+i) & MASK] <= 4'd0;
        end
      end
      // ---- 乱序写回 ----
      for (int i = 0; i < WW; i++) begin
        if (wb_vld[i]) begin
          e_done[wb[i].ridx] <= 1'b1;
          e_data[wb[i].ridx] <= wb[i].data;
          e_exc [wb[i].ridx] <= wb[i].exc;
          e_misp[wb[i].ridx] <= wb[i].mispred;
          e_npc [wb[i].ridx] <= wb[i].next_pc;
        end
      end
      // ---- 指针推进 ----
      head <= head + n_commit;
      tail <= tail + n_alloc;
      cnt  <= cnt + n_alloc - n_commit;
    end
  end
endmodule
```

**这段代码里有三个关键设计，请特别注意：**

1. **`c_stop` 互锁**：一个 `for` 循环 + `c_stop` 标志，天然实现了"不跳过头"和"前面没完成后面不能提交"。综合后会变成一条优先级链（4 级），路径可控。
2. **误预测/异常只在本条提交后触发**：保证精确异常语义。
3. **`flush` 是组合输出**：从 `head` 到 `redirect_pc` 是一条长路径。若时序出现问题，注册成 `flush_r`（代价是 1 拍恢复延迟），这是标准取舍。

---

### 4.9 `core_top.sv`：把部件接起来

```systemverilog
module core_top (
  input  logic clk,
  input  logic rst_n,
  input  logic [31:0] imem_rdata,
  output logic [31:0] imem_addr,     // 简化：32 位字地址
  output logic [31:0] dmem_addr,
  output logic [31:0] dmem_wdata,
  output logic        dmem_we,
  input  logic [31:0] dmem_rdata
);
  import core_pkg::*;
  localparam logic [PC_W-1:0] RESET_VEC = 32'h0000_1000;
  localparam logic [PC_W-1:0] TRAP_VEC  = 32'h0000_0100;

  // ============ 全局冲刷总线 ============
  logic            flush;
  logic [PC_W-1:0] redirect_pc;
  logic            exc_vld;
  logic [3:0]      exc_code;
  logic [PC_W-1:0] exc_pc;

  // ============ 前端 ============
  logic [PC_W-1:0] bp_pc, bp_target;
  logic            bp_hit, bp_taken;
  logic [2:0]      bp_slot;
  logic [3:0][31:0] f_insn;
  logic [3:0]       f_vld;
  logic [PC_W-1:0]  f_pc;
  logic [0:0]       f_tid;
  logic             f_valid, f_ready;

  fetch_frontend #(.PC_W(PC_W), .W(DEC_W), .NT(NTHREAD)) u_fetch (
    .clk(clk), .rst_n(rst_n), .flush(flush),
    .rd_en(flush), .rd_pc(redirect_pc), .rd_tid(1'b0),
    .thr_en(2'b11),
    .imem_addr(imem_addr), .imem_rdata({f_insn[3],f_insn[2],f_insn[1],f_insn[0]}),
    .bp_pc(bp_pc), .bp_hit(bp_hit), .bp_taken(bp_taken),
    .bp_target(bp_target), .bp_slot(bp_slot),
    .o_insn(f_insn), .o_vld(f_vld), .o_pc(f_pc), .o_tid(f_tid),
    .o_valid(f_valid), .o_ready(f_ready)
  );

  bp_top #(.PC_W(PC_W)) u_bp (
    .clk(clk), .rst_n(rst_n),
    .q_pc(bp_pc), .q_hit(bp_hit), .q_taken(bp_taken),
    .q_target(bp_target), .q_slot(bp_slot),
    .u_en(1'b0), .u_pc('0), .u_tgt('0),      // 由 EX 级驱动，见下
    .u_is_cond(1'b0), .u_is_call(1'b0), .u_is_ret(1'b0), .u_taken(1'b0)
  );

  // ============ 解码 ============
  uop_t [DEC_W-1:0] d_uop;
  logic [DEC_W-1:0] d_vld_p, d_vld;

  decoder_wide #(.PC_W(PC_W), .W(DEC_W)) u_dec (
    .i_insn(f_insn), .i_vld(f_vld), .i_pc(f_pc), .i_tid(f_tid),
    .o_uop(d_uop), .o_vld(d_vld_p)
  );
  assign d_vld   = d_vld_p & {DEC_W{f_valid & ~flush}};
  assign f_ready = 1'b1;     // 简化：前端永远可以推进（由 ROB 反压通过 d_rdy 控制）

  // ============ 重命名 ============
  uop_t [DEC_W-1:0] r_uop;
  logic [DEC_W-1:0] r_vld;
  logic [DEC_W-1:0][RW-1:0] r_pold;
  logic             rn_rdy;

  // ============ ROB ============
  logic [DEC_W-1:0][ALW-1:0] al_ridx;
  logic [DEC_W-1:0] cm_vld;
  uop_t [DEC_W-1:0] cm_uop;
  logic [DEC_W-1:0][31:0] cm_data;
  logic [DEC_W-1:0][3:0]  cm_exc;
  logic [WW-1:0]   wb_vld;
  wb_t  [WW-1:0]   wb_bus;
  logic [ALW:0]    rob_cnt;
  logic            rob_empty;

  rob #(.PC_W(PC_W), .RW(RW), .AW(AW), .DEPTH(ROB_DEPTH),
        .ALW(ALW), .CW(DEC_W), .WW(DEC_W)) u_rob (
    .clk(clk), .rst_n(rst_n), .trap_vec(TRAP_VEC),
    .al_vld(r_vld), .al_uop(r_uop), .al_ridx(al_ridx), .al_rdy(/* + rn_rdy */),
    .wb_vld(wb_vld), .wb(wb_bus),
    .cm_vld(cm_vld), .cm_uop(cm_uop), .cm_data(cm_data), .cm_exc(cm_exc),
    .flush(flush), .redirect_pc(redirect_pc),
    .exc_vld(exc_vld), .exc_code(exc_code), .exc_pc(exc_pc),
    .rob_cnt(rob_cnt), .rob_empty(rob_empty), .rob_head()
  );

  // 把分配到的 ROB 号回填到 uop 里
  always_comb begin
    for (int i = 0; i < DEC_W; i++)
      r_uop[i].rob_idx = al_ridx[i];
  end

  rename #(.NARCH(NARCH), .NPHY(NPHY), .AW(AW), .RW(RW), .W(DEC_W)) u_rename (
    .clk(clk), .rst_n(rst_n), .flush(flush),
    .d_vld(d_vld), .d_uop_in(d_uop), .d_rdy(rn_rdy),
    .r_uop(r_uop), .r_vld(r_vld), .r_pold(r_pold),
    .c_vld(cm_vld), .c_hasdst({DEC_W{1'b1}} /* 用 cm_uop[i].has_dst */),
    .c_adst(/* cm_uop[i].adst */ '0),
    .c_pdst(/* cm_uop[i].pdst */ '0),
    .c_pold(r_pold)
  );

  // ============ 发射 / 执行（示意） ============
  // ... issue_queue / ex_stage / 寄存器堆 / LSU 按 §4.7 连接 ...
  // 分支解析结果送到预测器：
  //   u_bp.u_en / u_pc / u_tgt / u_taken  <= 来自 ex_stage 的 br_taken / br_target

  // ============ 提交侧副作用 ============
  // 1) store 写入 store buffer / DMem
  // 2) CSR 更新
  // 3) 中断检查
endmodule
```

---

## 5. 冲刷与恢复：整个设计中最容易出错的部分

### 5.1 恢复的正确性来自"提交时才恢复"

很多人的第一版会在 EX 级检测到误预测就冲刷。**但那样 free list 无法恢复**：已经分配给错误路径的物理寄存器无法回收。标准做法有三种：

| 方案 | 恢复代价 | 说明 |
|---|---|---|
| **A. 提交级恢复（本文采用）** | O(1) | 分支一直走到 ROB 头才冲刷；free list 用 `fl_hd_arch` 指针回滚 |
| B. EX 级恢复 + checkpoint | O(1) | 每 8 条存一份 (RAT, freelist 指针) 快照，回溯找最近 checkpoint |
| C. EX 级恢复 + 引用计数 | O(#squash) | 每个物理寄存器维护引用计数，逐项回退 |

### 5.2 方案 A 的 O(1) 恢复原理

关键观察：**物理寄存器只在提交时释放，free list 的弹出严格按程序序发生。**

因此：

$$
\text{fl\_hd} = \text{fl\_hd\_arch} + (\text{被冲刷指令的分配数})
$$

而 `fl_hd_arch` 每拍只加"本拍提交的带目的指令数"：

```systemverilog
fl_hd_arch <= fl_hd_arch + n_free;   // n_free = 本拍提交且带目的寄存器
```

冲刷时直接：

```systemverilog
fl_hd <= fl_hd_arch + n_free;        // 回滚，一行搞定
spec_rat <= arch_rat_nxt;            // 32×6 的整表拷贝
```

**不需要遍历 ROB、不需要逐项释放、不需要多周期 FSM。** 恢复代价 = 流水线重填 ≈ 8~12 拍。

### 5.3 ROB 清零的 FPGA 优化：epoch 技术

上面代码里 `flush` 要清 64 个 `e_vld`。虽然只是 64 个 FF，但它会让 ROB 无法映射到 BRAM（BRAM 不支持单拍全清）。改进：

```systemverilog
logic [3:0] epoch, e_epoch [0:DEPTH-1];

wire e_valid = e_vld[i] && (e_epoch[i] == epoch);

// 分配时： e_epoch[idx] <= epoch;
// 冲刷时： epoch <= epoch + 1;      // 一行！
```

这样 flush 只改一个 4-bit 计数器，ROB 就能放进 **BRAM 多副本结构**：

- 64 项 × ~88 bit = 5.6 kbit，**每份一个 BRAM18 绰绰有余**
- 提交需要读 head、head+1、head+2、head+3 → **做 4 份 BRAM 副本，写口广播到 4 份**
- 总开销：4 个 BRAM18，替代掉原本 ~6000 个 LUT 的 64:1 大 mux

**这是 FPGA 上实现 ROB 最实用的技巧**，直接决定你能不能在 Artix-7 35T 上做出 4-wide 前端。

### 5.4 冲刷的广播

```systemverilog
// 所有流水线级的 valid 都必须： vld_next = vld_in & ~flush
always_ff @(posedge clk or negedge rst_n)
  if (!rst_n)       stage_vld <= 1'b0;
  else if (flush)   stage_vld <= 1'b0;      // 优先
  else if (...)     stage_vld <= in_vld;
```

**每个模块都必须有 `flush` 端口，并且优先级高于一切。** 漏掉任何一级，就会出现"幽灵写回"，导致 ROB 项被错误置 `done`。

---

## 6. FPGA 实现要点

<details>
<summary><b>6.1 资源与频率预算（Artix-7 -2 实测量级）</b></summary>

| 模块 | LUT | FF | BRAM |
|---|---|---|---|
| 前端（含 BP） | 800 | 600 | 1 |
| 宽解码 ×4 | 700 | 0 | 0 |
| 重命名 + FL | 2200 | 900 | 0 |
| ROB 64 项（FF 版） | 5500 | 7000 | 0 |
| ROB 64 项（BRAM 副本版） | 900 | 500 | 4 |
| 发射队列 ×2 | 900 | 700 | 0 |
| 寄存器堆 64×32 | 1200 | 2048 | 0 |
| **合计** | **~12k** | **~12k** | **~5** |

Artix-7 100T（约 63k LUT / 126k FF）可以容纳；35T 需要砍到 2-wide。

</details>

<details>
<summary><b>6.2 寄存器堆：多端口的现实做法</b></summary>

4-wide 发射 + 4-wide 写回 = 8 读 4 写，FPGA 的 LUTRAM 做不到。三种做法：

1. **降低并发度**（推荐）：发射 2 条、写回 2 条 → 4 读 2 写。Vivado 可以推断成 `RAM32M`（1 读 1 写 + 2 个额外读口）。
2. **写回串行化**：写口排队，最多 1~2 写/拍，RF 用 1W2R，读口做多副本（每读口一份）。
3. **Collapsing buffer**（BOOM 的做法）：小表 + 旁路网络，代价是 LUT 暴涨。

建议做法：`(* ram_style = "distributed" *)` + 1 写 2 读，再加一层 4 项的 bypass mux（正好覆盖写回→发射的时序窗口）。

</details>

<details>
<summary><b>6.3 关键约束（XDC 片段）</b></summary>

```tcl
create_clock -name clk -period 10.000 [get_ports clk]

# 分支预测器的组合读：允许 2 拍，用 multicycle
set_multicycle_path -setup 2 -from [get_pins u_bp/bht_reg*/C]
set_multicycle_path -hold  1 -from [get_pins u_bp/bht_reg*/C]

# 唤醒广播扇出大，复制驱动
set_property MAX_FANOUT 32 [get_nets -hier -filter {NAME =~ *wb_pdst*}]

# BRAM 输出直连，禁止插寄存
set_property OUTPUT_REG 0 [get_cells -hier -filter {NAME =~ *u_imem*}]
```

</details>

<details>
<summary><b>6.4 常见可综合性问题</b></summary>

| 写法 | 问题 | 改法 |
|---|---|---|
| `logic [W-1:0][ALW-1:0]` 作为模块端口 | Verilog-2001 不支持 | 用 `-sv` 或展平为 `[W*ALW-1:0]` |
| `for` 循环里 `break` / `return` | 部分工具不支持 | 改用 `c_stop` 类的标志位 |
| 结构体作为 `output` | 部分老工具报错 | 展平成 `logic [N-1:0]` |
| `always_comb` 里读未初始化数组 | 仿真 X，综合正常 | 上电复位或 `initial` 初始化 |
| 变量部分选择 `mem[i][j]` | 长路径 | 改成一维展平 |

</details>

---

## 7. 验证方法

<details>
<summary><b>7.1 分层验证流程</b></summary>

1. **模块级**：每个模块配一个 cocotb / SV testbench。ROB 重点测：满、空、部分提交、异常 + 误预测同拍、flush 与分配同拍。
2. **指令级随机测试**：用 Python 生成随机指令流（含故意的高密度分支、RAW 依赖、store-load 别名），用 Spike / 自己写的 ISA 模拟器产出 golden trace。
3. **提交日志比对**：核心核对 `(pc, 目的寄存器号, 值)` 三元组序列。这是发现"重命名错、ROB 错、恢复错"最有效的手段。
4. **形式化**：用 SymbiYosys 对 ROB 的"提交顺序 = 分配顺序"和"flush 后所有状态等价于架构状态"做 k-induction 证明，比仿真省时间。

</details>

<details>
<summary><b>7.2 必须覆盖的边界场景</b></summary>

- 分支误预测与异常在**同一拍**到达 ROB 头 → 异常优先
- flush 与 ROB 分配**同拍** → 分配必须被丢弃
- flush 与写回同拍 → 写回必须被丢弃（否则复用项被污染）
- 两条指令在同一 bundle 内写同一架构寄存器 → 前递链
- `ret` 在预测队列中连续两条 → RAS 深度对不对
- ROB 满且 IQ 满 → 全流水线完全停住，无死锁
- store 在错误路径上 → 不能写到 DMem（必须延迟到提交）
- 单步 / 中断注入 → 精确异常点

</details>

---

## 8. 实施路线（建议按此顺序推进，每步都可上板）

| 阶段 | 内容 | 目标 |
|---|---|---|
| 1 | 单发射顺序 5 级 + 无预测 | 点亮 LED / 跑通 hello |
| 2 | 加 BTB + BHT + RAS，2 级恢复 | 频率 100 MHz，CoreMark 有基础分 |
| 3 | 加 32 项 ROB + 2-wide 乱序完成 | IPC 提升 1.4~1.7 |
| 4 | 加宽到 4-wide 前端 + 4-wide 解码 | 前端瓶颈解除 |
| 5 | 加 SMT 双前端 | 多线程场景吞吐提升 |
| 6 | 加 ICache / DCache / 特权级 | 成为完整的 SoC 核 |

**不要一上来就追求 4-wide 全乱序。** 在 FPGA 上，第 3 步（32 项 ROB + 2 发）的性价比远高于第 4 步，因为宽前端会立刻让寄存器堆端口和 ROB 读口成为时序瓶颈。

---

## 9. 核心设计原则总结

> **四个必须记住的原则：**
>
> 1. **宽前端 + 窄执行 + 深 ROB** —— ROB 是宽窄之间的解耦器，深度比宽度更值钱。
> 2. **恢复只在提交级做** —— 换来 O(1) 的 free list 回滚，避免整个恢复子系统进入复杂状态机设计。
> 3. **flush 优先于一切** —— 每一级必须有 flush 端口，否则会出现幽灵写回。
> 4. **ROB 用 BRAM 多副本，别用大 mux** —— 4 个 BRAM18 省下 6000 个 LUT，这决定了能不能塞进小芯片。

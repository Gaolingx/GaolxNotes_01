# CUDA Stream 同步方式详解

### 🧑‍💻 User

cuda编程中，多个cuda stream之间要如何同步？

---

### 🤖 Assistant

好的，在CUDA编程中，对多个CUDA Stream进行同步是实现复杂并行模式和确保数据依赖关系正确的关键。多个Stream之间的同步主要有以下几种方式，从粗粒度到细粒度排列：

### 1\. `cudaDeviceSynchronize()` - 全局同步 (最强但最低效)

这是最简单粗暴的同步方式，通常被称为“大锤”（big hammer）。

  - **作用**: 阻塞**主机线程 (CPU)**，直到**设备 (GPU) 上所有之前已发布的任务**（包括所有Stream中的所有Kernel执行和内存拷贝）都全部完成。
  - **函数签名**: `cudaError_t cudaDeviceSynchronize(void);`
  - **使用场景**:
      - 程序结束前，确保所有GPU工作都已完成。
      - 在进行性能分析或调试时，需要一个明确的同步点来测量时间或检查错误。
      - 当你无法或不想管理复杂的Stream间依赖关系，需要一个全局屏障时。
  - **缺点**: 完全破坏了并行性。在调用它的时候，CPU会空闲等待GPU，GPU上不同Stream之间本可以并行的任务也失去了并行的可能性。在性能敏感的代码路径中应极力避免使用。

-----

### 2\. `cudaStreamSynchronize()` - 主机与单个Stream同步

这种方式比全局同步更精细，它只针对某一个特定的Stream。

  - **作用**: 阻塞**主机线程 (CPU)**，直到**指定的Stream中**所有之前已发布的任务都完成。
  - **函数签名**: `cudaError_t cudaStreamSynchronize(cudaStream_t stream);`
  - **使用场景**: 当主机需要从某个特定的Stream中获取计算结果时。例如，在一个Stream中执行了一个Kernel，然后主机需要立即读取该Kernel产生的结果（通过 `cudaMemcpy` 从设备复制到主机）。在调用 `cudaMemcpy` 之后，需要调用 `cudaStreamSynchronize` 来确保数据已经准备好。
  - **缺点**: 它会阻塞CPU，如果此时GPU上其他Stream还在忙碌，这部分并行性得以保留，但CPU与被同步Stream之间的并行性丢失了。

-----

### 3\. **CUDA Events** - Stream间同步 (最灵活、最高效的方式)

使用CUDA Event是实现不同Stream之间精细化同步和依赖关系管理的**最佳实践**。它不会阻塞主机线程，而是在GPU上创建依赖关系，让硬件直接处理，从而最大化地保留了并行性。

整个过程分为三步：

1.  **创建Event**: `cudaEventCreate()`
2.  **记录Event**: 在一个Stream（我们称之为 `streamA`）的某个点记录一个Event。这个操作本身是异步的，会立即返回。当 `streamA` 中所有在该记录点**之前**的任务都执行完毕后，该Event的状态才变为“已完成”。
3.  **等待Event**: 让另一个Stream（我们称之为 `streamB`）等待这个Event。这会在 `streamB` 的任务队列中插入一个“等待”指令。`streamB` 会正常执行在“等待”指令之前的任务，当执行到该指令时，它会暂停，直到它所等待的Event变为“已完成”状态，然后才会继续执行后续的任务。

#### 关键函数:

  - **创建/销毁Event**:

    ```c++
    cudaEvent_t event;
    cudaEventCreate(&event);
    // ...
    cudaEventDestroy(event);
    ```

  - **记录Event**:

    ```c++
    // 在streamA中记录event。当streamA中此调用之前的所有任务完成后，event才被触发。
    cudaEventRecord(event, streamA);
    ```

  - **让一个Stream等待Event**:

    ```c++
    // 让streamB等待event。streamB在此调用之后的任务，必须等event被触发后才能开始执行。
    cudaStreamWaitEvent(streamB, event, 0); // 第三个参数flags，通常为0
    ```

  - **让主机等待Event**:

    ```c++
    // 阻塞主机线程，直到event被触发。这是 cudaStreamSynchronize 的事件版本。
    cudaEventSynchronize(event);
    ```

#### 使用场景:

当你有一个依赖关系，例如：Stream B中的某个Kernel (`kernel_B`) 需要使用Stream A中某个Kernel (`kernel_A`) 的计算结果。

**错误的做法**:

```c++
kernel_A<<<..., streamA>>>(...);
kernel_B<<<..., streamB>>>(...); // 错误！无法保证kernel_A执行完
```

**正确的做法 (使用Event)**:

```c++
// 1. 创建Event
cudaEvent_t event_A_done;
cudaEventCreate(&event_A_done);

// 2. 在streamA中启动kernel_A，并在其后记录Event
kernel_A<<<..., streamA>>>(...);
cudaEventRecord(event_A_done, streamA);

// 3. 让streamB等待该Event，然后再启动kernel_B
cudaStreamWaitEvent(streamB, event_A_done, 0);
kernel_B<<<..., streamB>>>(...);

// ...
cudaEventDestroy(event_A_done);
```

这样，`kernel_B` 的执行就被安排在了 `kernel_A` 完成之后，而CPU线程在提交这些任务后可以继续做其他事情，完全不会被阻塞。

-----

### 4\. 隐式同步 (默认Stream的行为)

CUDA有一个特殊的Stream，即**默认Stream**（`stream = 0` 或 `NULL`）。它与其他所有手动创建的Stream（非默认Stream）之间存在隐式同步行为：

  - **规则1**: 在默认Stream中发布一个操作，会等待**所有其他Stream中**此前发布的所有操作全部完成。
  - **规则2**: 在任何一个非默认Stream中发布一个操作，会等待**默认Stream中**此前发布的所有操作全部完成。

这个行为使得默认Stream像一个隐藏的全局同步点，很容易意外地破坏并行性。因此，在现代CUDA编程中，推荐为每一个并发任务都显式创建Stream，并避免混合使用默认Stream和非默认Stream，除非你特意想利用这种隐式同步行为。

### 总结与选择

| 同步方式 | 作用范围 | 是否阻塞主机 | 性能影响 | 推荐使用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **`cudaDeviceSynchronize()`** | 整个GPU设备 | **是** | **巨大** | 调试、程序结束、全局屏障 |
| **`cudaStreamSynchronize()`** | 主机与单个Stream | **是** | **中等** | 主机需要立即获取特定Stream的结果 |
| **CUDA Events** | Stream与Stream之间 | **否** (除非用`cudaEventSynchronize`) | **微小** | **管理不同Stream间的复杂依赖关系，最大化并行** |
| **隐式同步 (默认Stream)** | 默认Stream与其他Stream | 否 (但行为上是同步点) | **高** (易破坏并行) | 避免使用，除非明确意图 |

**最佳实践**: 对于多个Stream之间的同步，**首选且绝大多数情况下应该使用CUDA Events** (`cudaEventRecord` 和 `cudaStreamWaitEvent`)。它提供了最精细的控制粒度，并且不会不必要地阻塞CPU，从而最大限度地发挥GPU的并行计算能力。

---

### 🧑‍💻 User

如果出现多条stream之间形成有向无环图（dag）的依赖，可以用cuda graph吗？

---

### 🤖 Assistant

当然可以！您提出的这个问题——“多条Stream之间形成有向无环图（DAG）的依赖”——**正是CUDA Graph被设计出来要解决的核心场景**。

可以说，对于一个结构固定的、需要重复执行的DAG依赖工作流，使用CUDA Graph是比手动管理Stream和Event**更高效、更现代、也更推荐**的解决方案。

### 为什么CUDA Graph是更优的选择？

如果您使用传统的多Stream和多Event来手动构建一个DAG，那么在每次执行这个DAG时，CPU都需要做以下工作：

1.  向驱动程序提交每一个Kernel Launch。
2.  向驱动程序提交每一次 `cudaMemcpyAsync`。
3.  向驱动程序提交每一次 `cudaEventRecord`。
4.  向驱动程序提交每一次 `cudaStreamWaitEvent`。

当DAG变得复杂（节点和依赖众多）或者执行频率非常高时，CPU提交这些指令的开销（Kernel Launch Latency）会变得非常显著，甚至可能成为整个流程的瓶颈。CPU会一直忙于“发号施令”，而无法去做别的工作。

**CUDA Graph彻底改变了这一点。** 它的核心思想是：**定义一次，运行多次 (Define once, run many times)**。

1.  **定义/捕获 (Capture)**: 你只需要像平常一样，用Stream和Event的API将你的整个DAG工作流执行一次，但这次是在一个特殊的“捕获模式”下。CUDA驱动不会立即执行这些指令，而是将它们记录下来，并分析它们之间的依赖关系，最终构建成一个完整的、原生的图（`cudaGraph_t`）对象。
2.  **实例化 (Instantiate)**: 在启动之前，CUDA Graph会被“实例化”为一个可执行的图（`cudaGraphExec_t`）。在这个阶段，驱动程序会对整个图进行**全局优化**。它可以看到你所有的工作流程，因此可以进行更深度的调度优化、资源管理，甚至是kernel fusion等，这些都是在零散的Stream/Event模型中无法做到的。
3.  **启动 (Launch)**: 之后，每次你需要执行这个DAG时，你只需要调用一个**单一的、非常轻量级**的函数 `cudaGraphLaunch()`。CPU的开销被降到了最低，因为它不再需要提交成百上千个单独的CUDA API调用，而只是简单地告诉GPU：“去执行那个已经优化好的图”。所有的调度和依赖管理都转移到了GPU上，由硬件高效处理。

### CUDA Graph的巨大优势

  * **极低的CPU开销**: 对于重复执行的DAG，启动开销从N次API调用降低到1次API调用，性能提升非常显著，尤其是在有很多小kernel的场景下（例如深度学习推理）。
  * **驱动/硬件层面的全局优化**: 由于驱动在实例化阶段就知道了整个工作流的全貌，它可以进行更智能的优化，从而获得比手动Stream/Event管理更高的GPU执行效率。
  * **代码更清晰**: 将复杂的工作流封装在一个`cudaGraph`对象中，可以使代码逻辑更清晰。它将工作流的“定义”与“执行”分离开来。

### 如何将现有的Stream/Event DAG代码迁移到CUDA Graph？

迁移过程非常直观，基本上就是把你现有的代码用 `cudaStreamBeginCapture` 和 `cudaStreamEndCapture` 包起来。

假设您已经有了一个用`streamA`, `streamB`和`event`构建依赖关系的代码：

**传统方式 (在循环中重复执行)**

```cpp
// 假设这些streams和events已创建
for (int i = 0; i < N_ITERATIONS; ++i) {
    // 更新输入数据...
    cudaMemcpyAsync(d_in_A, h_in_A, size_A, cudaMemcpyHostToDevice, streamA);
    kernelA<<<..., streamA>>>(d_out_A, d_in_A);
    cudaEventRecord(eventA_done, streamA);

    cudaMemcpyAsync(d_in_B, h_in_B, size_B, cudaMemcpyHostToDevice, streamB);
    kernelB<<<..., streamB>>>(d_out_B, d_in_B);
    cudaEventRecord(eventB_done, streamB);
    
    // kernelC 依赖于 kernelA 和 kernelB 的结果
    cudaStreamWaitEvent(streamC, eventA_done, 0);
    cudaStreamWaitEvent(streamC, eventB_done, 0);
    kernelC<<<..., streamC>>>(d_final, d_out_A, d_out_B);
    cudaMemcpyAsync(h_result, d_final, size_final, cudaMemcpyDeviceToHost, streamC);
    cudaStreamSynchronize(streamC);
}
```

**使用CUDA Graph的方式**

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;
cudaStream_t captureStream = streamC; // 可以在任意一个stream上捕获

// ============ 1. 捕获阶段 (执行一次) ============
cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal);

// --- 将一次迭代的完整DAG指令放入捕获块 ---
cudaMemcpyAsync(d_in_A, h_in_A, size_A, cudaMemcpyHostToDevice, streamA);
kernelA<<<..., streamA>>>(d_out_A, d_in_A);
cudaEventRecord(eventA_done, streamA);

cudaMemcpyAsync(d_in_B, h_in_B, size_B, cudaMemcpyHostToDevice, streamB);
kernelB<<<..., streamB>>>(d_out_B, d_in_B);
cudaEventRecord(eventB_done, streamB);

cudaStreamWaitEvent(captureStream, eventA_done, 0);
cudaStreamWaitEvent(captureStream, eventB_done, 0);
kernelC<<<..., captureStream>>>(d_final, d_out_A, d_out_B);
cudaMemcpyAsync(h_result, d_final, size_final, cudaMemcpyDeviceToHost, captureStream);
// ---------------------------------------------

cudaStreamEndCapture(captureStream, &graph);

// ============ 2. 实例化阶段 (执行一次) ============
// 可以在这里检查图的合法性等
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// ============ 3. 启动阶段 (在循环中，开销极低) ============
for (int i = 0; i < N_ITERATIONS; ++i) {
    // 如果需要，更新输入数据...
    // 注意：如果kernel的参数或内存地址在每次迭代中会变，需要使用 cudaGraphExecUpdate
    // 这里假设它们不变
    cudaGraphLaunch(instance, captureStream);
    cudaStreamSynchronize(captureStream);
}

// ============ 4. 清理 ============
cudaGraphExecDestroy(instance);
cudaGraphDestroy(graph);
```

### 结论

**是的，不仅可以用，而且强烈推荐使用CUDA Graph来处理多Stream形成的DAG依赖关系。**

  - 如果你的工作流是**静态的**（即计算图的结构、kernel参数等不随迭代而改变），CUDA Graph会带来巨大的性能收益。
  - 如果你的工作流是**动态的**（即每次迭代的计算路径都可能不同），那么传统的Stream/Event模型依然是更灵活的选择。

---

### 🧑‍💻 User

相较于手写event，将多条stream之间的依赖关系抽象成cuda graph有什么好处？

---

### 🤖 Assistant

好的，这是一个非常深刻的问题。将多条Stream之间的依赖关系从手动编写 `cudaEvent` 升级到抽象成 `cudaGraph`，带来的好处是多维度、且非常显著的。

我们可以用一个核心比喻来理解：**手动管理Event就像是“解释执行”一段脚本，而使用CUDA Graph就像是“编译后执行”一个程序。**

下面，我们详细展开这个比喻，列出CUDA Graph的核心优势：

### 1\. 核心优势：显著降低CPU开销 (CPU Overhead)

这是CUDA Graph最直接、也是最大的性能优势。

  * **手动Event模式 (解释执行)**:
    在每次循环或每次执行你的工作流（DAG）时，CPU都需要向CUDA驱动程序逐一提交**所有**的指令：`cudaMemcpyAsync`, `kernel_launch`, `cudaEventRecord`, `cudaStreamWaitEvent` 等等。如果你的DAG有几十上百个节点，那就意味着CPU每一轮都要进行几十上百次API调用。这些调用本身是有开销的，我们称之为“提交开销”或“启动延迟”。当GPU上的任务本身很短时，这个CPU开销甚至可能成为整个系统的瓶颈。

  * **CUDA Graph模式 (编译后执行)**:
    你只需要在程序初始化阶段，通过“捕获”的方式将整个DAG的指令序列**定义一次**。CUDA驱动会接收这完整的指令图，并将其“编译”成一个优化的、原生的GPU工作包（`cudaGraphExec_t`）。之后在每次循环或执行时，CPU**只需要提交一条指令**：`cudaGraphLaunch()`。CPU的负担从提交N个指令骤降到提交1个指令，开销几乎可以忽略不计。这极大地释放了CPU资源，使其可以去处理其他任务，并且对于延迟敏感型应用至关重要。

### 2\. 驱动/硬件层面的全局优化 (Global Optimization)

  * **手动Event模式**:
    驱动程序一次只能看到一个指令。它就像一个短视的工人，你告诉他“搬砖”，他就搬；你告诉他“砌墙”，他就砌。他无法知道你最终的目标是盖一座房子，因此他只能进行局部、孤立的优化。

  * **CUDA Graph模式**:
    在你“捕获”并“实例化”图的时候，驱动程序**获得了整个工作流的全局视角**。它知道了所有的任务、所有的数据流向和所有的依赖关系。这使得驱动可以进行一系列深度的、在手动模式下不可能实现的全局优化，例如：

      * **更优的调度**: 预先规划所有kernel的执行顺序和资源分配，减少调度冲突。
      * **资源预分配**: 提前分配好所有需要的内部资源，避免运行时动态分配的开销。
      * **潜在的融合**: 在某些情况下，驱动可能能够将图中的某些节点（例如连续的小kernel）融合成一个更大的kernel，进一步减少开销。
      * **减少寄存器压力**: 通过全局分析，更有效地管理寄存器使用。

这些优化完全由驱动在底层完成，最终转化为GPU执行效率的提升。

### 3\. 代码抽象层次更高，逻辑更清晰 (Better Abstraction & Readability)

  * **手动Event模式**:
    工作流的“定义”（各种API调用）和“执行”（循环调用这些API）是混杂在一起的。当依赖关系变得非常复杂时（一个复杂的DAG），代码会充斥着大量的`cudaEventRecord`和`cudaStreamWaitEvent`，变得难以阅读和维护。

  * **CUDA Graph模式**:
    它强制你将工作流的\*\*定义（Capture & Instantiate）**和**执行（Launch）\*\*清晰地分离开来。`cudaGraph_t`对象本身就成为了一个代表完整计算任务的可复用组件。这使得代码结构更清晰，意图更明确。你可以轻松地创建、存储、甚至序列化这个计算图，大大提高了代码的模块化和可维护性。

### 总结对比表格

| 特性/方面 | 手动管理Event | CUDA Graph | 优势说明 |
| :--- | :--- | :--- | :--- |
| **执行模型** | **命令式 (Imperative)** \<br\> CPU逐条下达指令 | **声明式 (Declarative)** \<br\> CPU声明要执行的图 | Graph模式更符合现代GPU编程思想，将控制权更多地交给驱动和硬件。 |
| **CPU开销** | **高** (每次执行都需要提交所有API调用) | **极低** (每次执行只需一次`cudaGraphLaunch`调用) | **核心性能优势**。尤其在小kernel或高频率执行场景下，提升巨大。 |
| **优化级别** | **局部/单次调用优化** | **全局/整体工作流优化** | 驱动拥有上帝视角，能做出更智能的优化决策，提升GPU效率。 |
| **代码结构** | 定义与执行耦合，逻辑分散 | 定义与执行分离，封装性好 | 代码更清晰，模块化程度更高，更易于维护和复用。 |
| **执行确定性** | 较低，受CPU调度影响 | 较高，图一旦实例化，执行路径更固定 | 更容易获得可复现的性能表现。 |
| **最适用场景** | **动态工作流**\<br\>(每次迭代的计算路径或依赖关系都不同) | **静态工作流**\<br\>(计算图的结构固定，需要重复执行) | CUDA Graph专为可重复的模式而设计，例如DL推理、迭代求解器等。 |

### 结论

**从手动编写`event`到使用`cuda graph`，不仅仅是一种API的替换，更是一种编程思想的升维。**

它将程序员从繁琐的、命令式的底层调度中解放出来，让你更专注于“做什么”（定义计算图），而不是“怎么做”（手动管理同步）。作为回报，CUDA运行时和驱动能够以远超手动的效率去执行这个任务。

因此，只要你的多Stream依赖关系是一个需要重复执行的固定DAG，那么毫无疑问，**CUDA Graph是实现最高性能和最佳代码结构的不二之选。**
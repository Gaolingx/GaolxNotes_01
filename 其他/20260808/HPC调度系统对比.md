# HPC调度系统对比

### 🧑‍💻 User

Sun Grid Engine vs slurm?

---

### 🤖 Assistant

# Sun Grid Engine (SGE) vs Slurm

## Overview

| Aspect | Sun Grid Engine (SGE) | Slurm |
|--------|----------------------|-------|
| **Origin** | Sun Microsystems (2000) | Lawrence Livermore National Lab (2003) |
| **License** | Proprietary → Open source forks | Open source (GPL) |
| **Current Status** | Legacy/maintenance mode | Actively developed |
| **Market Share** | Declining | Dominant in HPC (~60%+ of Top500) |
| **Primary Maintainer** | Univa (commercial) / OGS (community) | SchedMD |

---

## Architecture Comparison

<details>
<summary>🔬 Technical Architecture Details</summary>

### SGE Architecture
- **Master/Exec daemon model**: `sge_qmaster` + `sge_execd`
- Uses Berkeley DB or PostgreSQL for accounting
- Hierarchical queue structure
- Component-based resource management

### Slurm Architecture
- **Controller/Compute daemon model**: `slurmctld` + `slurmd`
- Optional database daemon (`slurmdbd`) for accounting
- Flat partition-based structure
- Plugin-based architecture for extensibility

</details>

---

## Command Comparison

| Operation | SGE | Slurm |
|-----------|-----|-------|
| **Submit job** | `qsub script.sh` | `sbatch script.sh` |
| **Interactive job** | `qrsh` or `qsh` | `salloc` or `srun --pty bash` |
| **Job status** | `qstat` | `squeue` |
| **Job info** | `qstat -j <jobid>` | `scontrol show job <jobid>` |
| **Delete job** | `qdel <jobid>` | `scancel <jobid>` |
| **List queues** | `qconf -sql` | `sinfo` |
| **Node info** | `qhost` | `scontrol show nodes` |
| **Hold job** | `qhold <jobid>` | `scontrol hold <jobid>` |

---

## Job Script Comparison

### SGE Script Example
```bash
#!/bin/bash
#$ -N my_job
#$ -q normal
#$ -pe mpi 16
#$ -l h_rt=24:00:00
#$ -l mem=4G
#$ -o output.log
#$ -e error.log

module load gcc
mpirun -np 16 ./my_program
```

### Slurm Script Example
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=normal
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --output=output.log
#SBATCH --error=error.log

module load gcc
srun ./my_program
```

---

## Key Differences

### 🟢 SGE Advantages
- Simpler configuration for small clusters
- Mature queue management
- Good for traditional batch workloads
- Familiar to legacy HPC users

### 🟢 Slurm Advantages
- **Active development** with regular releases
- Better **scalability** (tested on 100k+ node clusters)
- Rich **plugin ecosystem**
- Superior **GPU support** and heterogeneous jobs
- Better **cloud integration** capabilities
- Strong **community support** and documentation
- Native **container support**

<details>
<summary>📋 Feature Comparison Matrix</summary>

| Feature | SGE | Slurm |
|---------|-----|-------|
| GPU scheduling | Limited (via complex values) | Native support |
| Cloud bursting | Third-party tools | Built-in plugins |
| Containers | Limited | Native (Apptainer, Docker) |
| Accounting | Basic | Comprehensive (sacct) |
| Fair share | Basic | Advanced (TRES-based) |
| Reservation system | Limited | Robust |
| MPI integration | Via parallel environments | Native support |
| Heterogeneous jobs | Difficult | First-class support |

</details>

---

## Mathematical Performance Considerations

Slurm's scheduling complexity scales as:

$$T_{schedule} = O(n \log n)$$

where $n$ is the number of jobs, while SGE typically exhibits:

$$T_{schedule} \approx O(n^2)$$

for large job counts, making Slurm more efficient at scale.

---

## Migration Considerations

If migrating from SGE to Slurm:

1. **Script conversion**: Use tools like `sge2slurm` or manual conversion
2. **Queue → Partition mapping**: Hierarchical queues become flat partitions
3. **Parallel environments → Tasks**: `#$ -pe mpi 16` becomes `#SBATCH --ntasks=16`
4. **Resource specification**: SGE complexes → Slurm TRES (Trackable RESources)

<details>
<summary>🔄 Common Migration Pitfalls</summary>

1. **Memory specification**: SGE uses `mem=4G` per slot; Slurm defaults to per-node
2. **Job arrays**: Syntax differs significantly
3. **Task distribution**: SGE `-pe` vs Slurm `--ntasks-per-node`
4. **Environment variables**: Different variable names (`$JOB_ID` vs `$SLURM_JOB_ID`)
5. **Array indexing**: SGE starts at 1; Slurm configurable (default 0)

</details>

---

## Recommendation

| Use Case | Recommended |
|----------|-------------|
| New HPC deployment | **Slurm** |
| Legacy SGE cluster | Maintain or migrate to Slurm |
| Small lab cluster (<50 nodes) | Either (Slurm preferred) |
| Cloud-native workloads | **Slurm** |
| GPU-heavy workloads | **Slurm** |

**Bottom line**: Slurm is the current industry standard with active development, better scalability, and superior support for modern HPC workloads. SGE remains functional but is primarily for legacy systems or organizations with existing SGE expertise.
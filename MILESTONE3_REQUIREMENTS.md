# Milestone 3 — Final Submission Requirements
**CS-3006 · Parallel & Distributed Computing · Spring 2026**
**Group:** Muhammad Umair (23i-0662) · Awais Basheer (23i-0506)
**Base Paper:** Merge-Based Parallel Sparse Matrix-Vector Multiplication — Merrill & Garland, SC'16
**Deadline:** May 8, 2026 | **Final Presentation:** May 11–13, 2026

---

## 1. What Must Be Submitted

| # | Deliverable | Notes |
|---|---|---|
| 1 | **Final project report** (paper-draft format) | See Section 5 for required structure |
| 2 | **Complete source code** | Baseline + parallel/SIMD implementations |
| 3 | **Presentation slides** | Full deck covering all sections below |
| 4 | **Presentation video** | YouTube or Google Drive link only |
| 5 | **README** | Build & execution instructions |
| 6 | **Scripts / datasets / config files** | Everything needed to reproduce results |

---

## 2. Final Presentation Must Cover

1. Brief recap of the selected base paper
2. Clearly defined project scope
3. Implementation details (languages, tools, architecture)
4. Baseline method (sequential or reference)
5. Optimization / parallelization approach + **correctness verification**
6. Results — critical analysis including:
   - Speedup, efficiency, and/or scalability comparisons
   - Performance trends, bottlenecks, and limitations
   - Reflection on deviations from expected behavior
7. **LLM Usage Disclosure and Reflection**
   - Where and how LLMs were used
   - Representative prompts used
   - Benefits and limitations observed
   - Correctness risks and over-reliance concerns
   - Statement of academic integrity
8. Conclusions
9. References

> A **live demonstration** is required where applicable.

---

## 3. Hardware-Level Performance Evaluation (Teacher's Addition)

> **Instructor's note (posted after Milestone 2):**
>
> *"For Milestone-3, along with the required code, report, presentation slides, and results, each group is strongly encouraged to perform hardware-level performance evaluation of the system used for experiments. Your final report should not only show execution time and speedup, but should also explain **why** performance improves or does not improve."*

### What This Means for Us

In addition to basic speedup/efficiency tables, the report must **characterize our hardware** and **map our implementation's behavior onto those limits**.

#### Tool: Empirical Roofline Model (ERT / Intel Advisor / likwid)
Use a roofline tool to measure:

| Hardware Metric | How to Measure |
|---|---|
| Peak compute (GFLOPs/s) | ERT, likwid, or `perf` with FP counters |
| Peak memory bandwidth (GB/s) | `stream` benchmark or ERT |
| L1/L2/L3 cache bandwidth | Roofline tool or likwid |
| Achieved bandwidth of our SpMV | `bytes_read / runtime` |
| Arithmetic intensity (AI) | `flops / bytes` — for SpMV ≈ `2·nnz / (bytes accessed)` |

#### The Four Binding Questions
Your report must clearly answer **which regime our implementation falls into** and **why**:

| Regime | Symptom | Likely Cause |
|---|---|---|
| **Compute-bound** | GFLOPs close to peak; adding threads scales perfectly | Dense inner loop, fully vectorized |
| **Memory-bound** | Bandwidth saturated; more threads don't help | Random `x[col[i]]` gathers, cache misses |
| **Synchronization / communication overhead** | Efficiency drops at high thread counts | Fix-up pass, false sharing, OpenMP overhead |
| **Poor data locality** | Low achieved bandwidth despite low thread count | Irregular column indices, cold cache |

#### Concrete Steps

```bash
# 1. Install ERT (Empirical Roofline Toolkit)
git clone https://bitbucket.org/berkeleylab/cs-roofline-toolkit.git

# 2. Run STREAM benchmark for memory bandwidth
wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c
gcc -O3 -fopenmp -march=native stream.c -o stream && ./stream

# 3. Use perf for hardware counter-based analysis
perf stat -e cache-misses,cache-references,instructions,cycles ./cpu_spmv --mtx=matrix.mtx

# 4. Use likwid (if available)
likwid-perfctr -C 0-7 -g MEM ./cpu_spmv --mtx=matrix.mtx

# 5. Compute arithmetic intensity manually
# AI = 2*nnz / (bytes_read)
# bytes_read = (nnz * 8)          # values[] (double)
#            + (nnz * 4)          # col_indices[] (int)
#            + ((num_rows+1) * 4) # row_offsets[] (int)
#            + (nnz * 8)          # x[] gather (worst case: all unique)
```

#### What to Plot
- **Roofline plot**: mark our implementation's (AI, GFLOPs/s) point on the ridge line
- **Bandwidth utilization** vs. thread count
- **Cache miss rate** for regular vs. irregular matrices
- **Efficiency** = speedup / num_threads vs. thread count

---

## 4. Evaluation Rubric (100 marks total)

| Criterion | Weight | Excellent (100%) |
|---|---|---|
| Milestone 1: Paper Selection | 10 | Clear, specific understanding + well-justified choice |
| Milestone 2: Presentation & Gap Analysis | 10 | Deep understanding; clearly articulated gap; feasible plan |
| Baseline Implementation | 10 | Correct sequential impl; clean code; verified output |
| **Parallel / Optimized Implementation** | **25** | Well-designed parallelization; correct OpenMP/SIMD; clean + documented |
| **Performance Evaluation & Analysis** | **20** | Speedup + efficiency + multiple input sizes + charts + explanation of WHY |
| Final Report Quality | 15 | Well-structured; complete sections; professional writing |
| Final Presentation & Defense | 10 | Both members explain confidently; strong answers |

> ⚠️ Performance Evaluation (20 marks) now implicitly includes hardware-level characterization per the instructor's additional requirement.

---

## 5. Report Structure

```
1. Introduction
2. Base Paper and Problem Context
3. Project Scope and Objectives
4. Baseline Method
5. Proposed Parallel / Optimized Approach
6. Experimental Setup
   - Hardware specifications (CPU model, cores, cache sizes, RAM)
   - Compiler, flags, OS
   - Input matrix descriptions (size, nnz, coefficient of variation)
   - Hardware limits (measured peak bandwidth, peak compute)
7. Results and Discussion
   - Speedup charts (thread count vs. speedup)
   - Efficiency charts
   - Scalability analysis
   - Roofline / bandwidth analysis
   - Explanation of WHY performance changes
8. Conclusion
9. References
```

---

## 6. Academic Integrity Checklist

- [ ] All external code and borrowed ideas are acknowledged
- [ ] Paper's baseline code is clearly distinguished from our contributions
- [ ] LLM usage documented with representative prompts
- [ ] Both group members can explain every part of the submitted work
- [ ] No plagiarism — work reflects genuine understanding

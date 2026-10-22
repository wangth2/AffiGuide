# AffiGuide

**AffiGuide** is an affinity-guided diffusion framework for antibody CDR sequence–structure co-design. It combines a DiffAb generator with two pretrained binding-property predictors: the Helixon binding-ddG predictor (**D**) and GearBind (**G**).

![Overview of AffiGuide and D/G consensus guidance](assets/overview.png)

The default method is **DG50 + K20 + Hydro**: scheduled D/G consensus guidance, multiple-candidate selection and a hydropathy-aware sequence prior. Run it through the original `design_pdb.py` or `design_testset.py` entry point. The supplied updated configuration supports **CDR-H3**; earlier configurations remain available through `--config`.

[Installation](#installation) · [Model weights](#model-weights) · [Quick start](#quick-start) · [Default method](#default-method) · [Optional settings](#optional-settings) · [GPU and parallel execution](#gpu-and-parallel-execution)

## Installation

Use **Linux x86_64**, Conda and an NVIDIA GPU compatible with the bundled runtime. The default backend includes kernels for compute capabilities **8.0, 8.6 and 8.9**; see [GPU and parallel execution](#gpu-and-parallel-execution).

From the repository root, create the main environment and the separate GearBind environment:

```bash
conda env create -f envs/affiguide-ddgpred.yml
conda env create -f envs/affiguide-gearbind.yml
conda activate affiguide

export AFFIGUIDE_GEARBIND_PYTHON="$(conda run -n affiguide-gearbind python -c 'import sys; print(sys.executable)')"
python scripts/check_environment.py --device cuda:0
```

Keep the `affiguide` environment active when running either design entry. The default DG method needs both environments; D-only guidance with D-only selection needs only the main environment. Set `AFFIGUIDE_GEARBIND_PYTHON` again in a new shell, or point it to the absolute path of the GearBind environment's Python.

The supplied environments use Python 3.8, PyTorch 1.12.1 and CUDA 11.3. The bundled GearBind extension requires this Linux/Python/PyTorch ABI. Keep these versions when using the bundled extension; a different environment requires a separately validated build. The YAML files pin key packages but are not full environment locks.

The environment check imports both runtimes, loads the bundled GearBind extension and checks GPU access. It does not download weights or run a design. Use `--device cpu` for an import-only check; this does not validate GPU execution. D-only users can check just the main environment with `--guidance d-only`.

The GearBind subprocess automatically prioritizes its own Conda `lib` directory while preserving external NVIDIA driver paths. No global `LD_LIBRARY_PATH` change is needed. If you encounter `GLIBCXX_3.4.31 not found` with an existing environment, update it with `conda env update -n affiguide-gearbind -f envs/affiguide-gearbind.yml`, verify that `AFFIGUIDE_GEARBIND_PYTHON` points to that environment, and rerun the check. Do not replace the system C++ library.

<details>
<summary>PyTorch 1.12.1 matrix-exponential diagnostics</summary>

This PyTorch version can emit an internal buffer-resize deprecation warning during rotation matrix exponentials and their derivatives. AffiGuide filters only the matching `3×3`/`6×6` warning around these operations, only on PyTorch 1.12.1. The original matrix-exponential and autograd calculations are unchanged; other warnings and errors remain visible. Set `AFFIGUIDE_SHOW_MATRIX_EXP_WARNINGS=1` to see the unfiltered diagnostics.

</details>

## Model weights

**Missing required weights are downloaded automatically on first use.** Files are checked against their expected size and SHA256 before loading. You can also fetch all three checkpoints before running a design:

```bash
python scripts/download_weights.py
python scripts/download_weights.py --check-only
```

The expected locations are:

- **DiffAb:** `trained_models/codesign_single.pt`
- **D predictor:** `diffab/tools/ddg/data/model.pt`
- **GearBind:** `trained_models/gearbind2.pth`

[MODEL_MANIFEST.json](MODEL_MANIFEST.json) records the exact checkpoint identities and upstream download URLs. AffiGuide's updated sampler uses these pretrained models without additional model training. Checkpoints are downloaded from upstream sources and are not stored in this source repository.

If an upstream host requires manual access or a download fails, obtain the matching checkpoint from the source listed in the manifest and place it at the corresponding path. Run `--check-only` to verify it. Add `--offline` to a design command to require existing local files and forbid downloads. Model weights retain their upstream terms.

## Quick start

### Design from your own complex

Provide an antibody–antigen complex PDB and its antibody chain IDs:

```bash
python design_pdb.py input.pdb --heavy H --light L --num-samples 1
```

This uses **DG50 + K20 + Hydro on CDR-H3**, with one worker. Replace `input.pdb`, `H` and `L` with your input and chain IDs. For a heavy-chain-only antibody, omit `--light`. The entry performs antibody renumbering; use `--no_renumber` only when the input already has the required Chothia numbering.

Use `--num-samples N` to request N complete designs per selected CDR and `--device cuda:0` to select a CUDA-visible device. If `--num-samples` is omitted, the default YAML requests **100 designs**.

Outputs are written under `results/` by default. Each updated-method design includes a designed PDB, posterior data, the effective configuration and a resumable checkpoint. `EFFECTIVE_CONFIG.json` records the settings and time schedule used. To continue an interrupted run, repeat its method/input/seed arguments and add `--resume`.

Normal single-worker runs show sampling progress and a final `Output: /path/to/results` line. Routine status details are saved in `log.txt`; warnings and errors remain visible. Environment and weight verification reports are printed only when you explicitly run their diagnostic scripts.

### Run a test-set example

Download the [SAbDab structure archive](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/) and extract it under `data/`, giving `data/all_structures/chothia/`. The repository includes `data/sabdab_summary_all.tsv`; dataset paths and the test split are specified under `dataset.test` in the [default YAML](configs/test/codesign_single_dg.yml). The loader creates the processed cache on first use.

Then run one entry of that test split by index:

```bash
python design_testset.py 0 --num-samples 1
```

The small PDB examples included in the repository do not replace the full test-set structures. Changing the dataset snapshot or split can change which complex an index selects.

## Default method

Both design entries use [configs/test/codesign_single_dg.yml](configs/test/codesign_single_dg.yml). Reverse diffusion proceeds from **t100 to t1**:

- **t100–t51:** D-only coordinate and sequence gradients, rotation guidance off, one candidate per step.
- **t50–t1:** D/G consensus for coordinates, rotations and sequence, with 20 candidates per step.

Candidate selection uses **D/G joint scoring (`dg_minimax`)**. D/G hard scores are also recorded during the early single-candidate steps; the D-only label above describes the online gradients. The sequence update uses MTV with the Hydro prior enabled (`hydro_prior_b: 0.8`). Coordinate, rotation and sequence guidance strengths default to `4.0`, `1.0` and `1.0`, respectively, with a `5°` rotation cap.

**K is the number of candidates within a diffusion step.** `--num-samples` is the number of complete designs. Users who want the main method can leave all guidance parameters at their defaults.

## Optional settings

Explicit command-line arguments override the corresponding YAML values. Preview the resolved method, time schedule and model requirements without loading models, downloading weights or using a GPU:

```bash
python design_pdb.py --print-config
```

The same options are available for both entries:

```bash
python design_pdb.py --help
python design_testset.py --help
```

<details>
<summary><strong>Guidance mode, candidate count and channel controls</strong></summary>

- `--K N` / `--candidate-count N`: per-step candidate count inside the candidate window; K is 1 outside it.
- `--guidance dg`: scheduled D-only and D/G consensus gradients, as in the default method.
- `--guidance d-only`: D gradients only; with the default `selector: auto`, this also uses D-only candidate scoring and does not load G.
- `--guidance selection-only`: disable online energy gradients while keeping candidate scoring and any enabled Hydro prior.
- `--selector auto|d_minimum|dg_minimax|single`: candidate scoring rule. `auto` resolves to `d_minimum` in D-only mode and `dg_minimax` in DG/selection-only mode. Explicitly selecting `dg_minimax` still needs G even with D-only gradients. `single` requires K1; the fully unguided historical path uses an original YAML and `--no_sample_ddg`.
- `--coord-guidance on|off`, `--rot-guidance on|off`, `--seq-guidance on|off`: switches for the coordinate (`p`), rotation (`omega`) and sequence (`q`) energy-gradient channels.
- `--coord-scale X`, `--rotation-scale X`, `--seq-scale X`: finite, nonnegative strengths; zero disables the corresponding energy gradient.
- `--hydro_prior_b X` / `--no_sample_hydro`: set or disable the separate Hydro prior.

Turning off an energy-gradient channel leaves its base diffusion active. It does not disable candidate scoring or Hydro. Choosing K1 retains the updated sampling protocol; select an original YAML for the historical method.

For example, inspect D-only guidance and selection with five candidates:

```bash
python design_pdb.py --guidance d-only --K 5 --print-config
```

The YAML counterparts are `gradient_source: DG|D|none`, `candidate_count`, `selector`, `channels`, `coordinate_scale`, `rotation_scale`, `sequence_scale` and `hydro_prior_b`. `--gradient-source DG|D|none` remains a mode alias; conflicting mode flags are rejected.

</details>

<details>
<summary><strong>Guidance and candidate time windows</strong></summary>

These options each take `START_T END_T`:

- `--d-window`: D-only gradient interval.
- `--dg-window`: D/G consensus interval, requiring DG mode.
- `--coord-window`, `--rotation-window`, `--seq-window`: active intervals for individual channels.
- `--candidate-window`: interval using the configured K candidates.

Intervals include both endpoints and must satisfy `100 >= START_T >= END_T >= 1`. D-only and DG intervals must not overlap; gaps disable online gradients. A channel acts at the intersection of its own interval and a source interval, provided its switch is on and its strength is positive. Candidate timing is independent of gradient timing.

In YAML, use `d_window`, `dg_window`, `candidate_window` and `channel_windows`, for example `channel_windows: {p: [100, 1], omega: [50, 1], q: [100, 1]}`. An empty list `[]` disables an interval. Null source/candidate windows derive their defaults from the mode and the existing start-time fields. The automatic D-only interval in DG mode covers the prefix before DG starts; it does not resume after DG ends. D-only mode defaults to `[100, 1]`. Coordinate/sequence channel windows default to `[100, 1]`; rotation defaults to `[rotation_start_t, 1]`.

Inspect the resolved schedule after changing time settings:

```bash
python design_pdb.py --d-window 100 71 --dg-window 50 10 \
  --rot-guidance off --seq-window 80 10 --candidate-window 50 1 --print-config
```

</details>

### Earlier configurations

To use the original guided method, select its unchanged configuration explicitly:

```bash
python design_pdb.py input.pdb --heavy H --light L --cdr H3 --num-samples 1 \
  --config configs/test/codesign_single_guided.yml
```

Original single-CDR, multi-CDR and other task configurations remain under [configs/test/](configs/test/); training configurations remain under [configs/train/](configs/train/). They retain their original sampling options. Updated-only guidance and multi-worker options are rejected for legacy configurations. The `--ddg_ref` and `--ddg_guidance_model` flags belong to the legacy path. The old coordinate/sequence strength aliases, `--gradient_scale_cord` and `--gradient_scale_categ`, remain supported.

## GPU and parallel execution

**Recommendation:** Keep the default single worker for the first run; for concurrent DG designs, a compatible **24 GB GPU such as an RTX 4090 and at least 32 GB host RAM** are a practical starting point, with `--workers 2` before considering four workers. Increase concurrency only after checking combined generator/D/G peak memory and throughput on your inputs; MPS is optional and does not share AffiGuide model weights across workers or guarantee a speedup.

This is a workload-based recommendation, not a minimum-memory specification. NVIDIA describes [MPS as useful when individual processes underutilize a GPU](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html).

<details>
<summary><strong>Hardware reference, MPS and execution details</strong></summary>

The bundled `paper_v4` device kernels cover **SM80/86/89**. Examples at those compute capabilities include A100, RTX A6000 and RTX 4090; see NVIDIA's [GPU compute-capability list](https://developer.nvidia.com/cuda/gpus). Architecture coverage alone does not establish full application acceptance or adequate memory. H100/H200 and RTX 50-series GPUs are outside the current bundled kernel coverage, even if they have more memory. The GearBind extension additionally requires Linux x86_64, Python 3.8, PyTorch 1.12.1, CUDA 11.3 and C++ ABI 0; see its [runtime notes](diffab/tools/GearBind/native/README.md).

Prior full-release measurements on an RTX 4090 24 GB provide a reference: single-worker DG50 + K20 + Hydro H3 runs had sampled device-memory peaks of **3.92–4.70 GiB**, and a separate four-worker private-MPS run had a sampled device-memory peak of **15.08 GiB** and process-tree PSS peak of **19.24 GiB**. Device usage was sampled every 0.5 seconds and included GearBind and CUDA contexts; short peaks can be missed. These observations used the earlier release launcher, main/GearBind intra-op thread counts of 1/2 per worker and specific H3 workloads. The current compact launcher has not repeated that GPU/MPS acceptance run. The 24 GB GPU / 32 GB RAM recommendation allows headroom for those observed workloads and is not a universal bound.

`--workers N` launches independent design workers on the selected GPU; each worker loads its own models and, for DG, starts its own GearBind process. `--num-samples` still sets the total number of complete designs per CDR, while `--K` controls per-step candidates. Start with two workers only after the single-worker command succeeds:

```bash
python design_pdb.py input.pdb --heavy H --light L --num-samples 4 --workers 2
```

Add `--mps` to request a private NVIDIA MPS service, or use `--no-mps` to disable AffiGuide-managed MPS. YAML equivalents are `execution: {workers: 1, mps: false}`. Managed MPS requires installed MPS tools and an idle GPU allocated to your job; the controller verifies the selected GPU and manages only its own service. It does not stop external services or change the GPU compute mode. `--no-mps` still inherits the external environment and does not disable a cluster-managed MPS service.

Worker assignment preserves logical design IDs and their seed derivation. This is not a promise of GPU bitwise equality across worker counts or machines. Monitor host RAM and CPU load as well as GPU memory; the default main-process thread count can depend on the environment, and excessive CPU threads across workers can reduce throughput. Parallel logs, plans and status are written below the output directory in `_execution/`. Legacy `--batch_size` controls legacy sampling; the updated sampler processes one trajectory at a time per worker.

</details>

## Release scope

- The updated DG configuration includes **H3 calibration only**. H1/H2/L1/L2/L3 require their own matching calibration resources; H3 calibration is not substituted for them.
- This compact distribution contains the runtime and original configurations. It does not include the complete paper evaluation suite or separate ablation presets.
- The default single-worker **DG50 + K20 + Hydro** H3 PDB workflow completed all 100 sampling steps on an **RTX 4090 24 GB**, using Python 3.8.20, PyTorch 1.12.1 and CUDA 11.3. Automatic first-use downloads and SHA256 checks passed for all three checkpoints. The matrix-exponential warning display change preserved the complete sampling trace, designed PDB and posterior probabilities exactly in the repeated GPU run. The CPU validation suite contains **92 passing checks**.
- Fresh creation of both Conda environments, the full `design_testset.py` GPU workflow and parallel/MPS execution have not yet been revalidated for this compact release. The single-example PDB run is not a full benchmark reproduction.
- Resume requires matching method, input and source identities. Checkpoints from the earlier V4 source tree require that source version.

## Attribution and license

AffiGuide builds on [DiffAb](https://github.com/luost26/diffab), [Antibody-Diffusion-Properties](https://github.com/amelvim/antibody-diffusion-properties), the [Helixon binding-ddG predictor](https://github.com/HeliXonProtein/binding-ddg-predictor) and [GearBind](https://github.com/DeepGraphLearning/GearBind). Third-party components retain their original licenses. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## References

If you use this code, please cite the relevant upstream methods.

<details>
<summary><strong>Upstream BibTeX entries</strong></summary>

```bibtex
@inproceedings{luo2022antigenspecific,
  title={Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures},
  author={Shitong Luo and Yufeng Su and Xingang Peng and Sheng Wang and Jian Peng and Jianzhu Ma},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=jSorGn2Tjg}
}
```

```bibtex
@article{villegas2024guiding,
  title={Guiding diffusion models for antibody sequence and structure co-design with developability properties},
  author={Amelia Villegas-Morcillo and Jana M. Weber and Marcel J. T. Reinders},
  journal={PRX Life},
  volume={2},
  number={3},
  pages={033012},
  year={2024}
}
```

```bibtex
@article{shan2022deep,
  title={Deep learning guided optimization of human antibody against SARS-CoV-2 variants with broad neutralization},
  author={Shan, Sisi and Luo, Shitong and Yang, Ziqing and Hong, Junxian and Su, Yufeng and Ding, Fan and Fu, Lili and Li, Chenyu and Chen, Peng and Ma, Jianzhu and others},
  journal={Proceedings of the National Academy of Sciences},
  volume={119},
  number={11},
  pages={e2122954119},
  year={2022}
}
```

```bibtex
@article{cai2024gearbind,
  title={Pretrainable geometric graph neural network for antibody affinity maturation},
  author={Cai, Huiyu and Zhang, Zuobai and Wang, Mingkai and Zhong, Bozitao and Li, Quanxiao and Zhong, Yuxuan and Wu, Yanling and Ying, Tianlei and Tang, Jian},
  journal={Nature Communications},
  volume={15},
  pages={7785},
  year={2024},
  doi={10.1038/s41467-024-51563-8}
}
```

</details>

# Paper companion video

Simple accompanying video for the VPRO paper: white background, black text, title cards, no
narration. Built 2026-10-05 on `tueilsy-st-07`. Simulation clips play at 2x, hardware clips in
real time; hardware uses the front camera, simulation demos the frontview camera.

Output: `slides/paper_video.mp4` (1920x1080, 30 fps, H.264, ~3 min) and one `slides/slide<n>_*.mp4`
per slide. Everything shown is listed in `sources.yaml`; this file says where each piece came from
and how the clips were chosen.

## Pipeline

```
fetch (huggingface_hub snapshot_download, see "Sources")   -> raw/<repo>/           (git-ignored)
python cut.py all <raw/repo> <video_key> clips/...          -> clips/<...>/epNNN.mp4 (git-ignored)
python cut.py sheet <raw/repo> <clips_dir> contact_sheets/… -> screening sheets      (git-ignored)
python screen.py last|strip …                               -> more screening aids
python compose.py [slide numbers]                           -> slides/*.mp4 + slides/paper_video.mp4
```

`cut.py` turns LeRobot v3 datasets (episodes concatenated per chunk file, boundaries in
`meta/episodes/*.parquet`) into one H.264 file per episode. It must use `/usr/bin/ffmpeg`:
the conda ffmpeg on st-07 has no libx264, and OpenCV cannot decode the AV1 sources, so every
source is re-encoded once before compositing. `vidlib.py` holds the layout/text/ffmpeg helpers,
`compose.py` the six slide builders. Plain `python3` of the miniconda base env suffices
(opencv, numpy, pandas, pyarrow, pyyaml, Pillow); no GPU, no model code.

## Slides and sources

Tiles loop while a slide plays; a slide lasts as long as its longest tile (grid slides: until the
tile with the most footage has shown all its rollouts once). The speed tag sits bottom-right.

| # | Slide | Content | Source |
|---|---|---|---|
| 1 | Simulation results: LIBERO | top: UR5e, Kinova3, KUKA iiwa, Sawyer demos of libero_goal task 1 "put the bowl on the stove" (first episode of that task in each repo: 102 / 120 / 125 / 36); bottom: **placeholder** | HF `OliverHausdoerfer/libero_goal_{ur5e,kinova3,iiwa,sawyer}_additionalCams`, `observation.images.frontview_image`. Bottom row needs MN5 (see below). |
| 2 | Simulation results: MimicGen (no title card) | top: IIWA, Sawyer, UR5e, Kinova3 Stack_D0 demos (episode 0); bottom: Panda rollouts Stack_D0 ep 1/3/4 + Square_D0 ep 1, all successes | `vpro_mimicgen_eval/datasets/lerobot_frontside/demo_src_stack_task_D0_robot_<R>_*` (frontview); `vpro_mimicgen_eval/results/eval_videos/eval_43344870_mgxemb_run4_fullft_80k/` (Table I "ours" row, policy 43344870 @80k, agentview render, outcome in filename) |
| 4 | Hardware results | 2x3 grid; each tile cycles through three successful rollouts of its task | rollout recordings (HF `LearningFromVideo/rollout_eval-*`), episodes listed in `sources.yaml` |
| 5 | Human-robot transfer | (a) trajectory clips with the "sharp rendering" (warped) panel cropped out and relabelled; (b) one latent action applied to a robot and a human frame, laid out from scratch (panels cropped from the headline clips); (c) robot/human clip pairs with their latent heatmaps, laid out from scratch (blue frames and burnt-in titles cropped away); (d) the latent grid cropped from its clip, with explanatory text on the left | `figures/xemb_headline_trajectory_*.mp4`, `figures/xemb_headline_{up_left,left,evaltasks_away}.mp4`, `figures/xemb_realworld_transfer_evaltasks.mp4` (0–5 s and 5–10 s = examples 1 and 2), `figures/xemb_latent_grid_evaltasks.mp4`. The LAM renders themselves are the ones in those clips; regenerating them needs `experiments/make_xemb_videos.py` (GPU, `mg-latent` env). |
| 6 | Failure cases | 2x3 grid, one failed rollout per task with a one-line description | same recordings as slide 4 |

Slide 3 (human demo / robot deployment pairs) was built and then dropped on Oliver's request on
2026-10-05; the human-demo repos are still under `raw/` and `clips/hw/`, the builder is gone.

### Hardware rollout recordings used

Arm choice does not matter for the video (Oliver, 2026-10-05); these are the
`withvideo-frozen` recordings, and for the two novel tasks the `budget05-withvideo` ones.
`banana-black-bowl-*-nt-*` is the novel-background variant (task 6).

| task | repo (HF `LearningFromVideo/…`) |
|---|---|
| milk on plate | `rollout_eval-milk-pink-plate-43535703-withvideo-frozen-25eps_20260721_190611` |
| salt on plate | `rollout_eval-salt-pink-plate-43535703-withvideo-frozen-25eps_20260723_115245` |
| banana in box | `rollout_eval-banana-cardboard-43535703-withvideo-frozen-25eps_20260721_131408` |
| banana in bowl | `rollout_eval-banana-black-bowl-43535703-withvideo-frozen-25eps_20260723_175055` |
| push milk right | `rollout_eval-push-milk-right-44227729-nt-budget05-withvideo-frozen-25eps_20260821_133754` |
| banana in bowl, new background | `rollout_eval-banana-black-bowl-44227729-nt-budget05-withvideo-frozen-25eps_20260824_202623` |

Only `meta/` and `videos/observation.images.front/` were downloaded (~40 MB per repo).

### How successes and failures were picked

The recordings carry **no per-episode success label**, and none exists elsewhere. Picks were
made by eye from the front camera (last-frame grids via `screen.py last`, then frame strips of
the candidates): a pick-and-place counts as a success when the object rests inside/on the target
at the end; push-milk-right when the green milk carton has clearly moved right. The milk task
is the hardest call (20 % SR in the paper); episodes 2, 13 and 23 end with the carton on the pink
plate. Failures were chosen to be visually unambiguous (object dropped beside the target, wrong
target, missed grasp). These are cherry-picked illustrations, not a statistic.

## LIBERO Franka deployment (slide 1, bottom row)

Filled on 2026-10-05 evening once BSC's transfer node had GPFS back (compute was still down).
`fetch_libero_franka.sh transfer1.bsc.es` pulled episodes 0-3 of
`43613157_rollout_43582857_withvideo_lam1_policy5_nact10_libero_goal_t1_sqsh_render5/videos/libero_goal_1/`
from the render5 group of the paper's dual-view "ours" arm (policy 43582857, ckpt 070000) into
`clips/libero/franka/`. `recorded_successes.json` in `libero-stage-probe-20260904/out` marks those
episodes as successes. The mp4 containers claim 80 fps while the LIBERO env runs at 20, so
`sources.yaml` pins `fps: 20` for them.

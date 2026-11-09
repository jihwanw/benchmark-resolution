# Timeline of pre-registration and execution

| Stage | Pre-registration text written (date in the memo) | Launch record (file, timestamp) | Runs | Result file |
|---|---|---|---|---|
| Killing experiment | 2026-11-01 | `code/launch/instances_20261102_0011.txt` (2026-11-02 00:11 KST) | 100 + 2 parity | `results/killing_v1.json` |
| Main experiment | 2026-11-02 (update; primary estimand redefined after the killing experiment and before this launch) | `code/launch/instances_v2_20261102_0910.txt` (2026-11-02 09:10 KST) | 340 | `results/main_v2.json` |
| Optical-flow arm | 2026-11-03 | `code/launch/instances_flow_prep_20261103_1145.txt`, `instances_flow_20261103_1213.txt` (2026-11-03 11:45 / 12:13 KST) | 150 + 2 zero-shot | `results/flow_v1.json` |
| Qualitative display runs | — (figures only) | `code/launch/instances_qual_20261104.txt` (2026-11-04) | 2 + 2 + 2 | `results/qual/*.json` |

The pre-registration lived in the author's internal memo, which was not under version control; the stage texts are released verbatim
in `PREREGISTRATION_ko.md` with the dates written in them, and the launch files above, whose names carry their creation time, date the
start of each experiment independently. Three analysis decisions were made after the data were in and are disclosed in the manuscript
(Sec. 4): the reported estimator (REML → design-exact method of moments), the seed model (run-level shift added), and the interval
method (scene-resampling bootstrap → parametric bootstrap conditional on the scenes). The versions of the analysis code they replaced
are in `code/history/`, and the results they produced in `results/*_pre20261104.json`, `results/archive_round2a/`, and
`results/archive_round3/`.

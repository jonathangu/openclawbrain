# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-027`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3f260af2c7b68b1309e9a87df75f2e99f6d28d47bb3f82fdbd20cd787e51e3c0`
- fixture hash: `sha256-4a50ee1d4a23bf54584481d6c799516fa1f1a51aa4c19299da0f6a6b73848dff`
- score hash: `sha256-169af59d07c3266b7bd99cd10e00c10f5bf72c478c1de73768037f9de421edbb`
- bundle hash: `sha256-7580a87f8a863cfb4dea8a3e2f53f376a692effbb994afe8d4f2b68710ded0c6`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9b2a597464226db9617a3470772ef24fd543ab0477b7bbc0a0ad5adf41bc0dc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-428a05513707d778860a90f73bee2cde4b991cc91c0ab4c37356b1eba6681c7d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4e90184fff4db343369c6ea4ba08a545db50fa68337c79d5d2415b5773f1a177 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ee41b7903088d91be2784d7542a156474f9d005f331779227768abbe47530ed8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ff23279d | sha256-e5d875dbbb78e2d60b48f1c4a13b7d41e030812b17c005c094a4fdb0349d7e65 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ff23279d | sha256-1b1d1f48a78bf0c9c9249801faba3cd4d7dfc139469b0320f0876699f7e7a4df |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ff23279d | sha256-e5d875dbbb78e2d60b48f1c4a13b7d41e030812b17c005c094a4fdb0349d7e65 |

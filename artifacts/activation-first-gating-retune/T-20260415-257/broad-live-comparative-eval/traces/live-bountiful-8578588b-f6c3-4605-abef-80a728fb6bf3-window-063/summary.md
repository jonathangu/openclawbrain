# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-063`
- winner mode: `graph_prior_only`
- trace hash: `sha256-de0740d5fcaea5454096d093094f34c04b3f0e1916a63c31d443d2fb518007d1`
- fixture hash: `sha256-fe0c7d701266203a3973486da92812ab1d527b01c29d86dc3ebbae41ae89dfcf`
- score hash: `sha256-bacadc7181fc7fbab79a37e1162043092d683fce6a7f3d8476296d0ab01ffee8`
- bundle hash: `sha256-20aa2e8563b14465209416f2a28162a3f51c53edbc66414dc6baafb59d6a8f18`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aec7ec8f6799a0ee01f6b4130aafafed76ac2a835827ba3dac19bfdac983b407 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a0e08b2ce3dc1297bcae881d5be9a99bbdbc01c6e3840c8c44fd2a839e47e617 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-0b82a95ca6e19da8064caaf7534a13e123159b69006e102ab94d48e11daf1d3b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-73c8eb7b7b93a497ffd1056da010a1769f3b66432123b6c4a020964cf0cffafa |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7fdb36ad | sha256-625eedada62c070c44931cd63d5a82a8e0932c22fe47f126439de70f1841077c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7fdb36ad | sha256-2aefa1020de369f7cb399dd5634da2868c4e1e3526e9df756a3bee7eb96fc1dd |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7fdb36ad | sha256-625eedada62c070c44931cd63d5a82a8e0932c22fe47f126439de70f1841077c |

# Recorded Session Replay Proof Bundle

- trace id: `live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0f24d3812e038d9d2b67d9309de9db96cd24c2faefb0b5dd93caf569b3c1d1f`
- fixture hash: `sha256-6b6c634b067ee2b84c6981ae8fc0d6c41efb6194e1723d88dd7d0087036cd1ac`
- score hash: `sha256-9b41f1670fdbbbda802ad636c667585ff64b1b2c35ff39582ceaee32a9b28554`
- bundle hash: `sha256-c9604b99739bae8e831d13a978178e3c89012d35da02502ed9418c5d91a82e76`

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
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0ea26975a8365e08832501930a2890706222216fe363c833adbd0065a774a3f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7ad8f9f8d6b9505fb30afcb619253bd686b6424c6642cc37ed1ce7d2d4d50bb3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0c112fa610b8ec20f040ca335454f1912ce430a1fbdc074081f9b26a388dddaa |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-50df4a76a6def5b338949ea4584339a1431f0bf5ac69ce48000ad59fd713c802 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d6e79118 | sha256-017a536c34f09d0388994f0afba81d20fdebe3f1d3d71497b38f540c497f9a8f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d6e79118 | sha256-30fb4f30a026778e4bf4078dada0c97598126a701b7ff9f32f03c752cad24b1e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d77b6cb3 | sha256-e9dcaf866c9bc21327056bbcdf78474862a03144f862515f16b1b3a3ea4e7faa |

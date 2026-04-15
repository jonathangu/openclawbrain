# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-11f89d0770e58c74a32e0ac08329b409440ac220cb647ec446567aadc15cbdd6`
- fixture hash: `sha256-7793c2d77fac055a1c7c47c9d026a76a01511a45ccb17bbe5db49943de3d0ea4`
- score hash: `sha256-654fa5dccbafe4d9db4d9163ed65244c1dc173153eb78ebff88804e0fb80c062`
- bundle hash: `sha256-dfb8fe997809abff59067ff96b4be50c57865a8d5b71626cfcecd0e914e1d449`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3297dca5d83084645cc80493377a366cf545c5142415159b972c4f8430720ab7 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-82c33d9a7d317adc2edfa01414b94ea122fbfeb5de13431190d2208bbe87a115 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7a44ddad5f52e31b271192a260bbc775a1686ff57d7c354df134ac19ee8c8220 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-419448cb5993a0c1da321b55202979ffd66c3f786af85a7d7b9e06895f8b1151 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3702bdfa | sha256-7a50e80d50526557aedc5a2124b16847c4915cee2f0949563add6ec9b24f5008 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-3702bdfa | sha256-257a8b6eced5622b83ecff1455a5d6f95d48f44bb4dd7fdb9846dd6848bcb3d1 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-3702bdfa | sha256-7a50e80d50526557aedc5a2124b16847c4915cee2f0949563add6ec9b24f5008 |

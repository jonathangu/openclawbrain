# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8b83288abc1a5c66a218574e9a089abcfea75ee1de4f5813fd07c339a4e34fa2`
- fixture hash: `sha256-d84bdb541f6a2d5c8236abca3a843aa21a0e1c20f003d0fc5eb1d79b307b698e`
- score hash: `sha256-c4d950898ac985859f05a449e51b2934999d2e94a376b7c8e9489bf14735d8ee`
- bundle hash: `sha256-5a73efef3cc2eb0b26a34b0e7b892d717300cc27569aa5d3596c5f90dde55605`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7ad0dcf523c4d76bf7e5aa9a9c949e660e04aa89d0cc57603f9d8d3b2165caa4 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3d06367e0913db3c64ec205d516859844ce20f2363974bcc7d171e2c64d2f857 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-258b53e26cca8fb682f428325c0f57dfe9c40a34c35eb6e8b99da5fc57b8d5ed |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-d03d4098daad5172b54a412711af1dd71e807302270b378ac70636407ad4d5cd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-268ea122 | sha256-9bd5b92c9032bb3e34973b90e45a3edf8e981fbc97153be1a3995deceb408f25 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-268ea122 | sha256-4a360ec4b2732e1704ac156c788cefc821b69b8d74814dcaf3fc5256502a09f9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cbe6d091 | sha256-d2f5a0c91bffa661139f19cded8ce0d6142958db7b6162e7bb1da5fc264d3594 |

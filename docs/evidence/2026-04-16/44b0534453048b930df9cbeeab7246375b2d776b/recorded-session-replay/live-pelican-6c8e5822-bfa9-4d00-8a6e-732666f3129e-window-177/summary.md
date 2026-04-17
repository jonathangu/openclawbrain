# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22c0c5cfe30f6528627aae6b3b1ce6c55137840c4388f7d03d5ba0c64043e114`
- fixture hash: `sha256-883333e2877ee56be18afd0bdb26f3a044eab5df448e40bf59cfd947e2e070a7`
- score hash: `sha256-ee0f208d062a1b8923c58eec03045ebc34b1773ce7cf9ed0b255e952f580e465`
- bundle hash: `sha256-a5904b0de85325920070e79a3c361c5509d0220c67d63457f5b1b4821241ce44`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bfacda8f5501f5e4f01bbebcdaf7a5c0e18d211755bb5803d41f576de0d46bba |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bd5c6566721be63e90628c3068afe437ecfbd085a25f1b945c90c59812ab51fd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-df32436e8ce718a76630a972552292e9073ecacf505f94e3b92ea3511256e0d6 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-9211e89909677b2b38e1d04e5468761ee52dd68d3e1a3663fc2b0b05e7224f13 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b52270db | sha256-03929820aabc35c9065d0fc8ffad3663479f2ad9ee15965a1bc18cdaa0681a00 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b52270db | sha256-fd1a43b6de0a4ab2f25267ee000cf3ca653626251ccfd38470a22ae0054604fd |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a6f8b18c | sha256-b6e738863c4b82a76badbd540d34a89382df6619aa7983ee6d4827d07de819d3 |

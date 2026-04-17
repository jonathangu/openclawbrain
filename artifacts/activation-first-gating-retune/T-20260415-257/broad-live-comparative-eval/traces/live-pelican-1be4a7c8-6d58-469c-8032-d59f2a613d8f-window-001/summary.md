# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9c0b90297f99ade602878feaa8cfde6e3a19db0e47440bfe22629154903dab61`
- fixture hash: `sha256-1baf21d3d9b73bfb53336d6a81b7f65e4d6e7e9fb603fe4e8af018eaeb0d47ef`
- score hash: `sha256-6b29f372c3d8512f6f1ebe1ba0f1133e6b89596d68398ccfa2c3cf640e638563`
- bundle hash: `sha256-b0babfb72920c39fb99ed2c6a2b2d531761459a987e84a32626b8ddf6a06494a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-27b2a69331fb76743637a0a59a8c052316c43dae2eb924cfbe90678912704fb5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b9411e2c487c8d8b88f2d9ec4b715b0918da4dfa86db6b5f765088c0091b0d23 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b0a5152eb881c764576f452cc86f242701cc5ac07dc8f11d5e57b9924df68344 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-fe7855279e7714b4c5a1836e1c91e983ef88096bcf676550bd19a05dadeab931 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f0f3ee8a | sha256-644841ba4f5e9dca3886a9628bb4e748afc92c4100b80e9efc19244d52105290 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f0f3ee8a | sha256-644841ba4f5e9dca3886a9628bb4e748afc92c4100b80e9efc19244d52105290 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ba1e07dd | sha256-41dfd6aa1890616aae9e9acb21359211cffeca417d6cc5b0fb9d24216395ff09 |

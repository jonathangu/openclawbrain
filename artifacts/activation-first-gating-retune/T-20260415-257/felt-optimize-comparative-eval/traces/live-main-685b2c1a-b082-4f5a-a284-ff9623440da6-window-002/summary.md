# Recorded Session Replay Proof Bundle

- trace id: `live-main-685b2c1a-b082-4f5a-a284-ff9623440da6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6e4f37e5b65aeaf586e4d2a4e71ad3cdd999df7cfb455f741dd0b11a1a3ee8f9`
- fixture hash: `sha256-8f110e8a4894d421efbb2427ce24dd5ba84d98a2490639e91780761dd48a619e`
- score hash: `sha256-5a25387c3e595ff56c929c26d5a9a906961b951c472b095a4cdcc753c8729920`
- bundle hash: `sha256-8dd111a37d0a376d461e0c45000c83d8ccdfd7619c83b6174c273e315f20d410`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb1995b9d3380942336af495a709abdc9277059dafdab742d11d02fd9c054a90 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-9be8d07fe4b617da0859299b1e73de77b3147edd5549b07d57c6c58308bf8f19 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-df60b0ad2e00ffcd3d1e4001e4671c557e6bf085e8e0bd69d4f44797d74a383d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2c506a68f5bbd1a7a1074af83dcbf98e9277a1776262eabc45e1b260c657384c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d5b7df56 | sha256-ea84ebc01b0751e1c786420097e013c3ebd2a1a521dc78e177d8cd4f28db985e |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d5b7df56 | sha256-630b616d60ba39cc7c8735be8d95cf853427e9531818a40ef4d119bfcb06053a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7e807075 | sha256-6cdcfa5a74e91fcfbdb5268262ec22ac73557afcc7e82e380d29e5f514f07be2 |

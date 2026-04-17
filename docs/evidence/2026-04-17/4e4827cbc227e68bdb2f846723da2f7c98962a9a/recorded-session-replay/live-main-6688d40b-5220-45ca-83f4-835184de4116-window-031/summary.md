# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98ce4509785da1d3e9688496a53303f79675442a91eaedda79bdab30b5e6b8cc`
- fixture hash: `sha256-ab905612bd3cc43deb68d413a855b981990f021bcff6e0685761c3af602b59e1`
- score hash: `sha256-839d3b575b58120cbb5203dae56cfd4300e928c48605bfb63c45f0e8e00320d6`
- bundle hash: `sha256-9a36a1e007001cb385163b16c8454cf65ed3f1c759bb707a6c4cd23e3f31fd50`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-16539ac70abd2ef9678c6c7835bb8d35322c600e9de7b2b4d16217df707851eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-43d1ef3d7f7b1c9b741e3ba89bad7c68af99f1b5c590a32241fb1f9d839861b3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-28d658ddcc82e16d38c1d4320d2bec9f5b06402cb29fd65b29dac2735b30b5b0 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-81e1ae4473555af2871e1d293f428c9bdfd4ec5b9b65b968a13730ba8827651d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0409e2ba | sha256-20910f365ba9ac8bbf1d453a65b09742f82d70e3fbe714b4132763798043f7a4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0409e2ba | sha256-4e920b83bdc2bb0d6e07b589aaacc8afafd93aa8ae8dfbcec28d19d2a0be8df4 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-72bb6077 | sha256-3419f7bb08baebebd005279537e956cf4bffa2e2fb8f1fe910c066a42345da03 |

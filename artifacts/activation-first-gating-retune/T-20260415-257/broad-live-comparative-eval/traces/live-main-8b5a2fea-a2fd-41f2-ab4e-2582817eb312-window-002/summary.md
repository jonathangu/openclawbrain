# Recorded Session Replay Proof Bundle

- trace id: `live-main-8b5a2fea-a2fd-41f2-ab4e-2582817eb312-window-002`
- winner mode: `vector_only`
- trace hash: `sha256-e0e56ffd1c26d20085e7a9eb3248f58dfab8c43d92d6bc35e804da203ef4f7d9`
- fixture hash: `sha256-e4b8d39277cb985d3e9ee559f9e373775182720bfc10b6d9350141f9c5016460`
- score hash: `sha256-efba7dd62607360cd2a86a742437f944ae8233ffe539d8f96e4a97a95d0b3f82`
- bundle hash: `sha256-79dacc978ba2388348931adcfbfa93b2e4a5aa9ddcce2eae0edc258f4190af26`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 80 |
| 2 | graph_prior_only | 40 |
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
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0bdf6c0bfdc77dfb35df2ddd80b080b8e6bbd2f8f1020fedbea4770e769e1c72 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-b7be5ab29c468925947694f05c898cb978d64884aa78ca88a89ebe0cba52ca23 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c0c6a2e0f2da0d96b02600cb47e2698fcc14bc1fa1bab42ce0a1ad6514601b71 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-07fff4d32d1108bfef4e9935b330cc3f804a949652712243c97c4520a87c9368 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-e959e665 | sha256-761f61fa99ec55fcb1a62ab07eb8bc29847506f9ab2d8047a6326fb7ebd4b670 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e959e665 | sha256-3870b8cb4afbaecab67443217eafbc5e1e68bc569301e5d233cbce4c4c1baf0d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4f84ad1e | sha256-492a2c02cfe16806aeed73b730a77dc3e5492513a9b2151654336eb96749971a |

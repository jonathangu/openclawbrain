# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92a53a83b75391e6ea2e19694e75cc46987c1fd7f2482c72c3850eb3ee758d5b`
- fixture hash: `sha256-a7a70c06edd57e7fef42061ce44261270b10f99213ced50cea189f13c03e8e7a`
- score hash: `sha256-913e872b6ac88286eb8069e199f94c37129548b17f2526ae26be1169f85a725f`
- bundle hash: `sha256-8ee1946cf57e9d87b79f680f1e2b26999b304b5b0fa9b36cebf493ed198eadd8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-467d90a6c748c6c78cf3c7ceb933156139020979bf5f7ad7e3a8103479da429a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-73d7202da5c730736984af60f860e5e7c61ff9bfcab09ad180f9d2f22833244b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0ae1c54c9aed31585edf6e0e3e32e1504736f56e9dcd60587d3ba02e76dba1d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-fa198f70eb98c05c866e56be7b29d0d40a343719db2ecd9b005605e2e6d150da |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f29e60fa | sha256-abe44db8ad6fdf525b630981fecdf38c0f1be5b9b9f39d53096f2a4cca9f9718 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f29e60fa | sha256-3146e9f4c3b5c78dc5b991aedd307cfd61c4b3fc831b66884c8b61a9b8a06ee8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-1002750f | sha256-f07549f2dfaf0c87456dc56d8aba2ee0ea07e6873664d95a6dc8baaec6751a2a |

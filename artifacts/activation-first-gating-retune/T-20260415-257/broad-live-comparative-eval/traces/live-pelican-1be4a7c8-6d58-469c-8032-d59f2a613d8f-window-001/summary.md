# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9c0b90297f99ade602878feaa8cfde6e3a19db0e47440bfe22629154903dab61`
- fixture hash: `sha256-1baf21d3d9b73bfb53336d6a81b7f65e4d6e7e9fb603fe4e8af018eaeb0d47ef`
- score hash: `sha256-d1f7bd88ac34bfad636ef6771b77c2aa34504351285649d49d4cf9ab602918f2`
- bundle hash: `sha256-5b6ab8fd26be8ba1eb02e8eae2ab1bc188a78952a11a81aea92559c7d55a6f27`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-10bb7e7ddca3675f44fe3ef3b86f35baa1b32d965a28a72bb9b8f2f84e7d7285 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b380b516544320def55ee3c166788be91429102416dc0cf73e648b636ea2d6aa |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-bde6a2d6afa376b5a413f7c9788b2e53b5aa25d2fef1cdb9efa220665b2340ee |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-754488a0 | sha256-61d08dda9ee8a9175ce3b11ae118f72972dcb43d068b9de247d583c04cdfddc0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-754488a0 | sha256-61d08dda9ee8a9175ce3b11ae118f72972dcb43d068b9de247d583c04cdfddc0 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3e6ea1f3 | sha256-6dc244c1c1a0feaace52144603e4d7663129a8e483a3b1c9b974b5245d02d2b2 |

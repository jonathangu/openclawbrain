# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d5b2e9dc9e67decfaf3c661978d40b3965c717607588db6b26b950194e4e66bc`
- fixture hash: `sha256-05d4de9ab3e3c70047bcf0e08acaa0f5e5762d96334a591c78e4a27669a8787c`
- score hash: `sha256-2e29d31eb636f038c17520f733c61e74cb1c0c86ed4bfcbf54928c6afa4b2183`
- bundle hash: `sha256-d761a07ea01496a211e645d481434674ce78ea708636c2a9c20789b896489737`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-044b12081781ee7b9e9814feab1eb91fdf156b393d98255c7373c2abeeff9d8d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-42a885c575b9cfa99ade6c3c790201d131540e12e0a79ca17b4a420663ba8e3c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e20e638ee45eff093ff79500a7ede57a570302a1684a178e24d3b63d39613a9 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-7e043f48b70837de9204780ae5f6bf8172547ba10dabb82fbdab6cb0dbe0c3f6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fa100763 | sha256-6e0ec1c11656e7346e00a784768b137d0d8ec5f66764a3c40c90bb0013ff3407 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fa100763 | sha256-6e0ec1c11656e7346e00a784768b137d0d8ec5f66764a3c40c90bb0013ff3407 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-8cf614a2 | sha256-03cc624ed6497bbcdf101758884be836775ec7ecdd260b6b03876377ad9491d4 |

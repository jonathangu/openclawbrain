# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80d3477f10050166bf08a79ad115cc0623875c77edbf3489b3449d2e77618193`
- fixture hash: `sha256-550621052f6f6f4dedd32e7dd1966df3bdae13f0842e74ffdcfed29aa308dfb9`
- score hash: `sha256-d74aabfc0d5660e2fe5e95a656dfb1affb4c3346c367f2a98c4a637e012ede48`
- bundle hash: `sha256-0e727c7f4e7f4f3249ab8384ac788fd9c73ae7dfe6b646e481ed454aedb434ca`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b053d8d2940510defb6223852e2cebf21b6ccd631a727caf5859e48b2c5a0baf |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c4f4cb129ec5897a35e9937557e9efed8f85a6e3f69df5974aa6c5adcb8f9d0a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b245fe67b2603d9e1fcc1e77713838c34f9c94865f2c43aedba698c2a27e2b0b |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c811cb2adefd07ab4cf40b1d7d05ff8832d7e4ca8c56b341ba771b1dee665de6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d3a09742 | sha256-72f0efd1d71310b32bf3e87ec86e5f740bcbbe3c363a41ec38f636e7f0cb94d1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d3a09742 | sha256-a5832486a7e3b613b29b0a67b476ec487f150a466610e426832554a58b83a1e9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c8c4eb05 | sha256-4064832911094e9a7c0891d52c36c504246478d472ad7719e09424b76b3bd672 |

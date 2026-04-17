# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-150`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5ab4245c094a4283a2fe623f159ada638f3a2335ec988d52906db167e4a412cf`
- fixture hash: `sha256-200539ec6ee07f9053b46fde1430980f62e83407874931f79115b5f9bd8b8337`
- score hash: `sha256-983d03e338cbf036286beff1e12e22bdd6cb85852c1edf386ebda381ae4a3c88`
- bundle hash: `sha256-03be0b9d4d6d2a3b68afb5deabda6de21f263ddddd6cfee3a4897faa0970b32b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd1ac429de4ad281ade866e710d7bcaf6542300ac52809bbbdfe005490548973 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2e4476aea792ad2cc5dd9a95f9d4dd1930b55a1d266a91019d5565be0a49ba20 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-96b7abb117b17d598d8fe8dcfc85dd268ad0ee87db444183296e9ce729d06c5a |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ef07038eece1eb4ca57f83d10cd281757b263fc05f6683036484d1b0cbdf4433 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b389b09c | sha256-404eb004cdf20c1a0f3d0b3bdb9cf197fb338a6a3b20a2c16310ea0cdd6d0c08 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b389b09c | sha256-0593523fe2e5b46d9604b1767e6c8ab5d3493f1c7253f2431a35c59cadeb1644 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d0f32ced | sha256-ec83255565ed4b803395a96b81150c36f99a558cff2c4b9493d181cb00ae08bc |

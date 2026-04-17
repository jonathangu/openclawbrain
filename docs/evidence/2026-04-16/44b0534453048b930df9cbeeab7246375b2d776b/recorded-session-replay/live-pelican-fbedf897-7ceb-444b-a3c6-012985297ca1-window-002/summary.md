# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4e6bcca0889e112786ccd30d6dc08d693afeab2955ee3e21db9f09dfe3094e0f`
- fixture hash: `sha256-14ff100a8ccae36fc1c57494dcba2b6e1338cfd708e5c890121212b4f7b539d1`
- score hash: `sha256-670d5372e3cb153019c459a333aaf810b70670dc0d06a8c3fa5875f007362e3b`
- bundle hash: `sha256-dc482ad9c2d74b50fb6e60ed097a8953548ab752d3d7a8c67e5bb13a4e4f798a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86a96bb1f89c4f625498603269cc86fe2157c50e9372e11582b94c39873a6510 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-74bdfd2378d9b7ffade417718e7f65d54503dfc52697ed8e4f09e2a4bc11cdc4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-27e949a6a461371e73002c301d432f0e1cbb2c228258e976afae5953c279f426 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-09929f31e9448d1983eccac5be0ff775b8f9488eb57eef8e0a663a8bd7493fd2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0680ab04 | sha256-6d119adca2f2cb113d69d76b87d88b7f774669911d42d0c8b52375e9797f6ecb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0680ab04 | sha256-35414928d802981d04ccf3f0edb5a7114c6246237b5d9b53a007883b9643f5d7 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-92ad0079 | sha256-a8220504a03dd2be5714e98f964d97941b42878547b2406636cf74f28d3485f7 |

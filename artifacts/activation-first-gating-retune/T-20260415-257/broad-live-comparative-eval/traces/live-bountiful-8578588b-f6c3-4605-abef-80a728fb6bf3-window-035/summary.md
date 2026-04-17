# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-36169ed7c6b7573ef644e5e670a60e6f7c1c993fe52a7f13686f76bf635cc43f`
- fixture hash: `sha256-16d6e6092a3d5f8800f19820b3a256e739ead32b4381f2896d9aeccf372e3bde`
- score hash: `sha256-048a503cc8249666d26b66fae37c263894881503e06aa68068ae0b621801be4e`
- bundle hash: `sha256-b831982c4822a32965e20b69ef9e3614fa3c1850858e6541d4ea161a724fb0b9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7c9a32ff2de979b024c8bfc14bcb8fa72199e8333c117ab27753dd3626b13edb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-75377e504fe08273b5779bdd8bed4595c233aad7d74d09a01a061e699561e228 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5a351e431c9a4701cac42cbca572775ab3b2ce20489f7a092c476d770bd5e5ab |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b568963ca26c9d9c7f4e432b11838024848ddf18b75872c51ca142b29084ee45 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9a829d40 | sha256-f328783305eab1a005326ef2b333ef9231b891c334fa0bac8c9be15df5a6de60 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9a829d40 | sha256-12ffe7b8255b106d7ca035dd6f5dca7a12b988f92ce0f737c97c0e5bacf4f983 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e2266ae1 | sha256-589e9f23aa8d6f00a6cc9ee7abff58c60de87b5d979ba1efe7cc2250c7c8c7e4 |

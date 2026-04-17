# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-32e0b4ec2c1ecbf5a44b66dab5340f30730d05ccd8fc6dea8e459b03d93bb729`
- fixture hash: `sha256-cd231e74dab2c7ac691e39a4ea475c769c350fe4115dc674162e2af0c0f3148d`
- score hash: `sha256-0e74741983c66a61adeb5550c1e65748d667e4d61d0191ae23cf67d21c7cb8c4`
- bundle hash: `sha256-ece196fb29ce123cd0f4432ae8a98ee96a5ef0c285a65ff7ef5a9655ae20f2fe`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | vector_only | 70 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-eafd155fdfa2fbd8e1c5739855382bb4aee55ae760f037b37bc2cd66c8f2b4cb |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-4065a0e311a9a1384e6369bcd8601d101e6f8bdf0f04da7f7284e2ccd5903b60 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-9ca4d51ad7003f6566787f7ecfd6865563ad1d547d0d1d7313113773d9942dc2 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c8dadfa945e75ac997208d797109666d0a00e0f5bc840c1eda48e487424cb56a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-ef645ab3 | sha256-ab25fa8c1a57db7d868cde19eaafe75198c200e08122e21ac2b39c824c7adbbb |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-ef645ab3 | sha256-f49b4ae0dfa774236052f860c4a7d1a399596b4c695b37dadbce531de1b4b432 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-f65fb3ae | sha256-45672a812cabeff1062ede22a18f40ca960b2701459cd81a9613540d91c86ad1 |

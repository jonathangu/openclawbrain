# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7d50a8bfbe12d6ec52d00a65d5c5309711fc92d4bd65677275533c95c1fbb9f9`
- fixture hash: `sha256-486866769a6220eac0c25d8477d823ddd1d78a29159bb789869bb12cfb7c0a16`
- score hash: `sha256-293e87c0c409d29699e5bc11a2ac4326d8605b2a9f8cb30d925b1cc37fba325d`
- bundle hash: `sha256-1f97ee52633e6d46543aa187a0d19c2f69a97272476be70aa0c1849f3190ae66`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-030b3c11ef3b6ff56c24da96c3a7b6b56306fdfbd30d56345e3f6aeb18dc6984 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c59e682959e47c3df105ac18edfb4962a98d3847befb2fd471b916f0d382df31 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d84cc6539900ac0a3a549c60da718fbc0d1f9d1415e66d9185f9d9fa268461fa |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-0dd891a1787f5f08af37ffab11532f8e1d09c241c3f9f7656d36fce5721c9afb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-675a0b34 | sha256-cf0929d24261ac9b7065aea8fecdc4b1de287879fb1abf455c59ba6ad1904cbc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-675a0b34 | sha256-16d970abf38214861d3666c36283088207c66363aeaf5693fabeefa266a6ee55 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cfc67b85 | sha256-1fc600b1d307de6a1ece42fac946613f963fda5956f12edbe2985ebb178c8da6 |

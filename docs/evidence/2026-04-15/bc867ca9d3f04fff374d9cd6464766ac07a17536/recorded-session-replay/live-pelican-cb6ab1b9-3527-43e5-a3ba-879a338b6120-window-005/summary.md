# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80f4bd70b8f229336838d17a92921bfca64745f162a9177679361b11e355a256`
- fixture hash: `sha256-1a25c630a19d83ff4b3784d9a97b879e228a965826872ffe6bcc2e6453fbac5a`
- score hash: `sha256-b722a71b0711e05c4658e361b5955886ff91728b732fdcfe8b7b7a1f0ed80372`
- bundle hash: `sha256-bc5170932c6b1483a582fc0db94360bcfdce859bbe240a18d691dbf3a27eb805`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94ade431a5c986254405c71afb4d4071b897c04f7cbfe57133fcaa9500ad06d1 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-750aec9fbda2a4b9f7e52cf3294e54b4a54661b17ce3031162a730fd6dba169e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-959d79b8d2d5a8655c5ef579f7a3c4581a17cce94176b2fd03cd03d36c6c7df8 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-39237a95a30354abdf54e3f9cd0eb67436ead98caa8942abcb95039c583edeb3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c81b624c | sha256-a1c4e6ceb651cabbbd0afbff4c24cc291439a89be9c8f7da027bc9a45196c9cb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c81b624c | sha256-17b6339ce07a330b6e239956ab139522ed2b1cfa4a70a90f93a318c86ba03064 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-c81b624c | sha256-a1c4e6ceb651cabbbd0afbff4c24cc291439a89be9c8f7da027bc9a45196c9cb |

# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80f4bd70b8f229336838d17a92921bfca64745f162a9177679361b11e355a256`
- fixture hash: `sha256-1a25c630a19d83ff4b3784d9a97b879e228a965826872ffe6bcc2e6453fbac5a`
- score hash: `sha256-f0fa27aab1913b8053f11a38a9bf3c872d5f381250ae1ea0c9a04ea1a70fbaf4`
- bundle hash: `sha256-c9cf1104ceb05c52ab787672e38abb4ad30f052b28adf04ed454febad7dbbafa`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94ade431a5c986254405c71afb4d4071b897c04f7cbfe57133fcaa9500ad06d1 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29dc0d6c8e81c29d5e12c1a14420528950fdbaaeba645756265d85e5a73887fa |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9a00568171515d43cb16b441d30085f0ee54cd928e74e825b76191255e103fb1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-60435629ec9a80df1f11598aacd41ca1b5c011f3e4a3b6e7768d7d3cab7e80cc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-97c1cbae | sha256-47e9c113e90b588d1405c7fcf7b4d9037a9940a06fee26cc23e6ff2995aa6a68 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-97c1cbae | sha256-dde9b87ec8dc4d75612458d6876542552855f3755bf8b126c9e7763ea12522ff |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c79b8601 | sha256-26c8de6408f47009a2f8e95409614034c4deb0bd7e595c6708f373e1b1ab3bb9 |

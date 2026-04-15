# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4ecb7ad01ebf51dd3aeb7754e784e70eea9f4067a9392ad81778aff88de83b03`
- fixture hash: `sha256-ad7aadbe694390cc07af980435b05bd2086d5294c79bda5f4f75ff348a4a3b75`
- score hash: `sha256-d984a035699d98e69adbbefdbd5abfeff86f310a5148e0e4d790a2e477d57bc6`
- bundle hash: `sha256-9d57571959f36f85144f50c3918e714f15e3ee0ca2eea4efb961b3560fc3c9e3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8a8a0e6d7dd1a7545143681fd0202299acfcab2ad5ce85ed5e5cddd516c7f67 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cf16e1f46a4369db1b99605964cef28f662621313d1aee219a30234942d6d99d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-50706d7eaa053e81321b025166ffcffca6a04e306a78bc95da3527acb321793b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a22c1e858f28ec1e95da2583ecf2fb28ffb4bb17eba8f83b17865d246d402ce4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-08bde710 | sha256-7122df0e5f692a826dd8bffb45e4d262039f36eebd2190db2eac69eb356fab6a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-08bde710 | sha256-496f6f868fa36e5ca0f31dbf5af76338987ec486f21eb424b5f5ecd4f6b2a55c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-08bde710 | sha256-7122df0e5f692a826dd8bffb45e4d262039f36eebd2190db2eac69eb356fab6a |

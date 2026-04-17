# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6aeb45f46257078a73e31a3ca01fc811e5a3a9b2828328d1595fb41ae1cb1b87`
- fixture hash: `sha256-b90901422fe4620c22145acdd76fedd90d08a07ca2636957ff33166af8db8c6b`
- score hash: `sha256-1f6dd96791d5a6c214c0855f412df53e88b0a17d8c99cb84318ac0ad1cfe5064`
- bundle hash: `sha256-f33838b0ccdabf6cb058b41a2e48d8c7b8c37c23a27775d3143befa2dc8640b3`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f9c00df70ff9e588e665c6961063a6f0105a883c9e9bd2b1d2f815eef1057f7d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-94e3c4a23736aa68b4e696d3c00102ed5c0fd3dd48c4399eb08e3d070cfb32b0 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c8f7d2b34bd1650b63748a3fe90d395035517845e3f64045423d77c1c71ddf03 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-4e026aceb0a2efa6d2119cb7f3bcca128482a5bd1f5088a114fe1fbb37737e69 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d0d56000 | sha256-7b35336de239b80b97de2a13f83b428c693a5a94f69ed0c57ee1c27c4f3b064b |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d0d56000 | sha256-2fad059c38308e3ad955ebc223adf71a599c80bd8bfba6514d65dd38c64c1b8c |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-26be8b4d | sha256-38e526cdf10e7cdf0d656c82a4d6bff0056d643f59d9eed77e8056397d7eef43 |

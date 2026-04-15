# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-188`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2286f1962f858995a9d11d68ccd4ff744be8c0925ede8b9595870bdf0f8216d1`
- fixture hash: `sha256-7131fce3dd7f89b87927812976c9719dadea253e34115e7f37e0887827e9427e`
- score hash: `sha256-9494541477b0d0361db5a3579123a9ec14f87a565213ce82af3ce50ec1210411`
- bundle hash: `sha256-2010dc5270876b0105d04eb6fdc6d9d0062c6edc39fd0c22ee30976343b9aaaf`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4cce362e58248df18caee133abbb86ea37c7c8cc312d9027b572d5a719da7a87 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c3c5bc5d8bec8f537236207f510fc6c5ddfaead2e163d5cd2f53b88f71ad5791 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ddb25733641152fb63d902189be62a91b8b8dad5b7d8561da4fff6ee99979bd6 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ae1d743cb652f7ba8c413de7ee2316a63667bde7950729406f0f4fc383a73f13 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9142e184 | sha256-13d6fd9a807dbcfe308c61984b87ece6791cb417f4104784142994b17823b1c7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9142e184 | sha256-925006c3659ed74023a478ba9fdeac92aa1dfb1bab488dce808bc170537b63fb |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9142e184 | sha256-13d6fd9a807dbcfe308c61984b87ece6791cb417f4104784142994b17823b1c7 |

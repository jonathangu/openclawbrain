# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174`
- winner mode: `graph_prior_only`
- trace hash: `sha256-681abae2fa5f82c72e9292e394b227021bc61d412148159906a6b997f617cca5`
- fixture hash: `sha256-c46f6cfeb0331761c1d2bb543d4b028a9a876c69162435d955a285bd82156828`
- score hash: `sha256-71ada67fbb7c6d5a68e86deeaada287dcf906c78da5c19b1810fb12dd2af3251`
- bundle hash: `sha256-f0b82c1300b18509fc9a81134e5c4f73d81f143ea08c1149f00c23a2592372e4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-461e5f784d6e942b4cdd1338a01f757f830996458c9f4abe17a0effaceafc63b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c625092adfab1b9c1c9133f1974f1ce0e248b604a76540ec03b5069ad7673ec4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b9feabbbd7f81e732d6f74dcd6de27fa24ce88a0a735890af4808ada7576fae6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5ee13ccdf13cfd7a70658b1dccb709d641de3752797f5d8006be2df4c7739358 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a2c940e9 | sha256-2301ea8ca31834773219ac1361aef081dafc5cb3bfedf735d6f11a9dbd8d71ff |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a2c940e9 | sha256-e0f4514f4433537da280d709821a395a07051f611db40d6be352c8ef4fabf2fd |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-608c1066 | sha256-c9b999892ca916302dc7521aac2dbaf8a0d11cd29ff0f0319acdb7f0e85da34e |

# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d380c2fd8773059a5893ffff2d380e86ca0f972a4140732a5832ea7865e5c2a`
- fixture hash: `sha256-55f386f545922fe7856a581e64b7fa651b1de1ce7956a55af05d3b2bdc86946b`
- score hash: `sha256-e10337e2f9407aa9fb0bb3035ecaf8de2bb9661aac9495fcc22204c2b6ffae12`
- bundle hash: `sha256-fa7f57fb793fa64a046bc587f0de3a8f472cbb9a884394572d07e8dd88bc9a01`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-63284d4d64db3291399ac8e17a28a524a22240af2c68e8497ef443766b42c4ca |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8bec0182355d21d6e74eb653798b4e3ef8741beeaeb2dd48ec5b1f41ef2d68a4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ef62377de6b7bf95da921f6f3ed780ca9b9fca9ff82f80cf2927c082242ff570 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d59cea2ec2810f7b6f166554a7937e8f928a55f57293f50ca8b1ced4dcfc5b9f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-621169ed | sha256-ff1f70e6cf97bd0766b9bfb9a193e4c4a53cee516cdeabe1615b7786bcecef5b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-621169ed | sha256-5841f73828c25c06da1b085ff904e760cd46a0c7ee8f1687bd9e2f39323a5da4 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7e24c468 | sha256-37bff2e180f7525b06527d272cd755cd9971316bcc2b72815fa3b4000c9757ed |

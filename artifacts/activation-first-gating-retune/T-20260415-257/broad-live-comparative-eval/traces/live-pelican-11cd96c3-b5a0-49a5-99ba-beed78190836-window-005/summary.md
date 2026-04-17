# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e98818ee1fbfe6af19470aba80e3474e972af078ccda49d0b283bf9b3f9cdf05`
- fixture hash: `sha256-d657d23463bf41cd4159e478f5223c1f2880e97d1b0706959b1e80d3f0d4e745`
- score hash: `sha256-47b706e7c4aad9451aa7e27456371cfd27716eaf21337810c813604ec0d731ae`
- bundle hash: `sha256-c1802fa18c9a859fc4546687f9a009d2f079f89091ec28a218bff4dd35ef4638`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eaf48763997381e6d9ac148445f4fc78050abde4363c03acd4f6f65040d7cf98 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-93b873ffabad051384e4057e49efb155c83a1e276c85c7ff43295010ab83e331 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-293a24ae714cfc6dc7e7b7f253119fad86b4e64875b19da95d22e8508f1b1673 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-c6405cb72d112f2c6e956ae7d65f93e082681f4c1273d097d0a12bf9a01b3210 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0d9e0498 | sha256-08148a93ed5e02ebe50637149c279e82ccdff51514c949058da0f418b3ba6c7e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0d9e0498 | sha256-9db6219743a9d4581b002e93f6ea91febe52b89f70700fb301b852383f1ef390 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d9129a15 | sha256-06d094842cc3564816475ee613dd66485f71c7f41d96a98e0ae5a2e7e2ecce70 |

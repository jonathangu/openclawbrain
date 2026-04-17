# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-55dd01ce-c43c-4b90-a4c6-c2fa97115709-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3e2ca93e26f85fd9a1adc63c2c9bbf2ce46e6fac1e384fb004554e3bdfbb894a`
- fixture hash: `sha256-2c882179454e5a495c5e21a4e1c041932e6e22fd8d004e6866e2895b395e2694`
- score hash: `sha256-ffd7ff5c00adfc819479068f9ee3fc99e2c86b3f0b4dd94ca46f6a15e94dad89`
- bundle hash: `sha256-ed459669b7e46e62438a7d58265c5c2192a9d6a70a2a4800301d7df43f3954a8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b6239dbac6a5fe5a4088922717f6aa5906be06e4cf4f4984972709cb77ebdbdf |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-033110223a450b6a49e86f186b394ec5095d93a835bc4f985793a7c31230287e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94858a9b686887fc4f86d47d3f9f40e4d6ecbaf795b8fa84e66a62d41aeb9821 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ebda3e93334b5189e49c11b13d5df9fe03b6882fdc30393996d7a71cdc459e4b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4a32248d | sha256-cc91b8bf9114e3874f4649b081b945ed1279ff15b2b6e9d50926fd077a69a9ae |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4a32248d | sha256-7f7dc36bb35ef0306a3b7b5216ec542e0e102fe0539022be2f5cdd5d42db59d3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d2dddf68 | sha256-5d4ec216235257f1093e1936bd9f2799c0aa4c4dc2892b411bbb286cc819a91d |

# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b202c1c438845d3c1c73ddb7c1ff7926a10fda7c3a64127ae541d469c9475d5`
- fixture hash: `sha256-b48968b0fefff768efffea4ced309b4343ca39a6dbbeda150f150e0d012ef675`
- score hash: `sha256-f0fd83604a76936e1bbbf6d7135a687c0c863bb181308508e3d816a97b9be58a`
- bundle hash: `sha256-b6cad2ff4eaac2ffa52525773f12f26bd4dcd14216a697e3fa9f1a82adc312cf`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-07841a59820286934b7db3a291f9a2a056f9291d9bd4bd106e744c3a6ac3c6f8 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec823ce338730d6e51217a4d17b6e5822719df942d6e1420da4d46e50fd9d253 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e5b2863bcb0e76a66ad51d5389c776a2435500cf937b73cdeac671e30c946388 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e2e5ab9f590354a7c4fd87467eccd1e3234bb6522815a7af1b2ee040d931279f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f2e65e90 | sha256-893cdbd329fa1f7b8edea534fcc694dba9c18fe4c514972430e669d1be96397e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f2e65e90 | sha256-cae3bb189ce7bd9f5a7c0964a385ed41bb81f0b44528efd7d720040e6fc9d5bb |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f216ba83 | sha256-fc0240fc6cd4f37eada0b33ec3b90450e16bb24c0d6f55c8563b5bf42a227c14 |

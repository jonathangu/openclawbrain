# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b9882f49bf30cb6d948087b310dd1f1c8c43cb51ebde7842866360d6db046b12`
- fixture hash: `sha256-371ddc3cfed0332b92f92e9c2b214fd34bd05f438837cc6562acfdd4c1e2c749`
- score hash: `sha256-a8452211b50bf69a59a3026b8139a623ce0f1188b8fd0d8c4ca3839352ba9015`
- bundle hash: `sha256-0efbc1bf9f2c3ef071112fbe23055584d21e38a5847f5c3e084b61f7d646f6d5`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d3958fc805f3776c38e6e687c85563bf09e68cc8dca03392a973d72cef995c7a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a0add9534836ab743f99e1a87b09c6d4e65f34a28d1f00ea0cb3483fbe681f7c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0756228898aef4636840a7004d94274c9ca7d4480354376665a526c84c89ede5 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-af848fcc8728f0bed83500e08108c48351d9419e007fc485b03c9851d38af119 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8ae3f938 | sha256-b179c55000982ff344010e6553d1e247e779f22394f0ca2fe1a932d6af041701 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8ae3f938 | sha256-77dc80b0c8977dee2b5cf281aef5bcc7b21b925fb7b9cb4a4fc345a705dd4330 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-cb0c5c71 | sha256-1742f29a0da40362be31f7b41b2f3ba9521c95eb22e6ccf698308bbcb18bb624 |

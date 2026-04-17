# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-3cb1eac8-82e4-48bf-8898-e9fa5bead77a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-79532f3b0ed4010e846f65618be48c2307af13b97dd581f294dd9d5e6325f8eb`
- fixture hash: `sha256-6cb2c0584a4478c43146057da013c5b788958c7763f4cd2e66653360656b5ed8`
- score hash: `sha256-d28b539d2766e0d622b40d15ee9d750406ad9600ad41678ae122828ac2061dec`
- bundle hash: `sha256-ba76c9aad395eb8904b724bfae052e51a97b6443d504cbfc91a648ea7dc53c62`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9133ee007dc762137868e0af1d2b1845ed239e3a386c6ccc9187f6b355e2ae22 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-146cb7e74c5cf321d1b4e2b4160f2f363c2a41c88c2f0e811191eb53190782d1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-170794d98cc22bcd94b3970648377c03b7c037fc467b8fd18ccb7dab2b943931 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c899d726bf823df7050cc444eafeff48ef51791a713fa5c804d996f7e221041e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2e266c84 | sha256-21a269e2714a278058d2dfc10b8d1890caa3e2c4d2918d791e8dcaab1ecde126 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2e266c84 | sha256-ada51c53f9676c5d5a436a684a2fd18b5644b7727adda1e343a0b9f80adb8b1a |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-55f31c25 | sha256-cc596292493246184bb561268384925dcbb141fd87b0072862bb13158b0fee50 |

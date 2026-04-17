# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c70b7e6acafa9f174da3df163120ba16044bc767e199909b1a7b96f75ed37549`
- fixture hash: `sha256-bf91f869d3956bf5fde31cf4fcbfa13c4356f4c344c72e681c59e051bd04b628`
- score hash: `sha256-a79bbd2605748839463665128fe4395901acfad4c5692158946e310d2be197e7`
- bundle hash: `sha256-19ef54ab03cb516ea54e602b5b1f741e7ac26f6b50d3e644708934e78464c468`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0b139f94f37d6885531ef5b31e5bde18e900dc87fd64f0c8059b9943917b139d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-66c1d90c364bd6a1f7f1109b4b28694c8dee704161814b10b55cc9c90bede01e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fa0184299e6ab993de3c8062bd5e303c18a84a363b5ee02140185b4c956df20 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-96f0aa07fb32dfe2cd2e4cf7b36d2965f8ac776fe37740d08b02ff958ec44252 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c7701014 | sha256-0bbc1b836c1139db9aead03473a1f663d109350399a626fd098c7b9487acadfa |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c7701014 | sha256-49fc8fb75dbe7edae6add59197e50f2e587eca20854781e16a5fc5e72be77080 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a15ec517 | sha256-7b60076a8e46b540918226c29625826c9a3e2eb53e45a5e5601712463818a949 |

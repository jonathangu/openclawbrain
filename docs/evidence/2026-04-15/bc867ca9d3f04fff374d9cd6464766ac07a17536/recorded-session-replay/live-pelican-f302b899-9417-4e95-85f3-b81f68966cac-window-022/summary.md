# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5d4b7ac6ed69712b1588ada5d64482dda6216ae5bbb670a70c4e5011448ae050`
- fixture hash: `sha256-c583a0a30dc7272198329e0ce06b64ff4fe39dce1f96b56a4f82e04f4a924ee7`
- score hash: `sha256-eece0b67c501a6ff2f734ef7e4da578b11aa2712b056f3b5339afaf23863c487`
- bundle hash: `sha256-8026cd00432f34b07479f813973d3fb592e837843009d81bd1ad26505134ef0b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-646908dce1c2aa715ec563720c445a9dc7233e215511f30956abcb8a6c0f9113 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bcff252b56cead44b2626a7e6e5ac0971f60e7e73e1eb9fd1bab1f2b41a2fce9 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec293a1f839247d8f6d90b036c62427461aae29094aa9dfd022a0feeb0869d39 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-cfc0f5701a53e3bc876dc24e324d457a7453c33f1392c872971203ae1a314d8c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ccfdaa0e | sha256-830b2942e1ce79819e48e321d1ca34c15eb0b2d69b257653dd5005b87ce203ee |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ccfdaa0e | sha256-8b140f07934601f7fabe0a4062dbe713e6b8465064437f9b43e8161d159847fc |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ccfdaa0e | sha256-830b2942e1ce79819e48e321d1ca34c15eb0b2d69b257653dd5005b87ce203ee |

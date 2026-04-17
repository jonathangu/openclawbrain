# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d855fd526b0432f6da4ae83a914585ce6467161a22fe45c628b20919e2994b08`
- fixture hash: `sha256-a059e9b8611b556f3c483b97168ab252147668d3316414532e38d0791f5cd0c4`
- score hash: `sha256-5a473921a7d9767e12c3cefe0955581b4c93e9b9c3a4c3285a2b7b4ffff0177c`
- bundle hash: `sha256-e5e0d0fb86bfae129331bd1e4b9a000505ac4fceddd68f4e9291688ff1611e99`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46808c7f90eba103441fec044b9224d9dea48b85cde7d0c53efec734a800db3f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-495b36a5036678dfe4195a81ed9b669708d9571239c50be5c903647913844a6d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d958b56839ca27d5afe98237c314a758395eaec7bd69ed00219e807569c355a4 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3091892c8f70cbd6744048e1e786eee2f4d2d1213bfe40cbd6f6e8790b44bf43 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5a0343a7 | sha256-2597fe900cf901f433c5408c0467b49520661db4e13ee46d42edb955a37e8868 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5a0343a7 | sha256-a0399ca50a860d79e2bbfaabdbe47824620f870c4958cb29bc36a83a6e742529 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b7e48778 | sha256-1948f8aa3f9bd591e44f369a96862a4cf33e471f54579305ba3918a0ae118172 |

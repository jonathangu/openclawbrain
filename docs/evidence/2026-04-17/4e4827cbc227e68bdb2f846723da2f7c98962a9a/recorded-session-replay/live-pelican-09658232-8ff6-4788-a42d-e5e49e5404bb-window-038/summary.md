# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-038`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb85648e6a75ae5a3ba6cc73943d9146864f7df475704ace40a5204fa142526`
- fixture hash: `sha256-7f13329cf1857fada1958a6fb5e614617a7842e6a6185ecb6a9d160264a3397f`
- score hash: `sha256-761ce2b584527069b285b026fb3d40a9037571dc9962fbcbaa580b6dd2f7b118`
- bundle hash: `sha256-bcb9d74f83527d1e9237c4824b5e8efaefaa476500458a397882e1e72212e824`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8dfb1b133ecf1f6249a4f0b0aded2fd5af80368793a13e3eb702ebb8c1e8fca |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7d59675ad80b01dc5e633c40b4930580402399609c41144ebcc484c6b2a2304c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3255f77dc71f5e8c3a79b588056f84164ca1d15ffca9eb0bc7ba706fd43e5ed1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c9ab00ca4491eee896027fd89cd692e4c5eecbc2a8a3bfe0ec159928109b9214 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0f02845b | sha256-0dfe1ddf5ff127884f91f049782ad027b9bbf4a2edc41d3aaf2e8dc46ff12437 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0f02845b | sha256-401e0b4021da81b316a1ec6c3bda17b2be2b9720c945fe47a4a397abfb0c014a |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-46a9d4a4 | sha256-9f67211d343b08a4f0bd790f089e6eab31414666832bfb127d5aa540a13eea65 |

# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-974c0caac77d24b74750a03083f2fe960327dbab94f044b1f352645b0c8977ef`
- fixture hash: `sha256-5332e2f75d9b84ce32dc4225385441cf4e1fdff1345733b1234e8eeb65449d9c`
- score hash: `sha256-06679f7c992bf6105887fa5971334fe6639b36cdfc3c229a9c62a871fa4863d0`
- bundle hash: `sha256-351d8a25aed317282ed3d76e85e28cf2c673f515c4eb5475aead840c41fd4e20`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-dbc3f90d1fd6323456b57dfe2268d1b7eda59d985039bca3aa485da45edbfc64 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2252031c202fa8c7364a9c97f0ff577a5f597f3f3eb57a9fb0fc9891ac15390e |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-65a7cfa275ceb5ddca6ff6c000a778aff6d07d75a0f459c436f766aad42485f2 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-fb4252f9650d8cd8378f69d5615846fb2d952175e6e3192ee1f86b3626f106f9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-175fbcc6 | sha256-53b9b8df2d50570ef0733768a42ad50928b03c0dc0fe68ce2a2acedb01d18922 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-175fbcc6 | sha256-a15864e428b7c0ceabc50d6b8f235466cc416ed4eaf66c48bbfabf85210ebf05 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-47382e43 | sha256-e37e3f0d0c8f55a6e91afcb53d636f1c4de256e820d11df99409682aa1f2475a |

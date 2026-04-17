# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70e107aa90463a0c77bc30d344eca5153707641920ed24320747bbd52e05a0e6`
- fixture hash: `sha256-2a5cd5afc4b09fa9beced059043152cd23fab3958640aae8275a1e91138ba120`
- score hash: `sha256-8e981659af62855b8d5ce037e7a8df19534c8ba32773463ffa85c271b54179a7`
- bundle hash: `sha256-3f5cecb5d393b7dfaa5adbe07e6514c1d13830caf731c8864928469c1a65e24e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c146c3106aa3a476acb28a8b075ba9caa0dc741d245d11ea00bbf3c4bbed6c9 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd5264d56fd7131cdc0abef121a381224536ff8fa898ce987651cd3ebf10ed00 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4c53e90ccabb60ab69a3e21564321b9117f9c34f00c6fdc06bc092c85de8a153 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-075507154c7eaef4de6b646b614971e9452f6305fdfb56593e6d9dadd3caf62e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-98cdc110 | sha256-6f01de24897669c6c1af58f1bf4c24c4a85606ae50570fede6b6b2b9ab8574da |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-98cdc110 | sha256-2f314bfd31fbdf7142803dbe51cc56e73e84f4939b77f967973655d39061f9d0 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-5e99f753 | sha256-f74baaec9e7bf67dd112b64e85de9562541e9c78bb2ae336925e03b3ba4f64d7 |

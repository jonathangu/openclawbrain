# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98f498b917834ee9c0a78d5b62a338d5c94ab2df87cb501ae8615cf42d07619a`
- fixture hash: `sha256-54eb8df766feda2c6211a7171b884e66a0008ed710f7d28bcb6341bc861e92a9`
- score hash: `sha256-3ab9a9a39e29426ab121b931396f0db3a36bb60383bd278890bda34a21fadc3e`
- bundle hash: `sha256-4c524cf90b3cbfd5935f129c7a08fcb12dc89b7ccc0a723bae5d08979038c3ec`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6670af2aca1f1e71cbf3c0f145ce7f96dddb89bb0330719aa7609642a8108f9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-338e451904562e406f9dc3843c889ef1f3b893b7404d6a10b40c57db6d655c26 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-858d3cb61cfcc287b9ccba455a8050e4d8d86b07155802cc737a8712bc6910ea |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-ca9bb636bfb4669e3f3ec11c3dd901e152ba84ef406f7eb148b0b5260dda20a3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1a1e1731 | sha256-5f38387e11039e1734ce5acab954ea32311af077ac9273194c22252af2b5581d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1a1e1731 | sha256-448b1f74841f42c1ee3c3af0e8fd8bf86c1d28ce901caf732bc0fec5d15d0b8b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d838a6e6 | sha256-4f99eb414080f9ab79cae9c081fca136692c1232fe7a084ee76bbb845af29f9d |

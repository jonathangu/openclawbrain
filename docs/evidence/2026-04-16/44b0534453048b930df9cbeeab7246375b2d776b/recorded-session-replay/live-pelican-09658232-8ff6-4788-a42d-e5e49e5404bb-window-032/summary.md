# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3d634932f1bce6d4693c067f779badfb747407b8de4f4dc108015f5fd2e78b`
- fixture hash: `sha256-178c9c975a3f9bee04b778ee3424e4eb908e1106cd7f867502edc61b1de425cc`
- score hash: `sha256-f9fd980a7e8d89b476da24b58b42a9932993f256fb3812e9ee7212047ec04303`
- bundle hash: `sha256-1032cf8fdd08c814ed704229c62685eb8fdf4bb985fe76ba7223572577262088`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-252786d108eae05056482d31aaad41cb1fd7abe9a8bca72a4a7a00c78ba84b59 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d94f52da0a25fede11b53493b78cc26953e74fd4435bdd833754718408e9afd6 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ea013a848feae1d63166ec94a8cd63692065738eb8abb3c51f5e7910e3eff31f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-2079adbb1386fb0a4223cf7cccaad0ef3d7db68315d6f39398df4a264f2762f0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b9e33772 | sha256-2837a04c096d3451cd0f7402f820644cfb6c7e439e104ceaf1f06c73d871c158 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b9e33772 | sha256-35cf770ee21e770dd35c6f0b92b6f16a0a525f6e622148c9001c8cda7c71d01c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-91707e51 | sha256-784ce95918e9a285a30d78998eb3ecb6df7c6954361f23daf313d07cb6a03db0 |
